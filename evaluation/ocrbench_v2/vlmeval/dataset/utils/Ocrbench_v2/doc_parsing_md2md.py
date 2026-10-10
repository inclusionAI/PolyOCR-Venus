# flake8: noqa
"""文档解析的 md2md 评测，完全对齐 OmniDocBench end2end 的 overall 公式。

OmniDocBench 的文档解析 overall 公式（vlmeval/dataset/OmniDocBench/omnidocbench.py
:408-413）是：

    overall = mean( Edit_dist(text_block),
                    Edit_dist(display_formula),
                    Edit_dist(table),
                    Edit_dist(reading_order) )

即：先把页面拆成 纯文本 / 公式 / 表格 等元素，每类各自用 OmniDocBench 的 `Edit_dist`
（元素级匹配 + 页级 Σedit / Σmax_len）打分，再对这些类的 Edit_dist 取平均。
CDM（公式）/ TEDS（表格）在 OmniDocBench 里是**单列**指标，并不计入 overall。

本模块按此公式计算，最终返回**单个总分**。为适配 OCRBench_v2「分越高越好」的口径
（其余子类如 chart/table/formula 都是越高越好），这里把 Edit_dist 取反成准确率：

    score = 1 - mean( 各类 Edit_dist )

完全等价于「各类准确率 (1 - Edit_dist) 取平均」。

与 OmniDocBench 的差异 / 取舍：
  - OmniDocBench 的 GT 是结构化标注 (page_info / 元素类别)，OCRBench_v2 的 document
    parsing GT 只是一段 markdown 字符串，没有结构化标注。因此这里对 **pred 和 GT 两端都跑
    md_tex_filter** 各自拆元素，再用 match_gt2pred_simple 做元素级对齐（OmniDocBench 的
    simple_match 口径），/page 级 Edit_dist 用 sum(edit)/sum(max_len) 复刻 call_Edit_dist。
  - reading_order 不单独成类：plain-markdown GT 没有独立的阅读顺序标注，顺序已隐含在文本
    元素序列里；按用户口径只算 纯文本 / 公式 / 表格 三类。
  - 只对「GT 侧存在该类元素」的类别计入平均（GT 没有该类元素则该类不计）。

依赖：只需 OmniDocBench.utils 的 md_tex_filter / match_gt2pred_simple / normalized_formula
（直接依赖 Levenshtein/bs4/pylatexenc/scipy/numpy）。**不**再依赖 cdm / apted / latexmlc。
任何一层依赖缺失或解析异常都逐级退化，最终退回旧的整段归一化编辑距离，保证评测不中断。
"""


# ----------------------------- 基础工具 -----------------------------

def _levenshtein(a, b):
    """字符级 Levenshtein 距离，优先用 C 实现的 Levenshtein，缺失则退到 nltk / 纯 python。"""
    try:
        import Levenshtein
        return Levenshtein.distance(a, b)
    except Exception:
        pass
    try:
        import nltk
        return nltk.edit_distance(a, b)
    except Exception:
        pass
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _edit_score(pred, gt):
    """归一化编辑距离准确率：1 - edit / max(len)，对齐 OmniDocBench Edit_dist 归一化口径。"""
    if not isinstance(pred, str):
        pred = '' if pred is None else str(pred)
    if not isinstance(gt, str):
        gt = '' if gt is None else str(gt)
    up = max(len(pred), len(gt))
    if up == 0:
        return 1.0
    return max(0.0, 1.0 - float(_levenshtein(pred, gt)) / float(up))


# ----------------------------- OmniDocBench utils 惰性加载 -----------------------------

_OMNIDOC_UTILS = None
_OMNIDOC_LOAD_ERR = None


def _load_omnidoc_utils():
    """惰性加载 OmniDocBench.utils 的 md_tex_filter / match_gt2pred_simple / normalized_formula。

    返回 (md_tex_filter, match_gt2pred_simple, normalized_formula)，失败返回 None。

    优先按文件路径用 importlib 加载 utils.py，绕开 vlmeval.dataset.__init__ 的整包导入链
    （那条链会拉着 num2words 等与文档解析无关依赖，环境缺一个就把整页 md2md 退化掉）。
    utils.py 无相对导入、不强依赖 num2words，只需 Levenshtein/bs4/pylatexenc/scipy/numpy。
    文件路径加载失败再回退到常规包导入。
    """
    global _OMNIDOC_UTILS, _OMNIDOC_LOAD_ERR
    if _OMNIDOC_UTILS is not None:
        return _OMNIDOC_UTILS
    if _OMNIDOC_LOAD_ERR is not None:
        return None

    wanted = ('md_tex_filter', 'match_gt2pred_simple', 'normalized_formula')

    # 1) 按文件路径加载（绕开整包 __init__ 链）
    try:
        import importlib.util
        import os
        utils_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'OmniDocBench', 'utils.py'))
        spec = importlib.util.spec_from_file_location(
            'vlmeval_dataset_omnidocbench_utils_standalone', utils_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _OMNIDOC_UTILS = tuple(getattr(mod, n) for n in wanted)
        return _OMNIDOC_UTILS
    except Exception as e_path:
        # 2) 回退到常规包导入
        try:
            from vlmeval.dataset.OmniDocBench.utils import (
                md_tex_filter, match_gt2pred_simple, normalized_formula)
            _OMNIDOC_UTILS = (md_tex_filter, match_gt2pred_simple, normalized_formula)
            return _OMNIDOC_UTILS
        except Exception as e_pkg:
            _OMNIDOC_LOAD_ERR = (e_path, e_pkg)
            if not getattr(_load_omnidoc_utils, '_warned', False):
                print(f'[doc_parsing_md2md] OmniDocBench utils 加载失败，将退化为整段编辑距离: '
                      f'path={e_path} pkg={e_pkg}')
                _load_omnidoc_utils._warned = True
            return None


# ----------------------------- 单类 Edit_dist 打分 -----------------------------

def _category_items(ds, *keys):
    items = []
    for k in keys:
        items.extend(ds.get(k, []) or [])
    return items


def _score_edit_category(gt_items, pred_items, line_type, img_name='doc'):
    """OmniDocBench call_Edit_dist 的页级口径，返回准确率 = 1 - (Σedit / Σmax_len)。

    - 用 match_gt2pred_simple 按 line_type 做元素级匹配，拿到每对的 norm_gt / norm_pred
      （text: clean+unicode；formula: normalized_formula；table: 原始 content）。
    - 页级聚合 Σlevenshtein(pred,gt) / Σmax(len(pred),len(gt))，复刻 call_Edit_dist。
    - 取反为准确率，保证越高越好。
    """
    _, match_gt2pred_simple, _ = _load_omnidoc_utils()
    matched = match_gt2pred_simple(gt_items, pred_items, line_type, img_name)
    total_edit = 0
    total_upper = 0
    for s in matched:
        gt = s.get('norm_gt') or s.get('gt') or ''
        pred = s.get('norm_pred') or s.get('pred') or ''
        gt = gt if isinstance(gt, str) else str(gt)
        pred = pred if isinstance(pred, str) else str(pred)
        up = max(len(pred), len(gt))
        if up == 0:
            continue
        total_edit += _levenshtein(pred, gt)
        total_upper += up
    if total_upper == 0:
        return 0.0
    return max(0.0, 1.0 - float(total_edit) / float(total_upper))


# ----------------------------- 公式(CDM) / 表格(TEDS) 子项 -----------------------------

def _score_cdm_formula_category(gt_items, pred_items, img_name='doc'):
    """公式子项走 CDM(对齐 OmniDocBench md2md): match_gt2pred_simple 匹配后,
    每对算 formula_cdm_score(pred, [gt]) 取 F1, 再平均。
    需系统级 pdflatex (TeX Live); 缺失时 formula_cdm_score 内部置 0 -> 公式子项=0。"""
    utils = _load_omnidoc_utils()
    if utils is None:
        return 0.0
    _, match_gt2pred_simple, _ = utils
    from vlmeval.dataset.utils.Ocrbench_v2.vqa_metric import formula_cdm_score
    matched = match_gt2pred_simple(gt_items, pred_items, 'formula', img_name)
    fs = []
    for s in matched:
        gt = s.get('gt') or ''
        if not gt:
            continue
        pred = s.get('pred') or ''
        try:
            fs.append(float(formula_cdm_score(pred, [gt])))
        except Exception:
            fs.append(0.0)
    return sum(fs) / len(fs) if fs else 0.0


def _score_teds_table_category(gt_items, pred_items, img_name='doc'):
    """表格子项走 TEDS(对齐 OmniDocBench md2md): 匹配后每对 wrap_html_table + teds.evaluate;
    latex 表/异常对降级编辑距离, 避免被拖到 0。"""
    utils = _load_omnidoc_utils()
    if utils is None:
        return 0.0
    _, match_gt2pred_simple, _ = utils
    from vlmeval.dataset.utils.Ocrbench_v2.TEDS_metric import TEDS, wrap_html_table
    teds = TEDS(n_jobs=1)
    matched = match_gt2pred_simple(gt_items, pred_items, 'html_table', img_name)
    ts = []
    for s in matched:
        gt = s.get('gt') or ''
        if not gt:
            continue
        pred = s.get('pred') or ''
        try:
            ts.append(float(teds.evaluate(wrap_html_table(pred), wrap_html_table(gt))))
        except Exception:
            ts.append(_edit_score(pred, gt))  # 降级(如 latex 表)
    return sum(ts) / len(ts) if ts else 0.0


# ----------------------------- 主入口 -----------------------------

def doc_parsing_md2md_score(pred, gt):
    """对齐 OmniDocBench md2md 的文档解析打分，返回单个总分。

    pred / gt : 一段 markdown 字串。
    三类子项分别打分后取平均(只对「GT 侧存在该类元素」的类别计入平均):
      - 纯文本  : Edit_dist (编辑距离)
      - 公式    : CDM       (渲染成图比 F1; 需系统级 pdflatex/TeX Live, 缺失则子项=0)
      - 表格    : TEDS      (HTML 树编辑距离; latex 表/异常降级编辑距离)
    任何依赖缺失/解析异常逐级退化到整段编辑距离。
    """
    if not isinstance(pred, str) or not isinstance(gt, str):
        return 0.0

    utils = _load_omnidoc_utils()
    if utils is None:
        return _fallback_whole_edit(pred, gt)
    md_tex_filter = utils[0]

    try:
        pred_ds = md_tex_filter(pred)
        gt_ds = md_tex_filter(gt)
    except Exception as e:
        if not getattr(doc_parsing_md2md_score, '_warn_filter', False):
            print(f'[doc_parsing_md2md] md_tex_filter 解析失败，退化为整段编辑距离: {e}')
            doc_parsing_md2md_score._warn_filter = True
        return _fallback_whole_edit(pred, gt)

    scores = []
    # 纯文本
    gt_text = _category_items(gt_ds, 'text_all')
    if gt_text:
        try:
            scores.append(_score_edit_category(gt_text, _category_items(pred_ds, 'text_all'), 'text'))
        except Exception as e:
            if not getattr(doc_parsing_md2md_score, '_warn_text', False):
                print(f'[doc_parsing_md2md] 文本类别打分异常，跳过: {e}')
                doc_parsing_md2md_score._warn_text = True
    # 公式 -> CDM (OmniDocBench md2md 口径; 需 TeX Live, 缺失则置 0)
    gt_formula = _category_items(gt_ds, 'equation_isolated')
    if gt_formula:
        try:
            scores.append(_score_cdm_formula_category(
                gt_formula, _category_items(pred_ds, 'equation_isolated')))
        except Exception as e:
            if not getattr(doc_parsing_md2md_score, '_warn_formula', False):
                print(f'[doc_parsing_md2md] 公式类别(CDM)打分异常，跳过: {e}')
                doc_parsing_md2md_score._warn_formula = True
    # 表格 -> TEDS (OmniDocBench md2md 口径; latex 表降级编辑距离)
    gt_table = _category_items(gt_ds, 'html_table', 'latex_table')
    if gt_table:
        try:
            scores.append(_score_teds_table_category(
                gt_table, _category_items(pred_ds, 'html_table', 'latex_table')))
        except Exception as e:
            if not getattr(doc_parsing_md2md_score, '_warn_table', False):
                print(f'[doc_parsing_md2md] 表格类别(TEDS)打分异常，跳过: {e}')
                doc_parsing_md2md_score._warn_table = True

    if not scores:
        # GT 无可拆元素 -> 整段对比
        return _fallback_whole_edit(pred, gt)
    return float(sum(scores) / len(scores))


def _fallback_whole_edit(pred, gt):
    """退化为旧的整段 markdown 归一化编辑距离。"""
    try:
        from vlmeval.dataset.utils.Ocrbench_v2.TEDS_metric import pre_clean
        try:
            pred = pre_clean(pred)
            gt = pre_clean(gt)
        except Exception:
            pass
    except Exception:
        pass
    return _edit_score(pred, gt)


if __name__ == '__main__':
    cases = [
        ("# Title\n\nHello world.\n\nSecond paragraph.",
         "# Title\n\nHello world.\n\nSecond paragraph."),
        ("Hello world", "Hello worlz"),
        ("text A\n\n$$E=mc^2$$", "text A\n\n$$E=mc^2$$"),
        ("text A\n\n$$E=mc^2$$", "text A\n\n$$E=mc^3$$"),
    ]
    for p, g in cases:
        print('score=', round(doc_parsing_md2md_score(p, g), 4))