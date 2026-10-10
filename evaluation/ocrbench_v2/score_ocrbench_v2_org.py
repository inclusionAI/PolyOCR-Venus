#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ocrbench_v21 离线打分脚本 (不需 vllm/torch/模型, 不需 gemini judge)。

直接对已推理好的预测结果 (xlsx) 用确定性指标打分:
  - 文本识别/OCR/精细识别 : 编辑距离 (ocr_edit_score)
  - VQA/多选/计数/数学     : 精确匹配 / vqa_evaluation / counting
  - 表格解析 / 图表解析    : TEDS
  - KIE / KIE-mapping      : F1
  - 定位 / spotting / 位置VQA : IoU / spotting
  - 文档解析 (en/cn)       : md2md 拆分编辑距离 (Edit_dist, 不需 TeX Live)
  - 公式识别 (en/cn)       : CDM F1 (需系统级 pdflatex/TeX Live + CJK; 缺失则置 0)
  - 手写简答 (cn)          : BLEU/METEOR/F1/Edit 平均 (需 nltk wordnet, 缺失 METEOR=0)

LLM-judge 子类:
  * "text translation cn": 传 --judge 时全量 judge; 不传时由 OCRBENCH_SKIP_LLM_JUDGE 置 None/ignore 跳过。
  * QA 类自由问答 (APP agent en / cognition VQA en·cn / diagram QA en / math QA en /
    reasoning VQA en·cn / science QA en): 传 --judge 后**默认开启** judge (rescue 模式):
    确定性规则分保底, 仅规则未满分的样本交 judge 复核, 最终分 = max(规则, judge);
    multiple choice 规则(抠字母整串相等)对思维链/<answer>输出必误杀, 同样进 rescue 复核;
    case sensitive 仍走确定性规则, 不进 judge。
    环境变量 OCRBENCH_QA_JUDGE: rescue(默认) | full(全部 judge 打分) | off(纯规则)。
    环境变量 OCRBENCH_QA_JUDGE_TYPES (逗号分隔): 只对列出的 QA 子类送 judge,
      未列出的 QA 子类退化为纯确定性规则 (即 vqa_evaluation / cn_vqa_evaluation / 多选抠字母,
      不再复核)。不设 -> 上述 8 个 QA 子类全量 judge (历史口径, 行为不变)。
      只认这 8 个名字; 列出 VQA with position en / text counting en 等无 judge 口径的子类会告警并忽略。
      例: OCRBENCH_QA_JUDGE_TYPES='reasoning VQA cn,science QA en'
  * 传 --judge/--judge-base-url/--judge-key: 建 OpenAI 兼容 judge (如
    antalchemy + gemini-3.5-flash-lite), 清空 OCRBENCH_SKIP_LLM_JUDGE 后真正评测。
  注: 必须用空串(非 '0') 解除跳过 —— process_predictions 里 os.environ.get 对任意非空串都判 truthy。
  注: 走 subset 重测链路时, build_qa_subset.py / 打分 / merge(finalize)_qa_rescore.py
      三步必须用同一份 OCRBENCH_QA_JUDGE_TYPES, 否则 merge 的一致性校验会 FATAL。

用法:
  python score_ocrbench_v2_org.py --pred <预测.xlsx> [--nproc 32] [--out-dir <dir>]
  python score_ocrbench_v2_org.py --auto [--judge gemini-3.5-flash-lite \
      --judge-base-url https://antalchemy.alipay.com/v1 --judge-key ak_xxx]   # auto 取最新预测并打分
  OCRBENCH_QA_JUDGE_TYPES='reasoning VQA cn,science QA en' python score_ocrbench_v2_org.py \
      --auto --judge gemini-3.5-flash-lite ...                                 # 只 judge 这 2 类
"""
import argparse
import ast
import glob
import json
import os
import re as _re
import sys
import time
from collections import defaultdict

# 必须在调用 process_predictions 之前设置: 跳过 LLM-judge 子类 "text translation cn"
os.environ.setdefault('OCRBENCH_SKIP_LLM_JUDGE', '1')

# --- NFS 降级兜底: 可选本地 overlay ---
# 当 共享存储 NFS 降级 (venv2 包导入卡 D-state) 时, 用本地 shadow 包 (跳过 vlmeval/dataset
# 整包导入链) + 本地安装的重依赖, 使得打分导入不再走慢/卡的 NFS。
# 正常 (NFS 健康) 时无需设置, 直接走 venv2 原路径即可。
_LOCAL_PKGS = os.environ.get('VLMEVAL_LOCAL_PKGS', '')
_SHADOW = os.environ.get('VLMEVAL_SHADOW_DIR', '')
if _LOCAL_PKGS and os.path.isdir(_LOCAL_PKGS):
    sys.path.insert(0, _LOCAL_PKGS)
if _SHADOW and os.path.isdir(_SHADOW):
    sys.path.insert(0, _SHADOW)
if os.environ.get('VLMEVAL_NO_TORCH', '') == '1':
    pass  # vlmeval/__init__ 据此跳过 torch; shadow 模式下 __init__ 本就空, 无 torch

# 多进程并行打分的模块级包装 (fork 后 worker 继承 process_predictions; _PP / _JUDGE 由 main 注入)
_PP = None
_JUDGE = None
_NPROC_FOR_CHUNK = 32
def _run_chunk(chunk):
    return _PP(chunk, judge_model=_JUDGE, nproc=_NPROC_FOR_CHUNK)


# ---- 跨进程 judge 全局限流 (antalchemy 租户 RPM=60/channel, GLINK) ----
# 多进程(multiprocessing fork)之间无法共享 threading.Lock; 用 flock 文件锁 + 状态文件
# 实现全局滑动窗口限流: 任意 60s 窗口内最多 RATE 次 judge 请求, 避免并发爆破 429。
import fcntl as _fcntl

_RL_LOCK = '/tmp/ocr_judge_ratelimit.lock'
_RL_STATE = '/tmp/ocr_judge_ratelimit.state'
_RATE_PER_MIN = int(os.environ.get('OCRBENCH_JUDGE_RPM', '55'))


# 匀速分发: 在 60s 窗口(<=RATE)之上, 再强制"请求间最小间隔 = 60/RATE 秒", 把原来
# "窗口一空就瞬间突发 50 个"改为"每节拍最多发 1 个" —— 消除瞬时突发, 避免 antchemy
# 网关 glink 理发 429 (实测 8 个突发即触发 429; 单调用 3.4s 本身很快)。
_MIN_INTERVAL = 60.0 / max(_RATE_PER_MIN, 1)

def _acquire_judge_slot(timeout=1800.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        wait_s = 0.05
        with open(_RL_LOCK, 'a+') as lf:
            _fcntl.flock(lf, _fcntl.LOCK_EX)
            try:
                now = time.time()
                stamps = []
                if os.path.exists(_RL_STATE):
                    with open(_RL_STATE) as f:
                        stamps = [float(x) for x in f.read().split()]
                stamps = [s for s in stamps if now - s < 60.0]
                last = max(stamps) if stamps else 0.0
                if now - last < _MIN_INTERVAL:
                    # 离上次发放不足一个节拍: 不发放, 等到下一节拍再试 (匀速)
                    wait_s = _MIN_INTERVAL - (now - last) + 0.03
                elif len(stamps) < _RATE_PER_MIN:
                    stamps.append(now)
                    with open(_RL_STATE, 'w') as f:
                        f.write('\n'.join(f'{t:.6f}' for t in stamps))
                    return
                else:
                    wait_s = 0.2  # 60s 窗口已满: 等旧 stamp 滑出窗口
            finally:
                _fcntl.flock(lf, _fcntl.LOCK_UN)
        time.sleep(min(wait_s, 1.0))
    raise RuntimeError('judge rate limit slot timeout (too many concurrent judges)')


def _release_judge_slot(ts):
    # 状态文件是追加式窗口, 无需显式释放; 保留钩子以防将来需要摘除特定时间戳
    pass


# ---- judge 结果断点缓存 (文本翻译 LLM-judge) ----
# 两次 judge 全程都被打断在中途 (2026-08-20 12:0x / 14:0x, 均未写出产物且每次从零重judge)。
# 增加 opt-in 缓存: 设 OCRBENCH_JUDGE_CACHE_DIR 即启用, 已成功返回的结果按 prompt hash 落盘,
# 重跑时复用, 实现"被打断->再启动续跑"而不重复烧配额。不设该环境变量时行为与原版完全一致。
# 注意: 只有成功(非 fail_msg)才入缓存; 失败结果不缓存, 便于续跑时重试。
import hashlib as _hashlib

_JUDGE_CACHE_DIR = os.environ.get('OCRBENCH_JUDGE_CACHE_DIR', '')
_JUDGE_CACHE_FILE = ''
_JUDGE_CACHE_LOCK = ''
if _JUDGE_CACHE_DIR:
    _JUDGE_CACHE_FILE = os.path.join(_JUDGE_CACHE_DIR, 'judge_cache.json')
    _JUDGE_CACHE_LOCK = os.path.join(_JUDGE_CACHE_DIR, 'judge_cache.lock')


def _judge_cache_load():
    if not _JUDGE_CACHE_FILE or not os.path.exists(_JUDGE_CACHE_FILE):
        return {}
    try:
        with open(_JUDGE_CACHE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _judge_cache_get(key):
    if not _JUDGE_CACHE_FILE:
        return None
    with open(_JUDGE_CACHE_LOCK, 'a+') as lf:
        _fcntl.flock(lf, _fcntl.LOCK_EX)
        try:
            cache = _judge_cache_load()
            val = cache.get(key)
        finally:
            _fcntl.flock(lf, _fcntl.LOCK_UN)
    return val


def _judge_cache_put(key, value):
    if not _JUDGE_CACHE_FILE:
        return
    os.makedirs(os.path.dirname(_JUDGE_CACHE_FILE) or '.', exist_ok=True)
    with open(_JUDGE_CACHE_LOCK, 'a+') as lf:
        _fcntl.flock(lf, _fcntl.LOCK_EX)
        try:
            cache = _judge_cache_load()
            cache[key] = value
            import tempfile
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(_JUDGE_CACHE_FILE) or '.')
            with os.fdopen(fd, 'w') as f:
                json.dump(cache, f, ensure_ascii=False)
            os.replace(tmp, _JUDGE_CACHE_FILE)
        finally:
            _fcntl.flock(lf, _fcntl.LOCK_UN)


class _OAICJudge:
    """最小 LLM-judge: 直接 POST OpenAI 兼容 /chat/completions。

    不依赖 vlmeval.api (导入它会拉起 34 个 api 模块, 有缺依赖即崩, 且对象可能不可 pickle / fork 死锁)。
    复刻 vlmeval gpt.py 对 gemini 系的 payload 形态 (Bearer + reasoning_effort=high; antalchemy 已验证可接受),
    metric 里 `judge_model.generate(prompt) -> str` 即可工作。
    """

    def __init__(self, model, api_base, key, retry=6, timeout=120, temperature=0, max_tokens=1024):
        self.model = model
        self.api_base = api_base  # 已含 /chat/completions
        self.key = key
        self.retry = retry
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fail_msg = 'Failed to obtain answer via API. '
        self.cur_idx = 0

    def generate(self, message, dataset=None, **kwargs):
        import requests, json as _json, time
        prompt = message if isinstance(message, str) else _json.dumps(message, ensure_ascii=False)
        # 断点缓存: 命中直接返回, 不占配额
        _key = _hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        _cached = _judge_cache_get(_key)
        if _cached is not None:
            return _cached
        payload = {
            'model': self.model,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': self.temperature,
        }
        # 对齐 vlmeval gpt.py 的 gemini 特例: 去掉 max_tokens/n, 加 reasoning_effort(antalchemy 可接受)
        if 'gemini' in self.model:
            payload.pop('max_tokens', None)
            payload['reasoning_effort'] = 'high'
        else:
            payload['max_tokens'] = self.max_tokens
        # charset=utf-8 + 字节体: requests 对 str body 默认按 latin-1 编码 -> 中文 judge 调用
        # 一律 UnicodeEncodeError; 显式 encode('utf-8') 字节 + header 声明 charset 即刻修掉。
        headers = {'Content-Type': 'application/json; charset=utf-8', 'Authorization': f'Bearer {self.key}'}
        last = None
        for attempt in range(max(1, self.retry)):
            # 全局限流: 保证并发下每分钟请求数不超租户 RPM 配额 (避免 429 爆破)
            _acquire_judge_slot()
            try:
                r = requests.post(
                    self.api_base, headers=headers, data=_json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                    timeout=self.timeout * 1.1, proxies=None)  # proxies=None: 不走环境代理/网关被代理拦截
                if 200 <= r.status_code < 300:
                    obj = _json.loads(r.text)
                    ans = obj['choices'][0]['message']['content']
                    _judge_cache_put(_key, ans)
                    return ans
                last = f'{r.status_code}:{r.text[:300]}'
                if r.status_code == 429:
                    # antalchemy 租户 RPM 限流(配额按分钟计); 慢速线性退避, 等窗口滑过再重试.
                    time.sleep(min(30.0, 5.0 * (attempt + 1)))
                    continue
            except Exception as e:
                last = f'{type(e).__name__}:{e}'
            time.sleep(0.5)
        print(f'[WARN] judge 调用失败 model={self.model} last={last}', flush=True)
        return self.fail_msg  # metric 的 _parse_score 对此返回 0 (best-effort)


def _safe_literal_eval(value):
    """与 image_vqa.py 同名实现保持一致: 处理 xlsx/csv 引号转义导致的 str 回退。"""
    raw = value if isinstance(value, str) else str(value)
    try:
        result = ast.literal_eval(raw)
    except Exception:
        result = raw
    if isinstance(result, (list, dict, tuple)):
        return result
    s = raw.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        inner = s[1:-1].replace('""', '"')
        try:
            recovered = ast.literal_eval(inner)
            if isinstance(recovered, (list, dict, tuple)):
                return recovered
        except Exception:
            pass
    return result


# ---- <answer>...</answer> 抽取 (配合 OCRBENCH_V2_ANSWER_FORMAT_TYPES) ----
# 提示词侧要求这些 QA 子类把最终答案放进 <answer>...</answer>; 打分侧对应只取块内内容
# 作为 predict, 其余标签文本一律丢弃。取**最后一个**块 (模型若先草稿后定稿, 定稿在后)。
# 未匹配到标签时原样返回 -> 对未加格式约束 / 未遵守格式的样本为 no-op, 不改变原口径。
DEFAULT_ANSWER_BOX_TYPES = (
    "cognition VQA cn,cognition VQA en,reasoning VQA cn,reasoning VQA en,"
    "math QA en,diagram QA en,science QA en"
)
_ANSWER_BOX_RE = _re.compile(r'<answer>(.*?)</answer>', _re.DOTALL)


def _answer_box_types():
    raw = os.environ.get('OCRBENCH_ANSWER_BOX_TYPES', DEFAULT_ANSWER_BOX_TYPES)
    return {t.strip() for t in raw.split(',') if t.strip()} if raw else set()


def _extract_answer_box(predict):
    """取 <answer>...</answer> 内文; 无标签则原样返回。"""
    if not isinstance(predict, str):
        return predict
    matches = _ANSWER_BOX_RE.findall(predict)
    if not matches:
        return predict
    return matches[-1].strip()


def auto_find_pred_xlsx(output_root):
    """在 output_root 下找最新 T*/ 的 *_ocrbench_v21.xlsx 预测文件。"""
    pattern = os.path.join(output_root, '**', '*_ocrbench_v21.xlsx')
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def build_predict_result(xlsx_path):
    """复刻 OCRBench_v2.evaluate: xlsx -> process_predictions 所需 dict 列表。"""
    import pandas as pd
    data = pd.read_excel(xlsx_path)
    has_orig = 'orig_idx' in data.columns
    answer_box_types = _answer_box_types()
    predict_result = []
    for i in range(len(data)):
        line = data.iloc[i]
        category = line['category']
        xlsx_idx = int(line['orig_idx']) if has_orig and pd.notna(line['orig_idx']) else i
        predict = str(line['prediction']) if pd.notna(line['prediction']) else ''
        # 指定 QA 子类: 只取 <answer>...</answer> 内文再打分 (规则 + rescue judge 均用抽取后的文本)
        if category in answer_box_types:
            predict = _extract_answer_box(predict)
        answers = _safe_literal_eval(line['answer'])
        questions = line['question']
        evals = line['eval']
        bbox_raw = line['bbox']
        content_raw = line['content']
        bbox = _safe_literal_eval(bbox_raw) if bbox_raw != 'without bbox' else bbox_raw
        content = _safe_literal_eval(content_raw) if content_raw != 'without content' else content_raw
        entry = {
            "__xlsx_idx": xlsx_idx,
            "type": category,
            "question": questions,
            "predict": predict,
            "answers": answers,
            "bbox": bbox,
            "content": content,
        }
        if pd.notna(evals) and evals != 'without eval':
            entry["eval"] = evals
        predict_result.append(entry)
    return predict_result, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', help='预测结果 xlsx 路径')
    ap.add_argument('--auto', action='store_true', help='自动取输出目录下最新预测 xlsx')
    ap.add_argument('--output-root',
                    default='./outputs/eval_qwen35_9b_ocrbench_v2_org_ckpt6998/eval_qwen35_9b_ocrbench_v2_org_ckpt6998',
                    help='--auto 时的搜索根目录')
    ap.add_argument('--nproc', type=int, default=32)
    ap.add_argument('--out-dir', help='结果输出目录 (默认与 xlsx 同目录)')
    # ---- 大模型评测 (text translation cn LLM-judge): OpenAI 兼容 judge, 默认不启用 ----
    ap.add_argument('--judge', default=None,
                    help='judge 模型名, 如 gemini-3.5-flash-lite; 提供即启用 LLM-judge 文本翻译子类')
    ap.add_argument('--judge-base-url', default=None, help='judge OpenAI 兼容 base url (如 https://antalchemy.alipay.com/v1)')
    ap.add_argument('--judge-key', default=None, help='judge API key')
    ap.add_argument('--judge-retry', type=int, default=6, help='judge 调用重试次数')
    ap.add_argument('--judge-timeout', type=int, default=120, help='单次 judge 调用超时(秒)')
    args = ap.parse_args()

    pred = args.pred
    if args.auto or not pred:
        pred = auto_find_pred_xlsx(args.output_root)
    if not pred or not os.path.isfile(pred):
        print(f'[FATAL] 找不到预测 xlsx: {pred}', file=sys.stderr)
        sys.exit(1)
    print(f'[INFO] 预测文件: {pred}')

    # ---- LLM-judge (text translation cn) 接入: 提供 --judge 时启用 OpenAI 兼容 judge ----
    judge_model = None
    if not args.judge:
        print(f'[INFO] OCRBENCH_SKIP_LLM_JUDGE={os.environ.get("OCRBENCH_SKIP_LLM_JUDGE")} '
              f'(未提供 --judge, text translation cn 将被跳过)')
    else:
        # 真正启用 LLM-judge: 覆盖模块顶部 setdefault 的 OCRBENCH_SKIP_LLM_JUDGE=1。
        # 注意: 必须置空串(''), 任何非空串(含 '0')在 os.environ.get 里都为 truthy -> 仍会跳过。
        os.environ['OCRBENCH_SKIP_LLM_JUDGE'] = ''
        print(f'[INFO] LLM-judge 已启用: model={args.judge} base_url={args.judge_base_url} '
              f'timeout={args.judge_timeout}s retry={args.judge_retry}; '
              f'OCRBENCH_SKIP_LLM_JUDGE 已清空 -> text translation cn 将调用 judge')
        judge_api_base = args.judge_base_url.rstrip('/') + '/chat/completions'
        judge_model = _OAICJudge(
            model=args.judge,
            api_base=judge_api_base,
            key=args.judge_key,
            retry=args.judge_retry,
            timeout=args.judge_timeout,
        )
        global _JUDGE
        _JUDGE = judge_model
        print(f'[INFO] judge 就绪: _OAICJudge(model={judge_model.model}, '
              f'api_base={judge_model.api_base})')

    # ---- CDM 依赖前置自检: 缺 TeX Live 等就 fail-fast 报错, 不静默把公式置 0 ----
    print('[INFO] CDM 依赖自检 (公式识别 / 文档解析公式子项) ...')
    try:
        import check_cdm_deps
        py_miss, tex_miss = check_cdm_deps.cdm_deps_status()
    except Exception as e:
        print(f'[WARN] 无法运行 CDM 依赖自检: {e}', file=sys.stderr)
        py_miss, tex_miss = [], []
    if py_miss or tex_miss:
        print('\n[FATAL] CDM 依赖缺失 —— 公式类无法正确评分, 拒绝静默降级。请先装齐再重跑:',
              file=sys.stderr)
        if py_miss:
            print('  python 缺: ' + ', '.join(py_miss), file=sys.stderr)
        if tex_miss:
            print('  TeX Live 缺: ' + ', '.join(tex_miss), file=sys.stderr)
            print('  -> 装 TeX Live 2025 (scheme-basic + collection-latexrecommended + '
                  'collection-latexextra + collection-langcjk + collection-langchinese),',
                  file=sys.stderr)
            print('     装到 .tools/texlive/2025 或设 CDM_TEXLIVE_ROOT 指向其根目录。',
                  file=sys.stderr)
        sys.exit(2)
    print('[INFO] CDM 依赖就绪 (pdflatex/CJK/gkai 可用)。')

    t0 = time.time()
    print('[INFO] 加载预测结果并构建打分输入 ...')
    predict_result, data = build_predict_result(pred)
    print(f'[INFO] 共 {len(predict_result)} 条, 耗时 {time.time()-t0:.1f}s')

    print('[INFO] 导入打分引擎 (ocrbrnch_v2_eval, 含 nltk/TEDS/CDM) ...')
    t1 = time.time()
    from vlmeval.dataset.utils.ocrbrnch_v2_eval import (
        process_predictions, ocrbench_v2_aggregate_accuracy)
    print(f'[INFO] 引擎导入耗时 {time.time()-t1:.1f}s')

    print('[INFO] 开始打分 ...')
    t2 = time.time()
    nproc = args.nproc
    # 分块多进程并行: CDM(pdflatex 子进程) + TEDS(纯 python) 都逐条独立;
    # 用 multiprocessing(fork 继承已导入模块, 各进程独立 GIL) 才能真并行, 线程会被 GIL 卡死。
    # 聚合只按 category 分组(与顺序无关), 故分块后直接拼接, 再按 xlsx 行号排序恢复原序。
    import os as _os
    n_chunks = int(_os.environ.get('OCRBENCH_PARALLEL_CHUNKS', str(nproc)))
    global _PP, _NPROC_FOR_CHUNK
    _PP = process_predictions
    _NPROC_FOR_CHUNK = nproc
    if n_chunks and n_chunks > 1 and len(predict_result) > n_chunks:
        import multiprocessing
        chunks = [predict_result[i::n_chunks] for i in range(n_chunks)]  # 轮询切块, 负载均衡
        with multiprocessing.Pool(processes=n_chunks) as pool:
            chunk_results = pool.map(_run_chunk, chunks)
        res_data_list = [item for sub in chunk_results for item in sub]
    else:
        res_data_list = process_predictions(predict_result, judge_model=judge_model, nproc=nproc)
    # 并行分块会乱序, 按原始 xlsx 行号恢复顺序(scored.jsonl 保持 xlsx 顺序, 便于后续按位置合并)
    res_data_list.sort(key=lambda x: x.get('__xlsx_idx', 0))
    print(f'[INFO] 打分完成, 耗时 {time.time()-t2:.1f}s')

    # ---- 按 category 统计 (排除 ignore / None) ----
    cat_scores = defaultdict(list)
    cat_none = defaultdict(int)
    cat_total = defaultdict(int)
    scored_rows = []
    for item in res_data_list:
        cat = item.get('type', 'UNKNOWN')
        cat_total[cat] += 1
        if item.get('ignore') == 'True' or item.get('score') is None:
            cat_none[cat] += 1
            scored_rows.append({'category': cat, 'score': None, 'skipped': True})
            continue
        cat_scores[cat].append(item['score'])
        scored_rows.append({'category': cat, 'score': item['score']})

    print('\n================ Sub-category Scores ================')
    print(f'{"Category":<42}{"Score":>9}{"Cnt":>7}{"Skip":>6}')
    print('-' * 64)
    subcat = {}
    for cat in sorted(cat_total.keys()):
        sc = cat_scores.get(cat, [])
        avg = (sum(sc) / len(sc)) if sc else 0.0
        subcat[cat] = round(avg, 6)
        print(f'{cat:<42}{avg*100:>8.2f}%{len(sc):>7}{cat_none.get(cat,0):>6}')
    print('-' * 64)

    # ---- en/cn 聚合 (与 evaluate 一致) ----
    en_scores, cn_scores = ocrbench_v2_aggregate_accuracy(res_data_list)
    final = {**en_scores, **cn_scores}
    if en_scores:
        final['English Overall Score'] = sum(en_scores.values()) / len(en_scores) if en_scores else 0
    if cn_scores:
        final['Chinese Overall Score'] = sum(cn_scores.values()) / len(cn_scores) if cn_scores else 0

    print('\n================ EN/CN Group Scores ================')
    for k, v in final.items():
        print(f'{k:<34}{(v*100 if isinstance(v,(int,float)) and v<=1 else v):>9}')

    # ---- 写出 ----
    out_dir = args.out_dir or os.path.dirname(pred)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(pred))[0]
    score_json = os.path.join(out_dir, base + '_score.json')
    subcat_json = os.path.join(out_dir, base + '_subcategory_score.json')
    scored_jsonl = os.path.join(out_dir, base + '_scored.jsonl')
    with open(score_json, 'w') as f:
        json.dump(final, f, indent=4, ensure_ascii=False)
    with open(subcat_json, 'w') as f:
        json.dump(subcat, f, indent=4, ensure_ascii=False)
    with open(scored_jsonl, 'w') as f:
        for r in scored_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'\n[OUT] score          : {score_json}')
    print(f'[OUT] subcategory    : {subcat_json}')
    print(f'[OUT] per-item scored: {scored_jsonl}')
    print(f'[INFO] 总耗时 {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()