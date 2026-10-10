#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总受 LLM-judge 口径影响子类的重测结果 -> 每个模型的完整分数 + 多模型对比表。

流程 (与 merge_qa_rescore.py 同口径):
  子集重测 scored.jsonl (受影响子类新分) + 全量参考 scored.jsonl (其余确定性分)
    -> 逐条合并 -> ocrbench_v2_aggregate_accuracy -> 完整 score/subcategory/scored

受影响子类 (AFFECTED) 与打分口径同源, 默认 9 类 (8 个 QA 子类 + text translation cn),
可由 OCRBENCH_QA_JUDGE_TYPES 收窄 (如只留 reasoning VQA cn / science QA en) 或 --types 指定。

产出:
  <out-dir>/<label>_final_score.json           每个模型的完整 EN/CN 分组分
  <out-dir>/<label>_final_subcategory_score.json
  <out-dir>/<label>_final_scored.jsonl
  <out-dir>/COMPARISON_qa_rescore.md           多模型对比表 (子类级 + 分组级 + Overall)
  <out-dir>/COMPARISON_qa_rescore.xlsx         同上, Excel

用法:
  python finalize_qa_rescore.py --out-dir <dir> \
      --model "标签A=<全量参考scored.jsonl>,<子集重测scored.jsonl>" \
      --model "标签B=<全量参考scored.jsonl>,<子集重测scored.jsonl>" [更多 --model ...]
  # 收窄口径时 (需与 build_qa_subset / 打分步骤一致):
  OCRBENCH_QA_JUDGE_TYPES="reasoning VQA cn,science QA en" python finalize_qa_rescore.py ...
"""
from pathlib import Path as _EvalPath
import os as _eval_os
_EVAL_ROOT = _EvalPath(_eval_os.environ.get("VLMEVAL_ROOT", str(_EvalPath(__file__).resolve().parents[0]))).resolve()

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, str(_EVAL_ROOT / '.'))
os.environ.setdefault('VLMEVAL_NO_TORCH', '1')
from vlmeval.dataset.utils.ocrbrnch_v2_eval import (QA_TEXT_TRANSLATION, _QA_JUDGE_TYPES,
                                                    ocrbench_v2_aggregate_accuracy,
                                                    qa_rescore_affected_types)

# 全部可进 judge 的 QA 子类 (8 类); 与本次实际生效的 AFFECTED 对比后写进报告,
# 收窄口径时让读者看清"哪些 QA 类这次根本没 judge, 仍是纯规则分"。
ALL_QA_TYPES = sorted(_QA_JUDGE_TYPES)
# 历史默认: 8 个 QA 子类 + 翻译子类 (与打分内核同源, 不再各写一份)。
# 实际值由 resolve_affected() 按 OCRBENCH_QA_JUDGE_TYPES 收窄。
AFFECTED_DEFAULT = ALL_QA_TYPES + [QA_TEXT_TRANSLATION]
# 展示顺序: 与 OCRBench_v2 官方子类顺序一致 (en 在前, cn 在后)
CAT_ORDER = [
    "full-page OCR en", "text recognition en", "fine-grained text recognition en",
    "text spotting en", "text grounding en", "text counting en",
    "key information extraction en", "key information mapping en",
    "document classification en", "ASCII art classification en",
    "table parsing en", "chart parsing en", "document parsing en", "formula recognition en",
    "handwritten answer extraction cn", "full-page OCR cn", "key information extraction cn",
    "table parsing cn", "document parsing cn", "formula recognition cn",
    "text translation cn", "cognition VQA en", "cognition VQA cn",
    "reasoning VQA en", "reasoning VQA cn", "diagram QA en", "math QA en",
    "science QA en", "APP agent en", "VQA with position en",
]
# 本次生效的受影响子类 (main 里由 resolve_affected 赋值; 各 helper 共用)
AFFECTED = list(AFFECTED_DEFAULT)


def resolve_affected(cli_types=None):
    """--types > 共享解析 (读 OCRBENCH_QA_JUDGE_TYPES 等环境变量) > 历史默认 9 类。

    返回按 CAT_ORDER 排好的列表, 保证 md/xlsx 里 ★ 标记与行序稳定。
    必须与 build_qa_subset.py / merge_qa_rescore.py 用同一份范围, 否则 merge()
    的"subset 子类越界"或"逐类条数一致"校验会 FATAL。
    """
    if cli_types:
        types_ = {t.strip() for t in cli_types.split(',') if t.strip()}
    else:
        types_ = qa_rescore_affected_types() or set(AFFECTED_DEFAULT)
    ordered = [c for c in CAT_ORDER if c in types_]
    ordered += sorted(types_ - set(CAT_ORDER))    # CAT_ORDER 没覆盖的名字排后面, 不丢
    return ordered
GROUP_ORDER = ["en_text_recognition", "en_text_detection", "en_text_spotting",
               "en_relationship_extraction", "en_element_parsing",
               "en_mathematical_calculation", "en_visual_text_understanding",
               "en_knowledge_reasoning", "cn_text_recognition",
               "cn_relationship_extraction", "cn_element_parsing",
               "cn_visual_text_understanding", "cn_knowledge_reasoning",
               "English Overall Score", "Chinese Overall Score"]


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cat_means(rows):
    d = defaultdict(list)
    for r in rows:
        if r.get('score') is not None:
            d[r['category']].append(r['score'])
    return {k: sum(v) / len(v) for k, v in d.items()}, {k: len(v) for k, v in d.items()}


def aggregate(rows):
    """与 score_ocrbench_v2_org.py / merge_qa_rescore.py 完全一致的聚合。"""
    data_list = [{'type': r['category'], 'score': r['score']} for r in rows
                 if r.get('score') is not None]
    en, cn = ocrbench_v2_aggregate_accuracy(data_list)
    final = {**en, **cn}
    if en:
        final['English Overall Score'] = sum(en.values()) / len(en)
    if cn:
        final['Chinese Overall Score'] = sum(cn.values()) / len(cn)
    return final


def merge(ref_path, sub_path):
    """确定性子类沿用全量参考, AFFECTED 列出的受影响子类用重测子集。含一致性校验。"""
    ref, sub = load_jsonl(ref_path), load_jsonl(sub_path)
    for name, rows in (('reference', ref), ('subset', sub)):
        bad = [r for r in rows if r.get('score') is None]
        if bad:
            raise SystemExit(f'[FATAL] {name} ({ref_path if name == "reference" else sub_path}) '
                             f'有 {len(bad)} 条 score 为 None, 首个: {bad[0]}')
    extra = {r['category'] for r in sub} - set(AFFECTED)
    if extra:
        raise SystemExit(f'[FATAL] subset 含非受影响子类: {sorted(extra)}\n'
                         f'        本次 AFFECTED 只有 {len(AFFECTED)} 类: {AFFECTED}\n'
                         f'        build_qa_subset.py 与 finalize 的 OCRBENCH_QA_JUDGE_TYPES '
                         f'必须一致 (或都传 --types)。')
    ref_cnt, sub_cnt = defaultdict(int), defaultdict(int)
    for r in ref:
        ref_cnt[r['category']] += 1
    for r in sub:
        sub_cnt[r['category']] += 1
    for cat in AFFECTED:
        if ref_cnt.get(cat, 0) != sub_cnt.get(cat, 0):
            raise SystemExit(f'[FATAL] 子类 {cat} 条数不一致: reference={ref_cnt.get(cat,0)} '
                             f'subset={sub_cnt.get(cat,0)}\n'
                             f'        subset 抽取范围比本次 AFFECTED 窄? 两步用同一份 '
                             f'OCRBENCH_QA_JUDGE_TYPES / --types。')
    merged = [r for r in ref if r['category'] not in AFFECTED] + sub
    assert len(merged) == len(ref), f'合并后 {len(merged)} != reference {len(ref)}'
    return merged, ref


def pct(v, digits=2):
    return f'{v*100:.{digits}f}%' if isinstance(v, (int, float)) else '-'


def main():
    global AFFECTED
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', action='append', required=True,
                    help='"标签=<全量参考scored.jsonl>,<子集重测scored.jsonl>"')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--title', default='ocrbench_v21 VQA/翻译子类 LLM-judge 重测汇总')
    ap.add_argument('--judge-info', default='(未填写)', help='judge 模型/端点/口径描述, 写进汇总表头')
    ap.add_argument('--types', default=None,
                    help='逗号分隔的受影响子类白名单, 显式覆盖 OCRBENCH_QA_JUDGE_TYPES 解析结果')
    args = ap.parse_args()
    AFFECTED = resolve_affected(args.types)
    print(f'[INFO] 受影响子类 ({len(AFFECTED)} 类): {AFFECTED}')
    print(f'[INFO] 来源: {"--types" if args.types else "环境变量 (OCRBENCH_QA_JUDGE_TYPES 等)"}')
    os.makedirs(args.out_dir, exist_ok=True)

    models = []
    for spec in args.model:
        label, rest = spec.split('=', 1)
        ref_path, sub_path = rest.rsplit(',', 1)
        merged, ref = merge(ref_path.strip(), sub_path.strip())
        final = aggregate(merged)
        cat_now, cnt = cat_means(merged)
        cat_old, _ = cat_means(ref)
        label = label.strip()
        models.append(dict(label=label, final=final, cat=cat_now, cat_old=cat_old,
                           cnt=cnt, merged=merged, ref=ref,
                           ref_path=ref_path.strip(), sub_path=sub_path.strip()))
        base = os.path.join(args.out_dir, f'{label}_final')
        with open(base + '_score.json', 'w') as f:
            json.dump(final, f, indent=4, ensure_ascii=False)
        with open(base + '_subcategory_score.json', 'w') as f:
            json.dump({c: round(cat_now[c], 6) for c in sorted(cat_now)}, f, indent=4, ensure_ascii=False)
        with open(base + '_scored.jsonl', 'w') as f:
            for r in merged:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f'[OUT] {base}_score.json  '
              f'EN={pct(final.get("English Overall Score"))} CN={pct(final.get("Chinese Overall Score"))}')

    labels = [m['label'] for m in models]
    md = [f'# {args.title}', '']
    md.append(f'- 参与模型: {", ".join(labels)}')
    md.append(f'- judge: {args.judge_info}')
    md.append(f'- 重测子类 ({len(AFFECTED)} 类): {", ".join(AFFECTED)}')
    md.append(f'- 其余确定性子类分数沿用各自全量参考 scored.jsonl (已验证与当前引擎逐类一致)')
    md.append('')

    # ---- 分组级 ----
    md.append('## 一、EN/CN 分组汇总')
    md.append('')
    md.append('| 指标 | ' + ' | '.join(labels) + ' |')
    md.append('|' + '---|' * (len(labels) + 1))
    for g in GROUP_ORDER:
        md.append(f'| {g} | ' + ' | '.join(pct(m['final'].get(g)) for m in models) + ' |')
    both = [(m['final'].get('English Overall Score'), m['final'].get('Chinese Overall Score')) for m in models]
    md.append('| **EN/CN 均值 (综合)** | ' + ' | '.join(
        f'**{pct((e+c)/2)}**' if e is not None and c is not None else '-' for e, c in both) + ' |')
    md.append('')

    # ---- 子类级 ----
    md.append('## 二、子类明细 (★=本次 judge 重测的子类; "旧"=存档参考分, "新"=judge 重测分)')
    md.append('')
    head = ['子类', 'n']
    for lb in labels:
        head += [f'{lb} 旧(存档)', f'{lb} 新(judge)']
    md.append('| ' + ' | '.join(head) + ' |')
    md.append('|' + '---|' * len(head))
    all_cats = list(CAT_ORDER) + [c for m in models for c in m['cat'] if c not in CAT_ORDER]
    seen = set()
    for c in all_cats:
        if c in seen:
            continue
        seen.add(c)
        star = ' ★' if c in AFFECTED else ''
        n = max((m['cnt'].get(c, 0) for m in models), default=0)
        cells = [f'{c}{star}', str(n)]
        for m in models:
            old, new = m['cat_old'].get(c), m['cat'].get(c)
            if c in AFFECTED:
                diff = (f' ({(new-old)*100:+.2f})' if new is not None and old is not None else '')
            else:
                diff = ''
            cells += [pct(old), f'{pct(new)}{diff}' if c in AFFECTED else pct(new)]
        md.append('| ' + ' | '.join(cells) + ' |')
    md.append('')
    qa_judged = [c for c in AFFECTED if c != 'text translation cn']
    rule_only = sorted(set(ALL_QA_TYPES) - set(qa_judged))
    md.append('> ★ = 本次用 LLM-judge 重测的子类; 括号内为相对存档分的增减(百分点)。')
    md.append(f'> "旧(存档)"口径: QA {len(ALL_QA_TYPES)} 类 = 纯确定性规则分(存档 run 未跑 rescue judge);'
              ' text translation cn = 存档 run 的 gemini-3.5-flash-lite judge 分(该类无规则分可依)。')
    if rule_only:
        md.append(f'> 本次仅 {len(qa_judged)} 个 QA 子类进 judge '
                  f'({", ".join(qa_judged)}); 其余 {len(rule_only)} 类 '
                  f'({", ".join(rule_only)}) 保持纯规则分, 不受本次重测影响。')
    md.append(f'> QA {len(qa_judged)} 类为 rescue 口径 `max(规则, judge)`, 只升不降; '
              'text translation cn 为 judge 全量口径, 可升可降。')
    md.append('')

    md_path = os.path.join(args.out_dir, 'COMPARISON_qa_rescore.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md) + '\n')
    print(f'[OUT] {md_path}')

    # ---- Excel ----
    try:
        import pandas as pd
        rows = []
        for c in sorted(seen):
            row = {'子类': c, 'judge重测': 'Y' if c in AFFECTED else 'N'}
            for m in models:
                row[f'{m["label"]} 旧(存档)'] = m['cat_old'].get(c)
                row[f'{m["label"]} 新(judge)'] = m['cat'].get(c)
            rows.append(row)
        for g in GROUP_ORDER + ['EN/CN 均值 (综合)']:
            row = {'子类': g, 'judge重测': '-'}
            for m in models:
                if g == 'EN/CN 均值 (综合)':
                    e, c2 = m['final'].get('English Overall Score'), m['final'].get('Chinese Overall Score')
                    row[f'{m["label"]} 新(judge)'] = (e + c2) / 2 if e is not None and c2 is not None else None
                else:
                    row[f'{m["label"]} 新(judge)'] = m['final'].get(g)
            rows.append(row)
        df = pd.DataFrame(rows)
        xlsx = os.path.join(args.out_dir, 'COMPARISON_qa_rescore.xlsx')
        df.to_excel(xlsx, index=False)
        print(f'[OUT] {xlsx}')
    except Exception as e:
        print(f'[WARN] Excel 导出失败: {e}')


if __name__ == '__main__':
    main()
