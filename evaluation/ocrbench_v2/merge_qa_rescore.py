#!/usr/bin/env python3
"""把受 LLM-judge 口径影响的子类的新分数, 与参考全量 run 的其余确定性子类逐条合并,
用 ocrbench_v2_aggregate_accuracy 重新聚合出完整 score.json / subcategory_score.json。

AFFECTED 与打分口径同源 (qa_rescore_affected_types, 受 OCRBENCH_QA_JUDGE_TYPES 收窄):
默认 9 类 = 8 个 QA 子类 + text translation cn。用 --types 可显式覆盖。
必须与 build_qa_subset.py / score_ocrbench_v2_org.py 用同一份范围, 否则下面的
"subset 子类越界" 或 "逐类条数不一致" 校验会 FATAL。

用法:
  python merge_qa_rescore.py --reference <全量run_scored.jsonl> \
      --subset <子集run_scored.jsonl> --out-prefix <输出前缀>
  # 收窄口径时 (需与 build_qa_subset.py / 打分步骤用同一份):
  python merge_qa_rescore.py ... --types "reasoning VQA cn,science QA en,text translation cn"
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
from vlmeval.dataset.utils.ocrbrnch_v2_eval import (ocrbench_v2_aggregate_accuracy,
                                                    qa_rescore_affected_types)

# vlmeval 导入正常但环境变量把 judge 全关了 -> affected 为空, 此时无 subset 可合并。
AFFECTED_DEFAULT = {
    "APP agent en", "cognition VQA en", "diagram QA en", "math QA en",
    "reasoning VQA en", "science QA en", "cognition VQA cn", "reasoning VQA cn",
    "text translation cn",
}


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', required=True, help='全量 run 的 scored.jsonl (提供不受影响子类的确定性分)')
    ap.add_argument('--subset', required=True, help='子集 run 的 scored.jsonl (受影响子类的新分)')
    ap.add_argument('--out-prefix', required=True, help='输出文件前缀')
    ap.add_argument('--types', default=None,
                    help='逗号分隔的受影响子类白名单, 显式覆盖 OCRBENCH_QA_JUDGE_TYPES 解析结果')
    args = ap.parse_args()

    if args.types:
        affected = {t.strip() for t in args.types.split(',') if t.strip()}
    else:
        affected = qa_rescore_affected_types() or AFFECTED_DEFAULT
    print(f'[INFO] 受影响子类 ({len(affected)} 类): {sorted(affected)}')
    print(f'[INFO] 来源: {"--types" if args.types else "环境变量 (OCRBENCH_QA_JUDGE_TYPES 等)"}')

    ref = load_jsonl(args.reference)
    sub = load_jsonl(args.subset)

    # 校验: 分数不得为 None; subset 子类必须正好落在受影响集合内
    for name, rows in (('reference', ref), ('subset', sub)):
        bad = [r for r in rows if r.get('score') is None]
        if bad:
            raise SystemExit(f'[FATAL] {name} 有 {len(bad)} 条 score 为 None, 首个: {bad[0]}')
    sub_cats = {r['category'] for r in sub}
    if not sub_cats <= affected:
        raise SystemExit(
            f'[FATAL] subset 含非受影响子类: {sorted(sub_cats - affected)}\n'
            f'        subset 实际含 {len(sub_cats)} 类, 本次 affected 只有 {len(affected)} 类。\n'
            f'        多半是 build_qa_subset.py 与 merge 用的 OCRBENCH_QA_JUDGE_TYPES 不一致 —— '
            f'两步请传同一份环境变量, 或都用 --types 指定同一列表。'
        )

    # 逐类条数校验: subset 每类条数必须与 reference 一致
    ref_cnt = defaultdict(int)
    for r in ref:
        ref_cnt[r['category']] += 1
    sub_cnt = defaultdict(int)
    for r in sub:
        sub_cnt[r['category']] += 1
    for cat in sorted(affected):
        if ref_cnt.get(cat, 0) != sub_cnt.get(cat, 0):
            raise SystemExit(
                f'[FATAL] 子类 {cat} 条数不一致: '
                f'reference={ref_cnt.get(cat,0)} subset={sub_cnt.get(cat,0)}\n'
                f'        subset 少了该子类? 说明 build_qa_subset.py 的抽取范围比这里窄,'
                f'两步请用同一份 OCRBENCH_QA_JUDGE_TYPES / --types。'
            )

    # 合并: 受影响子类用 subset 新分, 其余沿用 reference
    merged = [r for r in ref if r['category'] not in affected] + sub
    assert len(merged) == len(ref), f'合并后条数 {len(merged)} != reference {len(ref)}'

    # 聚合 (与 score_ocrbench_v2_org.py 同逻辑)
    data_list = [{'type': r['category'], 'score': r['score']} for r in merged]
    en_scores, cn_scores = ocrbench_v2_aggregate_accuracy(data_list)
    final = {**en_scores, **cn_scores}
    if en_scores:
        final['English Overall Score'] = sum(en_scores.values()) / len(en_scores)
    if cn_scores:
        final['Chinese Overall Score'] = sum(cn_scores.values()) / len(cn_scores)

    cat_scores = defaultdict(list)
    for r in merged:
        cat_scores[r['category']].append(r['score'])
    subcat = {cat: round(sum(v) / len(v), 6) for cat, v in sorted(cat_scores.items())}

    score_json = args.out_prefix + '_score.json'
    subcat_json = args.out_prefix + '_subcategory_score.json'
    jsonl_out = args.out_prefix + '_scored.jsonl'
    with open(score_json, 'w') as f:
        json.dump(final, f, indent=4, ensure_ascii=False)
    with open(subcat_json, 'w') as f:
        json.dump(subcat, f, indent=4, ensure_ascii=False)
    with open(jsonl_out, 'w') as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f'[OUT] {score_json}')
    print(f'  EN Overall: {final["English Overall Score"]*100:.2f}  CN Overall: {final["Chinese Overall Score"]*100:.2f}')


if __name__ == '__main__':
    main()
