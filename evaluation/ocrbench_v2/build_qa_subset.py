"""按受 LLM-judge 口径影响的子类抽取子集 xlsx (复刻 qa_translation_subset 口径)。

抽取范围与打分口径**同源**: 由 vlmeval 的 qa_rescore_affected_types() 按
OCRBENCH_QA_JUDGE_TYPES / OCRBENCH_QA_JUDGE / OCRBENCH_SKIP_LLM_JUDGE 解析。
不设任何环境变量时 = 原 9 个子类, 行为不变; 设了 OCRBENCH_QA_JUDGE_TYPES 收窄
judge 范围时, 这里只抽真正会被重判的子类 (少抽的类别由 reference 全量分沿用)。
可用 --types 显式覆盖。

注意: 抽取 / 打分 / merge 三步必须用同一组环境变量, 否则 merge_qa_rescore.py 的
"subset 子类必须在 affected 内" 与 "逐类条数一致" 校验会 FATAL (loud fail, 不会静默错分)。
"""
from pathlib import Path as _EvalPath
import os as _eval_os
_EVAL_ROOT = _EvalPath(_eval_os.environ.get("VLMEVAL_ROOT", str(_EvalPath(__file__).resolve().parents[0]))).resolve()

import argparse, os, sys
import pandas as pd
from collections import Counter

WORKDIR = str(_EVAL_ROOT / '.')
# vlmeval 不可用时 (NFS 降级 / 缺依赖) 的退回值 —— 与历史口径一致的全量 9 类。
DEFAULT_AFFECTED = ['APP agent en', 'cognition VQA en', 'diagram QA en', 'math QA en',
                    'reasoning VQA en', 'science QA en', 'cognition VQA cn', 'reasoning VQA cn',
                    'text translation cn']
NEED = ['index', 'orig_idx', 'category', 'question', 'answer', 'eval', 'bbox', 'content', 'prediction']


def resolve_affected(cli_types=None):
    """--types 优先; 否则走 vlmeval 共享解析 (读环境变量); 再退回默认 9 类。"""
    if cli_types:
        return [t.strip() for t in cli_types.split(',') if t.strip()]
    sys.path.insert(0, WORKDIR)
    os.environ.setdefault('VLMEVAL_NO_TORCH', '1')
    try:
        from vlmeval.dataset.utils.ocrbrnch_v2_eval import qa_rescore_affected_types
        resolved = sorted(qa_rescore_affected_types())
        if not resolved:
            print('[INFO] judge 已全局关闭 (OCRBENCH_SKIP_LLM_JUDGE 非空) -> affected 为空, '
                  '子集将只含 0 行; 这通常意味着本次无需重测 QA 子类。')
        return resolved
    except Exception as e:
        print(f'[WARN] 无法从 vlmeval 解析 judge 子类 ({type(e).__name__}: {e}); 退回默认 9 类。'
              f'\n       若打分/merge 时设了 OCRBENCH_QA_JUDGE_TYPES, 请改用 --types 传同一份列表, '
              f'否则 merge 的条数校验会 FATAL。')
        return list(DEFAULT_AFFECTED)


ap = argparse.ArgumentParser()
ap.add_argument('--xlsx', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--types', default=None,
                help='逗号分隔的子类白名单, 显式覆盖环境变量解析结果')
a = ap.parse_args()

AFFECTED = resolve_affected(a.types)
print(f'[INFO] 抽取子类 ({len(AFFECTED)} 类): {AFFECTED}')
print(f'[INFO] OCRBENCH_QA_JUDGE_TYPES={os.environ.get("OCRBENCH_QA_JUDGE_TYPES")!r} '
      f'OCRBENCH_QA_JUDGE={os.environ.get("OCRBENCH_QA_JUDGE")!r} '
      f'OCRBENCH_SKIP_LLM_JUDGE={os.environ.get("OCRBENCH_SKIP_LLM_JUDGE")!r}')

d = pd.read_excel(a.xlsx)
miss = [c for c in NEED if c not in d.columns]
if miss:
    raise SystemExit(f'[FATAL] {a.xlsx} 缺列 {miss}; 实际列: {list(d.columns)}')
if 'orig_idx' not in d.columns:
    d['orig_idx'] = d['index'] if 'index' in d.columns else range(len(d))

sub = d[d['category'].isin(AFFECTED)][NEED].copy()
missing_cats = set(AFFECTED) - set(sub['category'])
if missing_cats:
    print(f'[WARN] 全量 xlsx 里没有这些子类: {sorted(missing_cats)}')
sub = sub.sort_values('index').reset_index(drop=True)
sub['orig_idx'] = range(len(sub))   # 子集内行号: 打分后按 __xlsx_idx 恢复子集顺序
out_dir = os.path.dirname(a.out)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
sub.to_excel(a.out, index=False)
print(f'[OK] {a.xlsx}\n  -> {a.out}  {sub.shape}')
for k, v in sorted(Counter(sub['category']).items()):
    print(f'    {k:26} {v}')
tot = sum(Counter(sub['category']).values())
print(f'  TOTAL {tot}')
