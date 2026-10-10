#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""收尾: 合并 6 个 shard 的打分 -> 按 orig_idx 还原原序 -> 用 gemini judge 补 text translation cn
-> 重算子类分 + EN/CN 总分 -> 写出结果 (pred_scored.jsonl / subcategory / score / tt_raw)。"""
from pathlib import Path as _EvalPath
import os as _eval_os
_EVAL_ROOT = _EvalPath(_eval_os.environ.get("VLMEVAL_ROOT", str(_EvalPath(__file__).resolve().parents[0]))).resolve()

import os
import sys
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

for p in (str(_EVAL_ROOT / '.cache/localpkgs'), str(_EVAL_ROOT / '.cache/shadow')):
    if os.path.isdir(p):
        sys.path.insert(0, p)

NSHARD = 6
SHARD_XLSX = ['/tmp/shard_%d.xlsx' % k for k in range(NSHARD)]
SHARD_OUT = ['/tmp/sout_%d' % k for k in range(NSHARD)]
FULL_XLSX = '/tmp/pred.xlsx'
OUT_DIR = '/tmp/score_out'


def main():
    import pandas as pd
    # 1) 合并 shards: zip shard xlsx 的 orig_idx 与其 scored.jsonl (同序) -> 还原原序
    combined = []  # list of {orig_idx, category, score, skipped}
    for k in range(NSHARD):
        xlsx = pd.read_excel(SHARD_XLSX[k]).reset_index(drop=True)
        jl = os.path.join(SHARD_OUT[k], f'shard_{k}_scored.jsonl')
        rows = [json.loads(l) for l in open(jl)]
        if len(rows) != len(xlsx):
            print(f'[WARN] shard {k}: xlsx {len(xlsx)} vs scored {len(rows)} 行数不一致')
        for j in range(min(len(rows), len(xlsx))):
            r = rows[j]
            combined.append({
                'orig_idx': int(xlsx.iloc[j]['orig_idx']),
                'category': r.get('category'),
                'score': r.get('score'),
                'skipped': r.get('skipped', False),
            })
    combined.sort(key=lambda x: x['orig_idx'])
    print(f'[INFO] 合并 {len(combined)} 条, 已按 orig_idx 排序。')

    # 2) 用 gemini judge 补 text translation cn (读全量 xlsx 拿 orig_idx -> 对应 predict/answers)
    from judge_text_translation import JudgeClient, _safe_literal_eval
    from vlmeval.dataset.utils.Ocrbench_v2.vqa_metric import text_translation_vlm_judge_evaluation
    env = {}
    for line in open(str(_EVAL_ROOT / './.env')):
        if '=' in line and not line.strip().startswith('#'):
            kk, vv = line.strip().split('=', 1); env[kk] = vv
    base = env.get('OPENAI_API_BASE'); key = env.get('OPENAI_API_KEY')
    judge = JudgeClient(base, key, 'gemini-3.5-flash', max_tokens=8192)
    full = pd.read_excel(FULL_XLSX).reset_index(drop=True)
    tt = full[full['category'] == 'text translation cn'].copy()
    tt_items = []
    for idx in tt.index:
        r = full.loc[idx]
        tt_items.append((int(idx), str(r['prediction']) if pd.notna(r['prediction']) else '',
                         _safe_literal_eval(r['answer'])))
    print(f'[INFO] text translation cn {len(tt_items)} 条, gemini judge 中 ...')
    tt_scores = {}
    os.makedirs(OUT_DIR, exist_ok=True)
    tt_cache = os.path.join(OUT_DIR, 'pred_tt_raw_scores.json')
    if os.path.exists(tt_cache):  # 已 judge 过则复用, 避免重复花 API
        tt_scores = {int(k): float(v) for k, v in json.load(open(tt_cache)).items()}
        print(f'[INFO] 复用已有 tt judge 分数 {len(tt_scores)} 条, 跳过 judge。')
    else:
        t0 = time.time(); done = [0]

        def work(it):
            oi, pred, ans = it
            return oi, float(text_translation_vlm_judge_evaluation(judge, pred, ans))

        with ThreadPoolExecutor(max_workers=24) as ex:
            futs = [ex.submit(work, it) for it in tt_items]
            for f in as_completed(futs):
                oi, s = f.result(); tt_scores[oi] = s
                done[0] += 1
                if done[0] % 50 == 0:
                    print(f'[judge] {done[0]}/{len(tt_items)} ({time.time()-t0:.0f}s)', flush=True)
        json.dump({str(k): v for k, v in tt_scores.items()}, open(tt_cache, 'w'), indent=2, ensure_ascii=False)
        print(f'[INFO] judge 完成 {time.time()-t0:.0f}s, 已存 {tt_cache}')
    # 注入 tt 分数
    for row in combined:
        if row['category'] == 'text translation cn':
            row['score'] = tt_scores.get(row['orig_idx'], 0.0)
            row['skipped'] = False

    # 3) 重算子类分 + EN/CN 总分
    from vlmeval.dataset.utils.ocrbrnch_v2_eval import ocrbench_v2_aggregate_accuracy
    cat_scores = defaultdict(list)
    for row in combined:
        if row.get('skipped') or row.get('score') is None:
            continue
        cat_scores[row['category']].append(row['score'])
    subcat = {c: round(sum(v) / len(v), 6) for c, v in cat_scores.items()}
    agg_in = []
    for r in combined:
        it = {'type': r['category'], 'score': r['score']}
        if r.get('skipped'):
            it['ignore'] = 'True'
        agg_in.append(it)
    en, cn = ocrbench_v2_aggregate_accuracy(agg_in)
    final = {**en, **cn}
    if en: final['English Overall Score'] = sum(en.values()) / len(en)
    if cn: final['Chinese Overall Score'] = sum(cn.values()) / len(cn)

    # 4) 写出
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'pred_scored.jsonl'), 'w') as f:
        for r in combined:
            f.write(json.dumps({'orig_idx': r['orig_idx'], 'category': r['category'],
                                'score': r['score']}, ensure_ascii=False) + '\n')
    json.dump(subcat, open(os.path.join(OUT_DIR, 'pred_subcategory_score.json'), 'w'), indent=4, ensure_ascii=False)
    json.dump(final, open(os.path.join(OUT_DIR, 'pred_score.json'), 'w'), indent=4, ensure_ascii=False)
    json.dump({str(k): v for k, v in tt_scores.items()},
              open(os.path.join(OUT_DIR, 'pred_tt_raw_scores.json'), 'w'), indent=2, ensure_ascii=False)
    print('[OUT] 写出 pred_scored.jsonl / pred_subcategory_score.json / pred_score.json / pred_tt_raw_scores.json')
    print(f'[INFO] EN overall={final.get("English Overall Score")}  CN overall={final.get("Chinese Overall Score")}')
    print(f'[INFO] formula recognition en={subcat.get("formula recognition en")}  cn={subcat.get("formula recognition cn")}')
    print(f'[INFO] text translation cn={subcat.get("text translation cn")}')
    print(f'[INFO] doc parsing en={subcat.get("document parsing en")}  cn={subcat.get("document parsing cn")}')


if __name__ == '__main__':
    main()