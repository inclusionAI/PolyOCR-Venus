#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""只重判 ocrbench_v21 的 "text translation cn" 子类 (400 条), 修复 antalchemy 限流把分打废的问题。

背景: 首轮评分时 text-translation 走 32 线程并发, 瞬间打满 antalchemy 租户 60 RPM 配额 -> 大量 429
-> judge 返 fail_msg -> 记 0, 导致 text translation cn ≈ 10%(垃圾值)。
本脚本只重判这 400 条(低并发 + 429 退避), 复用盘上 *已正确* 的 CDM/确定性逐条分数
(直接读 _scored.jsonl, 不重跑 25 分钟 CDM), 用官方 ocrbench_v2_aggregate_accuracy 重聚合,
先备份再覆盖 score.json / subcategory_score.json / _scored.jsonl。

用法:
  VLMEVAL_NO_TORCH=1 venv2/bin/python rejudge_text_translation_cn.py \
    --xlsx <预测.xlsx> --out-dir <T... 目录(含 _scored.jsonl 与三个 json)> \
    --judge gemini-3.5-flash-lite --judge-base-url https://antalchemy.alipay.com/v1 \
    --judge-key ak_xxx --concurrency 3 --judge-retry 8 --judge-timeout 120
"""
import argparse
import ast
import json
import os
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from vlmeval.dataset.utils.Ocrbench_v2.vqa_metric import text_translation_vlm_judge_evaluation
from vlmeval.dataset.utils.ocrbrnch_v2_eval import ocrbench_v2_aggregate_accuracy
from score_ocrbench_v2_org import _OAICJudge


def _safe_literal_eval(value):  # 复刻 scorer 同名实现
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xlsx', required=True, help='预测 xlsx')
    ap.add_argument('--out-dir', required=True, help='含 _scored.jsonl 与 score json 的目录')
    ap.add_argument('--judge', default='gemini-3.5-flash-lite')
    ap.add_argument('--judge-base-url', default='https://antalchemy.alipay.com/v1')
    ap.add_argument('--judge-key', required=True)
    ap.add_argument('--judge-retry', type=int, default=8)
    ap.add_argument('--judge-timeout', type=int, default=120)
    ap.add_argument('--concurrency', type=int, default=3)
    args = ap.parse_args()

    base = os.path.splitext(os.path.basename(args.xlsx))[0]
    out_dir = args.out_dir
    scored_jsonl = os.path.join(out_dir, base + '_scored.jsonl')
    score_json = os.path.join(out_dir, base + '_score.json')
    subcat_json = os.path.join(out_dir, base + '_subcategory_score.json')

    # 1. 读 xlsx, 取 text translation cn 行(predict, answers), 按 orig_idx/index/row 保序
    print(f'[INFO] 读 xlsx: {args.xlsx}')
    df = pd.read_excel(args.xlsx)
    has_orig = 'orig_idx' in df.columns
    items = []  # [(orig_idx, predict, answers), ...] 顺序与 scored.jsonl(__xlsx_idx 序)对齐
    for i in range(len(df)):
        line = df.iloc[i]
        if str(line['category']).strip() != 'text translation cn':
            continue
        predict = str(line['prediction']) if pd.notna(line['prediction']) else ''
        answers = _safe_literal_eval(line['answer'])
        idx = int(line['orig_idx']) if has_orig and pd.notna(line['orig_idx']) else i
        items.append((idx, predict, answers))
    items.sort()  # 按 idx 升序 -> 与 scored.jsonl 的 __xlsx_idx 序一致
    print(f'[INFO] text translation cn 行: {len(items)}')

    # 2. 构建限流安全 judge, 低并发重判
    judge = _OAICJudge(
        model=args.judge,
        api_base=args.judge_base_url.rstrip('/') + '/chat/completions',
        key=args.judge_key,
        retry=args.judge_retry,
        timeout=args.judge_timeout,
    )
    print(f'[INFO] judge={args.judge} concurrency={args.concurrency} retry={args.judge_retry} '
          f'timeout={args.judge_timeout}s -> 重判 {len(items)} 条 (限流安全)')
    t0 = time.time()
    new_scores = [None] * len(items)  # 与 items 同序(idx 升序)
    done = [0]
    def _one(pos):
        idx, predict, answers = items[pos]
        s = text_translation_vlm_judge_evaluation(judge, predict, answers)
        return pos, s
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(_one, p): p for p in range(len(items))}
        for f in as_completed(futs):
            pos, s = f.result()
            new_scores[pos] = s
            done[0] += 1
            if done[0] % 50 == 0 or done[0] == len(items):
                print(f'[INFO] 重判进度 {done[0]}/{len(items)}  耗时 {time.time()-t0:.0f}s '
                      f'当前非零占比 {sum(1 for x in new_scores if x and x>0)}/{done[0]}', flush=True)
    n_ok = sum(1 for s in new_scores if s and s > 0)
    print(f'[INFO] 重判完成: {len(new_scores)} 条, 非零 {n_ok}, 耗时 {time.time()-t0:.0f}s')

    # 3. 读盘上 _scored.jsonl (复用 CDM/确定性逐条分), 只替换 text translation cn 行的 score
    rows = [json.loads(l) for l in open(scored_jsonl, 'r', encoding='utf-8')]
    tt_positions = [i for i, r in enumerate(rows) if r.get('category') == 'text translation cn']
    print(f'[INFO] scored.jsonl 中 text translation cn 行: {len(tt_positions)}')
    assert len(tt_positions) == len(new_scores), \
        f'对齐失败: scored={len(tt_positions)} 新判={len(new_scores)}'
    old_tt = [rows[i].get('score') for i in tt_positions]
    for pos, ns in zip(tt_positions, new_scores):
        rows[pos]['score'] = float(ns) if ns is not None else 0.0
        rows[pos]['skipped'] = False

    # 4. 备份原 json (限流版), 再覆盖
    for p in (scored_jsonl, score_json, subcat_json):
        if os.path.exists(p) and not os.path.exists(p + '.bak_ratelimit'):
            shutil.copy2(p, p + '.bak_ratelimit')
            print(f'[INFO] 备份 {os.path.basename(p)} -> .bak_ratelimit')

    # 5. 重建 res_data_list -> 官方聚合
    res_data_list = []
    cat_scores = defaultdict(list)
    for r in rows:
        cat = r.get('category', 'UNKNOWN')
        item = {'type': cat}
        if r.get('skipped') or r.get('score') is None:
            item['score'] = None
            item['ignore'] = 'True'
        else:
            sc = float(r['score'])
            item['score'] = sc
            cat_scores[cat].append(sc)
        res_data_list.append(item)
    en_scores, cn_scores = ocrbench_v2_aggregate_accuracy(res_data_list)
    final = {**en_scores, **cn_scores}
    if en_scores:
        final['English Overall Score'] = sum(en_scores.values()) / len(en_scores)
    if cn_scores:
        final['Chinese Overall Score'] = sum(cn_scores.values()) / len(cn_scores)
    subcat = {cat: round(sum(v) / len(v), 6) for cat, v in cat_scores.items()}

    # 6. 写回
    with open(score_json, 'w') as f:
        json.dump(final, f, indent=4, ensure_ascii=False)
    with open(subcat_json, 'w') as f:
        json.dump(subcat, f, indent=4, ensure_ascii=False)
    with open(scored_jsonl, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # 7. 前后对比
    old_tt_mean = sum(s if s else 0 for s in old_tt) / len(old_tt) if old_tt else 0.0
    new_tt_mean = sum(float(s) if s else 0 for s in new_scores) / len(new_scores) if new_scores else 0.0
    print('\n================ 重判前后 ================')
    print(f'  text translation cn (mean): {old_tt_mean*100:.2f}% -> {new_tt_mean*100:.2f}%  '
          f'(非零 {n_ok}/{len(new_scores)})')
    print(f'  cn_knowledge_reasoning group : {0.0} -> {cn_scores.get("cn_knowledge_reasoning",0)*100:.2f}%')
    print(f'  Chinese Overall Score         : (旧 0.6712) -> {final.get("Chinese Overall Score",0)*100:.2f}%')
    print(f'  English Overall Score (不变)  : -> {final.get("English Overall Score",0)*100:.2f}%')
    print(f'\n[OUT] 更新: {score_json}\n       {subcat_json}\n       {scored_jsonl}')


if __name__ == '__main__':
    main()
