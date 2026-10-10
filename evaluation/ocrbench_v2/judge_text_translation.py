#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对 ocrbench_v21 的 LLM-judge 子类 "text translation cn" 用 gemini-3.5-flash 补打分,
并合并进已有结果 (scored.jsonl / subcategory_score.json / score.json)。

- 复用 vqa_metric.TEXT_TRANSLATION_JUDGE_PROMPT / _parse_score / text_translation_vlm_judge_evaluation,
  保证与官方 judge 逻辑一致; judge 客户端自实现 (直接打 matrixllm OpenAI 兼容接口),
  避免导入 vlmeval.api(在降级 NFS 上很重)。
- gemini-3.5-flash 是推理模型 (reasoning_tokens), 官方默认 max_tokens=128 会被推理吃光 -> 空内容。
  这里用 max_tokens=JUDGE_MAX_TOKENS(默认 8192) 留足推理+数字答案空间。
"""
from pathlib import Path as _EvalPath
import os as _eval_os
_EVAL_ROOT = _EvalPath(_eval_os.environ.get("VLMEVAL_ROOT", str(_EvalPath(__file__).resolve().parents[0]))).resolve()

import argparse
import ast
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 本地 overlay (抗 NFS 降级) ---
_LP = os.environ.get('VLMEVAL_LOCAL_PKGS', '')
_SH = os.environ.get('VLMEVAL_SHADOW_DIR', '')
if _LP and os.path.isdir(_LP):
    sys.path.insert(0, _LP)
if _SH and os.path.isdir(_SH):
    sys.path.insert(0, _SH)

import requests


def _safe_literal_eval(value):
    raw = value if isinstance(value, str) else str(value)
    try:
        r = ast.literal_eval(raw)
    except Exception:
        r = raw
    if isinstance(r, (list, dict, tuple)):
        return r
    s = raw.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        try:
            rec = ast.literal_eval(s[1:-1].replace('""', '"'))
            if isinstance(rec, (list, dict, tuple)):
                return rec
        except Exception:
            pass
    return r


class JudgeClient:
    """最小 OpenAI 兼容 judge 客户端: .generate(prompt) -> 返回 message.content 字符串。"""
    def __init__(self, base, key, model, max_tokens=8192, timeout=120, max_retries=4):
        self.base = base
        self.key = key
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

    def generate(self, prompt):
        url = self.base
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens, "temperature": 0}
        last = None
        for _ in range(self.max_retries):
            try:
                r = requests.post(url, headers=headers, json=body, timeout=self.timeout)
                if r.status_code != 200:
                    last = f"http {r.status_code}: {r.text[:200]}"
                    time.sleep(2)
                    continue
                data = r.json()
                msg = data.get("choices", [{}])[0].get("message", {})
                content = msg.get("content")
                fr = data.get("choices", [{}])[0].get("finish_reason")
                if content:
                    return content
                if fr == "length":
                    # 推理把预算吃光, 没有内容 -> 视为 0 (重试也难好, 直接返回空让 _parse_score->0)
                    return ""
                last = f"no content: {str(data)[:200]}"
            except Exception as e:
                last = repr(e)
                time.sleep(2)
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', default='/tmp/pred.xlsx')
    ap.add_argument('--env', default=str(_EVAL_ROOT / './.env'))
    ap.add_argument('--model', default='gemini-3.5-flash')
    ap.add_argument('--nproc', type=int, default=24)
    ap.add_argument('--max-tokens', type=int, default=8192)
    ap.add_argument('--scored', default='/tmp/score_out/pred_scored.jsonl',
                    help='已有 scored jsonl (来自确定性打分)')
    ap.add_argument('--out-dir', default='/tmp/score_out')
    args = ap.parse_args()

    # 读 .env
    env = {}
    for line in open(args.env):
        if '=' in line and not line.strip().startswith('#'):
            k, v = line.strip().split('=', 1)
            env[k] = v
    base = env.get('OPENAI_API_BASE') or os.environ.get('OPENAI_API_BASE')
    key = env.get('OPENAI_API_KEY') or os.environ.get('OPENAI_API_KEY')
    if not base or not key:
        print('[FATAL] 缺 OPENAI_API_BASE/OPENAI_API_KEY', file=sys.stderr); sys.exit(1)
    print(f'[INFO] judge model={args.model}  base={base}  max_tokens={args.max_tokens}')

    # 读 xlsx, 取 text translation cn
    import pandas as pd
    d = pd.read_excel(args.pred).reset_index(drop=True)
    mask = d['category'] == 'text translation cn'
    tt = d[mask].copy()
    print(f'[INFO] text translation cn 条数: {len(tt)}')
    items = []
    for idx in tt.index:
        r = d.loc[idx]
        predict = str(r['prediction']) if pd.notna(r['prediction']) else ''
        answers = _safe_literal_eval(r['answer'])
        items.append((idx, predict, answers))

    from vlmeval.dataset.utils.Ocrbench_v2.vqa_metric import text_translation_vlm_judge_evaluation
    judge = JudgeClient(base, key, args.model, max_tokens=args.max_tokens)

    # 并行 judge
    scores = {}
    t0 = time.time()
    done = [0]
    def work(it):
        idx, predict, answers = it
        s = text_translation_vlm_judge_evaluation(judge, predict, answers)
        return idx, s
    with ThreadPoolExecutor(max_workers=args.nproc) as ex:
        futs = [ex.submit(work, it) for it in items]
        for f in as_completed(futs):
            idx, s = f.result()
            scores[int(idx)] = float(s)
            done[0] += 1
            if done[0] % 25 == 0 or done[0] == len(items):
                print(f'[judge] {done[0]}/{len(items)}  ({time.time()-t0:.0f}s)')
    print(f'[INFO] judge 完成, 耗时 {time.time()-t0:.0f}s')

    # 分布
    vals = list(scores.values())
    print(f'[INFO] score 分布: mean={sum(vals)/len(vals):.4f}  >0: {sum(1 for v in vals if v>0)}/{len(vals)}')

    # 先落盘原始 judge 分数 (index->score), 防止后续合并出错丢分
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'pred_tt_raw_scores.json'), 'w') as f:
        json.dump({str(k): v for k, v in scores.items()}, f, indent=2, ensure_ascii=False)
    print('[OUT] raw judge scores ->', os.path.join(args.out_dir, 'pred_tt_raw_scores.json'))

    # ---- 合并进已有 scored.jsonl ----
    rows = []
    with open(args.scored) as f:
        for line in f:
            rows.append(json.loads(line))
    # scored jsonl 按 xlsx 全量 10000 行顺序写, 位置 = xlsx 行号 = scores key
    for gi, row in enumerate(rows):
        if row.get('category') == 'text translation cn':
            row['score'] = scores.get(gi, 0.0)
            row.pop('skipped', None)
    with open(args.scored, 'w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    # ---- 重算 subcategory + EN/CN 聚合 (复刻 ocrbench_v2_aggregate_accuracy 口径) ----
    from vlmeval.dataset.utils.ocrbrnch_v2_eval import ocrbench_v2_aggregate_accuracy
    agg_in = []
    for row in rows:
        it = {'type': row['category'], 'score': row['score']}
        if row.get('skipped'):
            it['ignore'] = 'True'
        agg_in.append(it)
    cat_scores = defaultdict(list)
    for row in rows:
        if row.get('skipped'):
            continue
        if row.get('score') is None:
            continue
        cat_scores[row['category']].append(row['score'])
    subcat = {c: round(sum(v) / len(v), 6) for c, v in cat_scores.items()}
    en, cn = ocrbench_v2_aggregate_accuracy(agg_in)
    final = {**en, **cn}
    if en:
        final['English Overall Score'] = sum(en.values()) / len(en) if en else 0
    if cn:
        final['Chinese Overall Score'] = sum(cn.values()) / len(cn) if cn else 0

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'pred_subcategory_score.json'), 'w') as f:
        json.dump(subcat, f, indent=4, ensure_ascii=False)
    with open(os.path.join(args.out_dir, 'pred_score.json'), 'w') as f:
        json.dump(final, f, indent=4, ensure_ascii=False)
    print('[OUT] 已更新:', os.path.join(args.out_dir, 'pred_subcategory_score.json'),
          '&&', os.path.join(args.out_dir, 'pred_score.json'))
    en_ov = final.get('English Overall Score')
    cn_ov = final.get('Chinese Overall Score')
    tt_avg = subcat.get('text translation cn')
    print(f'[INFO] EN overall={en_ov}  CN overall={cn_ov}')
    print(f'[INFO] text translation cn avg = {tt_avg}')


if __name__ == '__main__':
    main()