#!/usr/bin/env python3
# =============================================================================
# ocrbench_v21 独立推理驱动 —— 模型调用完全照用户手写 SDK 方式 (不用 vlmeval 的调用代码):
#
#   from openai import OpenAI
#   client = OpenAI(base_url="https://antalchemy.alipay.com/v1", api_key=<AK>)
#   client.chat.completions.create(model=MODEL, messages=[...])
#
# 设计 (针对本机链路分钟级抖动):
#   - 每条请求连接级失败指数退避无限重试 (封顶 60s), 窗口关着就等, 开了就续跑
#   - JSONL 逐条落盘断点续跑 (--ckpt), 崩了/停了重跑即补
#   - 全局 RPM 门 (默认 48) + 多线程 worker (默认 8) 填慢条目
#   - 4xx (鉴权/参数/找不到) 不重试: 写失败标记继续下一条 (日志看得到)
#   - 7 个 QA 子类 question 末尾追加 <answer> 格式约束 (answerfmt 口径, 与打分侧对齐)
# 支撑打分管线: 完成后写 <run_key>_ocrbench_v21.xlsx (index/category/question/answer/
#   eval/bbox/content/prediction/orig_idx), 交 score_ocrbench_v2_org.py 离线打分。
# =============================================================================
from pathlib import Path as _EvalPath
import os as _eval_os
_EVAL_ROOT = _EvalPath(_eval_os.environ.get("VLMEVAL_ROOT", str(_EvalPath(__file__).resolve().parents[0]))).resolve()

import argparse
import base64
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

ANSWER_FMT_TYPES_DEFAULT = (
    'cognition VQA cn,cognition VQA en,reasoning VQA cn,reasoning VQA en,'
    'math QA en,diagram QA en,science QA en')

ANSWER_FORMAT_SUFFIX = (
        '\n\nResponse format: Put the complete final answer in exactly one <answer>...</answer> block. '
        'If reasoning is provided, put it in <think>...</think>. Do not output any text outside these blocks. '
        'Preserve all requested sub-answers, units, and answer structure.'
    )

_THINK_RE = re.compile(r'<think>.*?</think>', re.S)

_rate_lock = threading.Lock()
_rate_t0 = 0.0


def rpm_gate(rpm):
    global _rate_t0
    if not rpm or rpm <= 0:
        return
    with _rate_lock:
        now = time.monotonic()
        wait = 60.0 / rpm - (now - _rate_t0)
        if wait > 0:
            time.sleep(wait)
        _rate_t0 = time.monotonic()


assert isinstance(ANSWER_FORMAT_SUFFIX, str) and '<answer>' in ANSWER_FORMAT_SUFFIX, \
    'ANSWER_FORMAT_SUFFIX 定义缺失/损毁'


# doubao 网关图片像素上限 36,000,000 (超限 400 InvalidParameter); 超大的图等比缩到上限内再传,
# 只影响极少数超大图 (实测全量 10000 条里 2 条), 其余原样直传不动。
_MAX_PIXELS = 34_000_000


def _fit_image_b64(b64):
    import io
    from PIL import Image
    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        w, h = img.size
        if w * h <= _MAX_PIXELS:
            return b64
        scale = (_MAX_PIXELS / (w * h)) ** 0.5
        img = img.convert('RGB').resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=90)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return b64

def build_messages(line, fmt_types):
    img_b64 = _fit_image_b64(line['image'])
    q = line['question']
    if line['category'] in fmt_types:
        q += ANSWER_FORMAT_SUFFIX
    if isinstance(img_b64, str) and img_b64:
        content = [
            dict(type='image_url', image_url=dict(url=f'data:image/jpeg;base64,{img_b64}', detail='high')),
            dict(type='text', text=q),
        ]
    else:  # 理论上 OCRBench 全带图, 兜底纯文本
        content = q
    return [dict(role='user', content=content)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tsv', default=str(_EVAL_ROOT / 'data/ocrbench_v21.tsv'))
    ap.add_argument('--out-dir', required=True)               # outputs/<run_key>/
    ap.add_argument('--xlsx-name', required=True)             # <run_key>_ocrbench_v21.xlsx
    ap.add_argument('--ckpt-name', default=None)              # 默认 <run_key>_ckpt.jsonl
    ap.add_argument('--model', default=os.environ.get('MODEL_API', 'doubao-seed-2-0-lite-260428'))
    ap.add_argument('--base-url', default=os.environ.get('API_BASE', 'https://antalchemy.alipay.com/v1'))
    ap.add_argument('--key', default=None)                    # 不传则读 env OPENAI_API_KEY
    ap.add_argument('--key-env', default='OPENAI_API_KEY')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--rpm', type=int, default=48)
    ap.add_argument('--max-tokens', type=int, default=32768)   # 0 = 不传该参数 (用户手写方式)
    ap.add_argument('--sdk-retries', type=int, default=2)     # SDK 连接级重试 (用户 test.py 的默认值)
    ap.add_argument('--answer-fmt-types', default=ANSWER_FMT_TYPES_DEFAULT)
    ap.add_argument('--only-types', default='')               # 冒烟: 逗号分隔子类过滤
    ap.add_argument('--limit', type=int, default=0)           # 冒烟: 只跑前 N 条
    ap.add_argument('--heartbeat', type=int, default=60)      # 链路断着时的心跳打印间隔 (秒)
    ap.add_argument('--rescue-after', type=int, default=3)     # 连续 N 次连接级失败后 -> 弃图纯文本兜底      # 链路断着时的心跳打印间隔 (秒)
    args = ap.parse_args()

    key = args.key or os.environ.get(args.key_env)
    assert key, f'缺 key: --key 或 env {args.key_env}'

    from openai import OpenAI                                  # 用户方式的唯一调用通道
    client = OpenAI(base_url=args.base_url, api_key=key, max_retries=args.sdk_retries)

    import pandas as pd
    print(f'[infer] 读 TSV {args.tsv} ...', flush=True)
    df = pd.read_csv(args.tsv, sep='\t')
    if args.only_types:
        want = {t.strip() for t in args.only_types.split(',') if t.strip()}
        df = df[df['category'].isin(want)].reset_index(drop=True)
        print(f'[infer] 只保留子类 {sorted(want)} -> {len(df)} 条', flush=True)
    if args.limit and args.limit < len(df):
        df = df.iloc[:args.limit].copy()
        print(f'[infer] limit -> {len(df)} 条', flush=True)

    fmt_types = {t.strip() for t in args.answer_fmt_types.split(',') if t.strip()}
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = os.path.join(args.out_dir, args.ckpt_name or (os.path.splitext(args.xlsx_name)[0] + '_ckpt.jsonl'))

    done = {}
    if os.path.isfile(ckpt):
        with open(ckpt) as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                    done[r['index']] = r          # 同 index 后写覆盖前写
                except Exception:
                    pass
        print(f'[infer] 断点续跑: 已完成 {sum(1 for v in done.values() if v.get("ok"))} 条'
              f' (失败待补 {sum(1 for v in done.values() if not v.get("ok"))} 条)', flush=True)

    todos = [i for i in range(len(df)) if not done.get(i, {}).get('ok')]
    print(f'[infer] 共 {len(df)} 条, 待跑 {len(todos)} 条 | model={args.model} base={args.base_url} '
          f'workers={args.workers} rpm={args.rpm} max_tokens={args.max_tokens or "不传"}', flush=True)
    if not todos:
        print('[infer] 全部完成', flush=True)
    ckpt_fp = open(ckpt, 'a', buffering=1)
    ckpt_lock = threading.Lock()
    stat = {'done': 0, 'fail': 0, 'total': len(todos), 'last_ok': time.time(), 'beat': 0,
            'ff': 0, 'okc': 0}
    stat_lock = threading.Lock()

    def one(i):
        line = df.iloc[i]
        payload = dict(model=args.model, messages=build_messages(line, fmt_types))
        if args.max_tokens:
            payload['max_tokens'] = args.max_tokens
        backoff = 1.0
        n_try = 0
        rescued = False                          # 毒图条目: 内容触发网关永久挂起, 降级纯文本
        while True:
            n_try += 1
            if not rescued and args.rescue_after and n_try > args.rescue_after:
                tq = line['question']
                if line['category'] in fmt_types:
                    tq += ANSWER_FORMAT_SUFFIX
                payload['messages'] = [dict(role='user', content=tq)]
                rescued = True
                print(f'  [rescue] idx={i} 带图连续 {args.rescue_after} 次连接级失败'
                      f'(疑似内容触发网关挂起) -> 第 {n_try} 次起弃图纯文本重试', flush=True)
            rpm_gate(args.rpm)
            try:
                resp = client.chat.completions.create(**payload)
                msg = resp.choices[0].message
                pred = (msg.content or '').strip()
                if resp.choices[0].finish_reason == 'length':
                    pred += f'\n[TRUNCATED finish=length]'   # 打分侧可见, 统计时好找
                rec = dict(index=int(i), ok=True, prediction=pred, model=resp.model,
                           finish=resp.choices[0].finish_reason,
                           out_tokens=getattr(resp.usage, 'completion_tokens', None),
                           rescue=('text_only' if rescued else ''))
                backoff = 1.0
                break
            except Exception as e:
                ename = type(e).__name__
                if e.__class__.__name__ in (
                        'BadRequestError', 'AuthenticationError', 'NotFoundError',
                        'PermissionDeniedError', 'UnprocessableEntityError'):
                    # 4xx 重试无意义: 记录后放弃这条 (改参数后重跑本条即可)
                    if 'unsafe' in str(e).lower():
                        # 网关图文安全拦截 (kimi/antalchemy: image_url_unsafe): 弃图纯文本重试
                        tq = line['question']
                        if line['category'] in fmt_types:
                            tq += ANSWER_FORMAT_SUFFIX if ANSWER_FORMAT_SUFFIX else ''
                        payload['messages'] = [dict(role='user', content=tq)]
                        rescued = True
                        print(f'  [rescue-unsafe] idx={i} 图片被安全拦截 -> 弃图纯文本重试', flush=True)
                        time.sleep(1)
                        continue
                    rec = dict(index=int(i), ok=False, prediction='', error=f'{ename}: {str(e)[:300]}')
                    print(f'  [fail-fast] idx={i} {ename}: {str(e)[:160]}', flush=True)
                    with stat_lock:
                        stat['ff'] += 1
                        # 保险: 一条都没成功却连续 4xx -> 参数/模型/网关在拒绝我们,
                        # 立刻停跑, 避免把 10000 条配额全烧在必败请求上 (改 max_tokens/模型后重跑即可续)
                        if stat['ff'] >= 20 and stat['okc'] == 0:
                            print(f'[infer][FATAL] 开局 {stat["ff"]} 条全部 4xx 且 0 成功 '
                                  f'(最后一条: {ename}: {str(e)[:200]})。停跑防烧配额。'
                                  f'按报错改参数 (如 MAX_TOKENS=16384) 或换模型后重跑本命令。', flush=True)
                            ckpt_fp.flush()
                            os._exit(3)
                    break
                with stat_lock:
                    stat['beat'] += 1
                    if time.time() - stat['last_ok'] > args.heartbeat and \
                            (stat['beat'] % max(1, args.workers)) == 1:
                        print(f'  [heartbeat {time.strftime("%H:%M:%S")}] 链路断着/慢: '
                              f'累计重试 {stat["beat"]} 次, 退避 {backoff:.0f}s '
                              f'(等好窗口自动续跑; 进度 {stat["done"]}/{stat["total"]})', flush=True)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
        with ckpt_lock:
            ckpt_fp.write(json.dumps(rec, ensure_ascii=False) + '\n')
        with stat_lock:
            stat['done'] += 1
            if rec['ok']:
                stat['okc'] += 1
                stat['last_ok'] = time.time()
            else:
                stat['fail'] += 1
            if stat['done'] % 20 == 0 or stat['done'] == stat['total']:
                print(f'  [prog {time.strftime("%H:%M:%S")}] {stat["done"]}/{stat["total"]} '
                      f'(累计失败 {stat["fail"]})', flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(one, todos))
    ckpt_fp.close()

    # ---- 合并 checkpoint -> 打分用 xlsx ----
    print('[infer] 写 xlsx ...', flush=True)
    preds = {}
    with open(ckpt) as f:
        for ln in f:
            try:
                r = json.loads(ln)
                preds[r['index']] = r
            except Exception:
                pass
    rows = []
    for i in range(len(df)):
        line = df.iloc[i]
        r = preds.get(i, {})
        rows.append(dict(
            index=line['index'], orig_idx=i, category=line['category'],
            question=line['question'], answer=line['answer'], eval=line['eval'],
            bbox=line['bbox'], content=line['content'],
            prediction=r.get('prediction', '')))
    out = pd.DataFrame(rows)
    xlsx = os.path.join(args.out_dir, args.xlsx_name)
    out.to_excel(xlsx, index=False)
    nbad = int((out['prediction'].str.strip() == '').sum())
    ntrunc = int(out['prediction'].str.contains('TRUNCATED finish=length', na=False).sum())
    print(f'[infer] DONE ({time.time()-t0:.0f}s): {len(out)} 行 -> {xlsx} | 空预测 {nbad} 条 | 截断 {ntrunc} 条', flush=True)
    print(f'[infer] 耗时示例统计 mean_tokens 见 {ckpt}; 重跑本命令即自动补空预测条目', flush=True)
    if nbad:
        print(f'[infer][WARN] 有 {nbad} 条空预测 (4xx 或未完成), 重跑本命令可续补; 打分前应归零', flush=True)


if __name__ == '__main__':
    main()
