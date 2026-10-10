#!/usr/bin/env python3
# =============================================================================
# ocrbench_v21 单阶段推理 —— 直接用 OpenAI SDK 调 doubao (antalchemy 网关)
# -----------------------------------------------------------------------------
# 参照 shell/eval_api_ocrbench_v2_org_answerfmt.sh 的「评测方法」, 但不复用其模型
# 调用部分 (vlmeval run.py + GPT4V + antchemy 网关 —— 后者在本容器 NXDOMAIN):
#   - 推理: 直接用用户给的 OpenAI client
#       base_url = https://antalchemy.alipay.com/v1
#       api_key  = $OPENAI_API_KEY  (shell wrapper 从 共享存储 .../.env 注入, 不落盘)
#       model    = doubao-seed-2-0-lite-260428
#   - prompt: 与 vlmeval/dataset/image_vqa.py 的 OCRBench_v2.build_prompt 同口径
#       messages = [user: [image_url(data:image/jpeg;base64,..; detail=high), text(question[+fmt_suffix])]]
#       无 system message (config 模板未设 system_prompt -> None)
#       temperature=0, max_tokens=32768 (同 config 模板)
#   - <answer> 格式约束: 仅对 7 个 QA 子类在 question 末尾追加 ANSWER_FORMAT_SUFFIX
#       (suffix 文本与 image_vqa.py OCRBench_v2.ANSWER_FORMAT_SUFFIX 逐字一致)
# 产物: 预测 xlsx, 列 = index/category/question/answer/eval/bbox/content/prediction
#       (score_ocrbench_v2_org.py 打分侧 -> 归一化要求这 8 列)
# 续跑: 逐条写 jsonl 日志; 重启读 jsonl 跳过已完成 index, 末尾合并出 xlsx。
# =============================================================================
from pathlib import Path as _EvalPath
import os as _eval_os
_EVAL_ROOT = _EvalPath(_eval_os.environ.get("VLMEVAL_ROOT", str(_EvalPath(__file__).resolve().parents[0]))).resolve()

import argparse, asyncio, contextlib, json, os, sys, time, traceback
from collections import deque

import pandas as pd


def _env(key, default):
    """env 取值: 缺失或空字符串 -> 用 default (父 shell 会 export 空的 OPENAI_BASE_URL)。"""
    v = os.environ.get(key, None)
    if v is None or v.strip() == "":
        return default
    return v

from openai import AsyncOpenAI
from openai import (
    APIConnectionError, APITimeoutError, RateLimitError, InternalServerError, APIStatusError,
)

# ---------- 与 image_vqa.py OCRBench_v2.ANSWER_FORMAT_SUFFIX 逐字一致 ----------
_ANSWER_BLOCK_OPEN = chr(60) + "answer" + chr(62)        # <answer>
_ANSWER_BLOCK_CLOSE = chr(47) + "answer" + chr(62)      # </answer>
_THINK_BLOCK_OPEN = chr(60) + "thinking" + chr(62)       # <thinking>
_THINK_BLOCK_CLOSE = chr(47) + "thinking" + chr(62)     # </thinking>
ANSWER_FORMAT_SUFFIX = (
    "\n\nResponse format: Put the complete final answer in exactly one "
    + _ANSWER_BLOCK_OPEN + "..." + _ANSWER_BLOCK_CLOSE + " block. If reasoning is provided, "
    "put it in " + _THINK_BLOCK_OPEN + "..." + _THINK_BLOCK_CLOSE
    + ". Do not output any text outside these blocks. "
    "Preserve all requested sub-answers, units, and answer structure."
)

DEFAULT_ANSWER_FMT_TYPES = (
    "cognition VQA cn,cognition VQA en,reasoning VQA cn,reasoning VQA en,"
    "math QA en,diagram QA en,science QA en"
)

# 13 task 类型外的 SVM 候选 —— 30 子类全集 (仅用于日志/校验, 不影响推理)
OUT_COLS = ["index", "category", "question", "answer", "eval", "bbox", "content", "prediction"]


# ---------- 网关全局 RPM 限速 (antchemy/antalchemy 租户 ~60 RPM) ----------
class RPMLimiter:
    def __init__(self, rpm):
        self.rpm = max(1, int(rpm))
        self.times = deque()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            while self.times and self.times[0] <= now - 60.0:
                self.times.popleft()
            if len(self.times) >= self.rpm:
                wait = 60.0 - (now - self.times[0]) + 0.05
                if wait > 0:
                    await asyncio.sleep(wait)
                now = time.monotonic()
                while self.times and self.times[0] <= now - 60.0:
                    self.times.popleft()
            self.times.append(time.monotonic())


def parse_image_field(raw):
    """TSV 的 image 列: 多数是单张图片的 base64(jpeg) 字符串; 个别是多图 json 列表。
    返回 base64 字符串列表 (空列表 = 无图, 退化为纯文本)。"""
    if raw is None:
        return []
    s = str(raw).strip()
    if s == "" or s.lower() == "nan":
        return []
    if s[0] == "[":  # json 列表
        try:
            lst = json.loads(s)
            if isinstance(lst, list):
                return [str(x) for x in lst if str(x).strip()]
        except Exception:
            pass
    return [s]


def build_messages(question, image_b64_list, detail="high"):
    content = []
    for b64 in image_b64_list:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": detail},
        })
    text = question if question is not None else ""
    content.append({"type": "text", "text": text})
    # 只有一条 text 且无图时, OpenAI 也接受 list-of-one; 统一用 list 形式更稳
    return [{"role": "user", "content": content}]


async def predict_one(client, args, rpm, sem, idx, row, fmt_set, log_f, state):
    """单条推理: 限速 -> 并发门 -> 带重试的 chat 模型调用 -> 写 jsonl。"""
    category = str(row["category"])
    question = "" if pd.isna(row["question"]) else str(row["question"])
    if category in fmt_set:
        question = question + ANSWER_FORMAT_SUFFIX
    image_list = parse_image_field(row["image"])
    messages = build_messages(question, image_list, detail=args.img_detail)

    async with sem:
        await rpm.acquire()
        content = ""
        last_err = None
        for attempt in range(args.retry):
            try:
                resp = await client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    n=1,
                    timeout=args.api_timeout,
                )
                msg = resp.choices[0].message
                content = msg.content or ""
                if not content.strip():
                    # 兜底: 极少数 thinking 模型把正文塞 reasoning_content
                    rc = getattr(msg, "reasoning_content", None)
                    if rc:
                        content = rc
                last_err = None
                break
            except (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError) as e:
                last_err = f"{type(e).__name__}: {str(e)[:160]}"
                backoff = min(args.backoff_max, args.backoff_base * (2 ** attempt))
                await asyncio.sleep(backoff)
            except APIStatusError as e:
                # 4xx 其它 (鉴权/参数) 不重试无限刷; 但 429 已被 RateLimitError 覆盖
                last_err = f"{type(e).__name__}: status={getattr(e,'status_code','?')} {str(e)[:160]}"
                if getattr(e, "status_code", 0) in (400, 401, 403, 404, 422):
                    break
                backoff = min(args.backoff_max, args.backoff_base * (2 ** attempt))
                await asyncio.sleep(backoff)
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:160]}"
                backoff = min(args.backoff_max, args.backoff_base * (2 ** attempt))
                await asyncio.sleep(backoff)

    if not content.strip():
        content = f"[INFER_FAIL] {last_err or 'empty response'}"

    # 逐条落盘 (asyncio 单线程 -> 顺序写, flush 保证崩溃可续)
    rec = {"index": int(idx), "category": category, "prediction": content}
    log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log_f.flush()
    state["done"] += 1
    state["pred"][int(idx)] = content
    done = state["done"]
    if done % 50 == 0 or done == state["total"]:
        el = time.monotonic() - state["t0"]
        rate = done / el if el > 0 else 0
        eta = (state["total"] - done) / rate if rate > 0 else 0
        print(f"[prog] {done}/{state['total']} ({100*done/state['total']:.1f}%) "
              f"{rate:.2f} it/s  ETA {eta/60:.1f}min  errs={state.get('errs',0)}", flush=True)
    return int(idx), content


def load_done(path):
    """读已有 jsonl -> {index: prediction} (续跑用)。"""
    done = {}
    if not path or not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done[int(r["index"])] = r.get("prediction", "")
            except Exception:
                continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=_env("ORG_TSV", str(_EVAL_ROOT / 'data/ocrbench_v21.tsv')))
    ap.add_argument("--out-xlsx", default=_env("OUT_XLSX", ""), help="预测 xlsx 输出路径")
    ap.add_argument("--log", default=_env("OUT_LOG", ""), help="续跑 jsonl 日志路径")
    ap.add_argument("--base-url", default=_env("OPENAI_BASE_URL", "https://antalchemy.alipay.com/v1"))
    ap.add_argument("--model", default=_env("MODEL_API", "doubao-seed-2-0-lite-260428"))
    ap.add_argument("--api-key", default=_env("OPENAI_API_KEY", ""), help="默认读 $OPENAI_API_KEY")
    ap.add_argument("--max-tokens", type=int, default=int(_env("MAX_TOKENS", "32768")))
    ap.add_argument("--temperature", type=float, default=float(_env("TEMPERATURE", "0")))
    ap.add_argument("--img-detail", default=_env("IMG_DETAIL", "high"))
    ap.add_argument("--nproc", type=int, default=int(_env("API_NPROC", "12")))
    ap.add_argument("--rpm", type=int, default=int(_env("API_RPM", "48")))
    ap.add_argument("--api-timeout", type=float, default=float(_env("API_TIMEOUT", "600")))
    ap.add_argument("--retry", type=int, default=int(_env("API_RETRY", "6")))
    ap.add_argument("--backoff-base", type=float, default=float(_env("BACKOFF_BASE", "2")))
    ap.add_argument("--backoff-max", type=float, default=float(_env("BACKOFF_MAX", "60")))
    ap.add_argument("--answer-fmt-types", default=_env("OCRBENCH_V2_ANSWER_FORMAT_TYPES", DEFAULT_ANSWER_FMT_TYPES))
    ap.add_argument("--only-types", default=_env("OCRBENCH_V2_EVAL_TYPE", ""), help="冒烟: 只跑指定子类 (逗号分隔)")
    ap.add_argument("--limit", type=int, default=int(_env("LIMIT", "0")), help="冒烟: 只跑前 N 行 (0=不限)")
    ap.add_argument("--checkpoint", type=int, default=int(_env("CHECKPOINT", "250")), help="每 N 条写一次 xlsx")
    args = ap.parse_args()

    if not args.api_key:
        sys.exit("[FATAL] 无 API key: 请 export OPENAI_API_KEY 或传 --api-key (从 共享存储 .../.env 注入)")

    out_xlsx = args.out_xlsx or _env(
        "OUT_XLSX",
        str(_EVAL_ROOT / './outputs/eval_doubao_api_ocrbench_v2_org_0916_answerfmt/eval_doubao_api_ocrbench_v2_org_0916_answerfmt_ocrbench_v21.xlsx'),
    )
    out_log = args.log or (os.path.splitext(out_xlsx)[0] + ".jsonl")
    os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)
    os.makedirs(os.path.dirname(out_log), exist_ok=True)

    fmt_set = {t.strip() for t in args.answer_fmt_types.split(",") if t.strip()}
    only_set = {t.strip() for t in args.only_types.split(",") if t.strip()} if args.only_types else None

    print(f"[cfg] base_url={args.base_url}", flush=True)
    print(f"[cfg] model={args.model}", flush=True)
    print(f"[cfg] tsv={args.tsv}", flush=True)
    print(f"[cfg] out_xlsx={out_xlsx}", flush=True)
    print(f"[cfg] out_log={out_log}", flush=True)
    print(f"[cfg] nproc={args.nproc} rpm={args.rpm} max_tokens={args.max_tokens} timeout={args.api_timeout}s retry={args.retry}", flush=True)
    print(f"[cfg] answer_fmt_types({len(fmt_set)})={sorted(fmt_set)}", flush=True)
    if only_set:
        print(f"[cfg] ONLY_TYPES={sorted(only_set)} (冒烟: 不是全量)", flush=True)

    # ---- 读 TSV (1.46GB, ~10000 行) ----
    t0 = time.time()
    df = pd.read_csv(args.tsv, sep="\t", dtype=str)
    df["index"] = df["index"].astype(int)
    print(f"[load] TSV {len(df)} 行, 用时 {time.time()-t0:.1f}s; 列={list(df.columns)}", flush=True)
    if only_set:
        df = df[df["category"].isin(only_set)].reset_index(drop=True)
        print(f"[load] ONLY_TYPES 过滤后 {len(df)} 行", flush=True)
    if args.limit and args.limit > 0:
        df = df.iloc[:args.limit].reset_index(drop=True)
        print(f"[load] LIMIT={args.limit} -> {len(df)} 行", flush=True)

    # 检查必备列
    need = ["index", "image", "question", "category", "answer", "eval", "bbox", "content"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        sys.exit(f"[FATAL] TSV 缺列 {miss}; 实际列={list(df.columns)}")

    # ---- 续跑: 读 jsonl 已完成的 index ----
    done = load_done(out_log)
    todo_idx = [int(i) for i in df["index"].tolist() if int(i) not in done]
    print(f"[resume] 已完成 {len(done)} 条; 待推 {len(todo_idx)} 条", flush=True)

    state = {"done": len(done), "total": len(df), "pred": dict(done), "t0": time.monotonic(), "errs": 0}
    state["pred_initial_done"] = len(done)

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=0)
    rpm = RPMLimiter(args.rpm)
    sem = asyncio.Semaphore(args.nproc)

    log_f = open(out_log, "a", encoding="utf-8")
    if done:
        print(f"[resume] (已有 {len(done)} 条不在 jsonl 追加, 直接用缓存)", flush=True)

    async def runner():
        tasks = []
        df_idx = df.set_index("index")
        for idx in todo_idx:
            row = df_idx.loc[idx]
            tasks.append(predict_one(client, args, rpm, sem, idx, row, fmt_set, log_f, state))
        for fut in asyncio.as_completed(tasks):
            try:
                await fut
            except Exception as e:
                state["errs"] += 1
                print(f"[err] run task: {type(e).__name__}: {str(e)[:200]}", flush=True)

    # 周期性写 xlsx 检查点
    async def checkpoint_loop():
        last = state["done"]
        while state["done"] < state["total"]:
            await asyncio.sleep(30)
            if state["done"] - last >= args.checkpoint:
                write_xlsx(df, state["pred"], out_xlsx)
                last = state["done"]
                print(f"[ckpt] 写 xlsx @ {state['done']}/{state['total']}", flush=True)

    async def main_task():
        ck = asyncio.create_task(checkpoint_loop())
        try:
            await runner()
        finally:
            ck.cancel()
            # CancelledError 在 py3.10 是 BaseException, 不属 Exception -> 用 suppress 兜底
            # 且 checkpoint 报错也不能阻挡最终 write_xlsx
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await ck

    try:
        asyncio.run(main_task())
    except BaseException as e:  # noqa: BLE001  —— Ctrl-C / cancel 也得留底, 不丢 3.5h 的预测
        print(f"[warn] asyncio.run 中断: {type(e).__name__}: {str(e)[:160]}", flush=True)
    finally:
        log_f.close()

    try:
        write_xlsx(df, state["pred"], out_xlsx)
    except Exception as e:
        print(f"[err] 末次 write_xlsx 失败: {e}", flush=True)
    el = time.monotonic() - state["t0"]
    print(f"[DONE] {state['done']}/{state['total']} 用时 {el/60:.1f}min -> {out_xlsx}", flush=True)


def write_xlsx(df_all, pred_map, out_path):
    """df_all: 原始 TSV 全表 (含 image); pred_map: index->prediction。写出 8 列 xlsx (去掉 image)。"""
    out = df_all[["index", "category", "question", "answer", "eval", "bbox", "content"]].copy()
    out["prediction"] = out["index"].map(lambda i: pred_map.get(int(i), ""))
    out = out.sort_values("index").reset_index(drop=True)
    # 保证列为字符串 (answer/eval/bbox/content 是 Python literal 字符串, 原样保留)
    for c in OUT_COLS:
        if c in out.columns and c != "index":
            out[c] = out[c].fillna("").astype(str)
    out["index"] = out["index"].astype(int)
    out.to_excel(out_path, index=False)


if __name__ == "__main__":
    main()
