#!/bin/bash

# Resolve paths from this checkout; override VLMEVAL_ROOT to relocate it.
_EVAL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export VLMEVAL_ROOT="${VLMEVAL_ROOT:-$(cd -- "$_EVAL_SCRIPT_DIR/." && pwd)}"
WORKDIR="$VLMEVAL_ROOT"
export CDM_TEXLIVE_ROOT="${CDM_TEXLIVE_ROOT:-$WORKDIR/.tools/texlive/2025}"
export NLTK_DATA="${NLTK_DATA:-$WORKDIR/.cache/nltk_data}"
cd -- "$WORKDIR" || exit $?

# 用外部 OpenAI 兼容 judge (aistudio proxy 上的 vLLM) 只重测受 LLM-judge 口径影响的子类:
#   默认 9 类: APP agent en / cognition VQA en·cn / diagram QA en / math QA en
#              / reasoning VQA en·cn / science QA en / text translation cn
#   收窄: 设 OCRBENCH_QA_JUDGE_TYPES='reasoning VQA cn,science QA en' 之类,
#         则只有白名单里的 QA 子类送 judge, 其余 QA 子类保留纯规则分 (翻译子类不受此影响)。
# 子集由 build_qa_subset.py 从全量预测 xlsx 抽取; 重测后用 merge_qa_rescore.py 与全量参考
# scored.jsonl 合并聚合出完整 score.json。
#
# !!! build_qa_subset.py / 本脚本 / merge(finalize)_qa_rescore.py 三步必须用同一份 !!!
# !!! OCRBENCH_QA_JUDGE_TYPES, 否则 merge 的条数/越界校验会 FATAL。              !!!
#
# 用法: bash rescore_qa_subset.sh <subset.xlsx> <out-dir>
#   OCRBENCH_QA_JUDGE_TYPES='reasoning VQA cn,science QA en' \
#     bash rescore_qa_subset.sh <subset.xlsx> <out-dir>
set -euo pipefail

WORKDIR="$VLMEVAL_ROOT"
PY="${PYTHON:-$WORKDIR/.venv/bin/python}"

JUDGE_MODEL="${JUDGE_MODEL:-Qwen3.5-397B-A17B-FP8}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://aistudio.alipay.com/proxy/workflow_62110137:8000/v1}"
JUDGE_TAG="${JUDGE_TAG:-qwen35_397b_a17b_fp8}"
NPROC="${NPROC:-64}"

PRED="${1:?用法: rescore_qa_subset.sh <subset.xlsx> <out-dir>}"
OUTROOT="${2:?用法: rescore_qa_subset.sh <subset.xlsx> <out-dir>}"
mkdir -p "$OUTROOT"

# judge 缓存放本地盘: 共享存储 NFS 已满且缓存每次 put 会整文件重写, 走 NFS 会 O(N^2) 拖死
export OCRBENCH_JUDGE_CACHE_DIR="${OCRBENCH_JUDGE_CACHE_DIR:-$WORKDIR/.cache/ocr_judge_cache_${JUDGE_TAG}}"
mkdir -p "$OCRBENCH_JUDGE_CACHE_DIR"

export OCRBENCH_SKIP_LLM_JUDGE=''          # 必须是空串: 任意非空串(含 '0')都会被当成 truthy
export OCRBENCH_QA_JUDGE="${OCRBENCH_QA_JUDGE:-rescue}"
# 只对白名单里的 QA 子类送 judge, 其余 QA 子类保留纯规则分。
# 不设 -> 8 个 QA 子类全 judge (历史口径)。示例: 'reasoning VQA cn,science QA en'
export OCRBENCH_QA_JUDGE_TYPES="${OCRBENCH_QA_JUDGE_TYPES-}"
export OCRBENCH_JUDGE_RPM="${OCRBENCH_JUDGE_RPM:-2000}"
export OCRBENCH_TEDS_NJOBS=1
export OCRBENCH_PARALLEL_CHUNKS=1          # 单进程: judge 并发靠 nproc 线程池
export VLMEVAL_NO_TORCH=1
export CDM_TEXLIVE_ROOT="${CDM_TEXLIVE_ROOT:-.tools/texlive/2025}"
rm -f /tmp/ocr_judge_ratelimit.state 2>/dev/null || true

echo "[INFO] judge   : $JUDGE_MODEL @ $JUDGE_BASE_URL"
echo "[INFO] pred    : $PRED"
echo "[INFO] out     : $OUTROOT"
echo "[INFO] cache   : $OCRBENCH_JUDGE_CACHE_DIR"
echo "[INFO] mode    : OCRBENCH_QA_JUDGE=$OCRBENCH_QA_JUDGE nproc=$NPROC"
if [ -n "$OCRBENCH_QA_JUDGE_TYPES" ]; then
  echo "[INFO] QA judge 子类白名单: $OCRBENCH_QA_JUDGE_TYPES (其余 QA 子类纯规则)"
else
  echo "[INFO] QA judge 子类白名单: (未设 -> 8 个 QA 子类全量 judge)"
fi

exec "$PY" "$WORKDIR/score_ocrbench_v2_org.py" \
  --pred "$PRED" \
  --out-dir "$OUTROOT" \
  --nproc "$NPROC" \
  --judge "$JUDGE_MODEL" \
  --judge-base-url "$JUDGE_BASE_URL" \
  --judge-key "dummy" \
  --judge-timeout 180 \
  --judge-retry 3
