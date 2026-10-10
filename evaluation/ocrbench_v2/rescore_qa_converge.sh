#!/bin/bash

# Resolve paths from this checkout; override VLMEVAL_ROOT to relocate it.
_EVAL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export VLMEVAL_ROOT="${VLMEVAL_ROOT:-$(cd -- "$_EVAL_SCRIPT_DIR/." && pwd)}"
WORKDIR="$VLMEVAL_ROOT"
export CDM_TEXLIVE_ROOT="${CDM_TEXLIVE_ROOT:-$WORKDIR/.tools/texlive/2025}"
export NLTK_DATA="${NLTK_DATA:-$WORKDIR/.cache/nltk_data}"
cd -- "$WORKDIR" || exit $?

# 收敛重试 judge 失败项: 反复重跑子集打分。已成功的 judge 结果都在本地缓存里(秒回),
# 因此每一轮只有上一轮失败(网关 504/超时)的样本会真正发请求; 每轮降并发以缩短单次
# 延迟, 直到某一轮 0 失败为止 (此时写出的 subset_score*.json 即为最终结果)。
#
# 用法: bash rescore_qa_converge.sh <subset.xlsx> <out-dir>
set -uo pipefail
PRED="${1:?用法: rescore_qa_converge.sh <subset.xlsx> <out-dir>}"
OUT="${2:?用法: rescore_qa_converge.sh <subset.xlsx> <out-dir>}"
NPROC_LIST=(${NPROC_LIST:-8 6 4 2 2})
mkdir -p "$OUT"

pass=0
for np in "${NPROC_LIST[@]}"; do
  pass=$((pass+1))
  log="$OUT/score_pass${pass}_np${np}.log"
  echo "[PASS $pass] nproc=$np -> $log  ($(date '+%H:%M:%S'))"
  NPROC="$np" bash ./rescore_qa_subset.sh "$PRED" "$OUT" > "$log" 2>&1
  fails=$(grep -c 'judge 调用失败' "$log" || true)
  echo "[PASS $pass] judge 失败 $fails 次"
  if [ "$fails" = "0" ]; then
    echo "[DONE] 第 $pass 轮 0 失败, 收敛。最终产物在 $OUT"
    exit 0
  fi
done
echo "[FAIL] ${#NPROC_LIST[@]} 轮后仍有 judge 失败, 请检查 $OUT/score_pass*.log"
exit 1
