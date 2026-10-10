#!/bin/bash

# Resolve paths from this checkout; override VLMEVAL_ROOT to relocate it.
_EVAL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export VLMEVAL_ROOT="${VLMEVAL_ROOT:-$(cd -- "$_EVAL_SCRIPT_DIR/.." && pwd)}"
WORKDIR="$VLMEVAL_ROOT"
export CDM_TEXLIVE_ROOT="${CDM_TEXLIVE_ROOT:-$WORKDIR/.tools/texlive/2025}"
export NLTK_DATA="${NLTK_DATA:-$WORKDIR/.cache/nltk_data}"
cd -- "$WORKDIR" || exit $?

# =============================================================================
# ocrbench_v21 两阶段分组评测 (full_sft_0915 checkpoint-33528) —— <answer> 格式约束版
# 模型 : models/polyocr/full_sft_0915_qwen35_2b_8p59m_vqaans_hy87_256g/v0-20260915-225503/checkpoint-33528
# 参照 : shell/eval_polyocr_v0-20260908-151800_ckpt22352_0909_org_answerfmt_two_stage.sh
# 数据 : data/ocrbench_v21.tsv (经 LMUData 软链为 ocrbench_v21.tsv)
#
# 与参照脚本的唯一差别 (其它口径完全一致):
#   对 7 个 QA 子类 (cognition VQA cn/en, reasoning VQA cn/en, math QA en,
#   diagram QA en, science QA en) 的 question 末尾追加统一输出格式约束:
#     "Response format: Put the complete final answer in exactly one <answer>...</answer>
#      block. If reasoning is provided, put it in <think>...</think>. Do not output any
#      text outside these blocks. Preserve all requested sub-answers, units, and answer
#      structure."
#   打分时对上述子类只取 <answer>...</answer> 内文作为 predict (规则 + rescue judge 均用抽取后的文本)。
#   实现:
#     - 推理侧: vlmeval/dataset/image_vqa.py OCRBench_v2.build_prompt 读
#       OCRBENCH_V2_ANSWER_FORMAT_TYPES (逗号分隔) 追加后缀; 未设 -> 行为不变。
#     - 打分侧: score_ocrbench_v2_org.py build_predict_result 读 OCRBENCH_ANSWER_BOX_TYPES
#       (逗号分隔) 对该类抽 <answer> 内文; 未匹配标签则原样返回 (no-op)。
#
# 需求:
#   1) 子类分组: 解析/识别/抽取类(19 类)不开 thinking; 问答/推理类(11 类)开 thinking。
#   2) 全部推理完并保存预测 xlsx 后, 再统一离线测评 (不在推理阶段在线打分)。
#   3) 测评: 公式识别(en/cn)走 CDM; text translation cn 全量 judge;
#      QA 类只送 reasoning VQA cn / science QA en 两类 (rescue), 其余 6 个 QA 子类纯规则。
#
# GPU : 4 × A100-80GB (CUDA_VISIBLE_DEVICES=0,1,2,3)。Qwen3VLChat 按 visible GPU 数定 tp_size=4。
#
# STAGE:
#   infer_parse | infer_qa | score | all (默认 all: infer_parse -> infer_qa -> score)
#   OCRBENCH_LLM_JUDGE=0 bash 本脚本         (跳过 translation LLM-judge 与 QA judge)
#   OCRBENCH_QA_JUDGE_TYPES='...' bash 本脚本 (覆盖 QA judge 白名单, 逗号分隔)
#   OCRBENCH_V2_ANSWER_FORMAT_TYPES / OCRBENCH_ANSWER_BOX_TYPES 可在外部覆盖目标子类列表。
# =============================================================================
set -euo pipefail

# ---------- 可配置 ----------
STAGE="${STAGE:-all}"                    # infer_parse | infer_qa | score | all
WORKDIR="$VLMEVAL_ROOT"
VENV="${VENV:-$WORKDIR/.venv}"
PY="${PYTHON:-$VENV/bin/python}"
BASE=eval_polyocr_v0-20260915-225503_ckpt33528_0919_org_answerfmt
OUTROOT="$WORKDIR/outputs/$BASE"
RUN_PARSE_NAME=${BASE}_parse_no_think
RUN_QA_NAME=${BASE}_qa_think
CFG_PARSE="$WORKDIR/config/${RUN_PARSE_NAME}.json"
CFG_QA="$WORKDIR/config/${RUN_QA_NAME}.json"
ORG_TSV="${ORG_TSV:-$WORKDIR/data/ocrbench_v21.tsv}"
case "$ORG_TSV" in /*) ;; *) ORG_TSV="$WORKDIR/$ORG_TSV" ;; esac
MODEL_PATH="${MODEL_PATH:-models/polyocr/full_sft_0915_qwen35_2b_8p59m_vqaans_hy87_256g/v0-20260915-225503/checkpoint-33528}"

# <answer> 格式约束目标子类 (推理侧追加 prompt 后缀; 打分侧抽取 <answer> 内文)
ANSWER_FMT_TYPES="cognition VQA cn,cognition VQA en,reasoning VQA cn,reasoning VQA en,math QA en,diagram QA en,science QA en"

# GPU: 4 × A100-80GB
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# ---------- 子类分组 (与参照脚本一致; 类别名须与 TSV 完全一致) ----------
# 解析/识别/抽取类: 不开 thinking (19 类)。
PARSE_TYPES="chart parsing en,document classification en,document parsing cn,document parsing en,fine-grained text recognition en,formula recognition cn,formula recognition en,full-page OCR cn,full-page OCR en,handwritten answer extraction cn,key information extraction cn,key information extraction en,key information mapping en,table parsing cn,table parsing en,text grounding en,text recognition en,text spotting en,text translation cn"
# 问答/推理类: 开 thinking (11 类)。
QA_TYPES="ASCII art classification en,APP agent en,VQA with position en,cognition VQA cn,cognition VQA en,diagram QA en,math QA en,reasoning VQA cn,reasoning VQA en,science QA en,text counting en"

# ---------- 大模型评测 (LLM-judge) 配置 ----------
JUDGE_ENABLE="${OCRBENCH_LLM_JUDGE:-1}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen3.5-397B-A17B-FP8}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://aistudio.alipay.com/proxy/workflow_62110137:8000/v1}"
JUDGE_KEY="${JUDGE_KEY:-dummy}"
JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-180}"
JUDGE_RETRY="${JUDGE_RETRY:-3}"
export OCRBENCH_JUDGE_RPM="${OCRBENCH_JUDGE_RPM:-2000}"
export OCRBENCH_JUDGE_CACHE_DIR="${OCRBENCH_JUDGE_CACHE_DIR:-$WORKDIR/.cache/ocr_judge_cache_polyocr_ck33528}"
mkdir -p "$OCRBENCH_JUDGE_CACHE_DIR"
SCORE_NPROC="${SCORE_NPROC:-64}"

# ---------- 环境固化 ----------
export PATH="$CDM_TEXLIVE_ROOT/bin/x86_64-linux:/usr/local/bin:$PATH"
export CDM_TEXLIVE_ROOT="${CDM_TEXLIVE_ROOT:-.tools/texlive/2025}"
export NLTK_DATA="${NLTK_DATA:-.cache/nltk_data}"
export NLTK_DATA_PERSIST="${NLTK_DATA_PERSIST:-$WORKDIR/.cache/nltk_data}"

if [ "$STAGE" = "score" ]; then
  export VLMEVAL_NO_TORCH=1
fi
export OCRBENCH_SKIP_LLM_JUDGE="${OCRBENCH_SKIP_LLM_JUDGE:-1}"
export LMUData="${LMUData:-$WORKDIR/data/LMUData}"
mkdir -p "$LMUData"
export VLLM_LOGGING_LEVEL=ERROR VLLM_CONFIGURE_LOGGING=0 VLLM_BATCH_SIZE=256

# ---------- 前置检查 ----------
[ -d "$MODEL_PATH" ] || { echo "[FATAL] 模型目录不存在: $MODEL_PATH" >&2; exit 1; }
[ -f "$MODEL_PATH/config.json" ] || { echo "[FATAL] 模型目录缺 config.json: $MODEL_PATH" >&2; exit 1; }
for f in "$CFG_PARSE" "$CFG_QA"; do
  [ -f "$f" ] || { echo "[FATAL] 缺推理配置: $f" >&2; exit 1; }
  grep -q "$MODEL_PATH" "$f" || { echo "[FATAL] $f 里的 model_path 不是 $MODEL_PATH" >&2; exit 1; }
done

# ---------- 数据集软链 ----------
LMU="$LMUData"
LINK="$LMU/ocrbench_v21.tsv"
if [ ! -f "$ORG_TSV" ]; then echo "[FATAL] 数据集不存在: $ORG_TSV" >&2; exit 1; fi
[ -L "$LINK" ] && { rm -f "$LINK"; ln -sf "$ORG_TSV" "$LINK"; } || { [ -e "$LINK" ] || ln -sf "$ORG_TSV" "$LINK"; }
LOCAL_TSV="$LMU/ocrbench_v21_local.tsv"
[ -f "$LOCAL_TSV" ] && [ "$ORG_TSV" -nt "$LOCAL_TSV" ] && rm -f "$LOCAL_TSV"

# ---------- 工具函数 ----------
ensure_wordnet () {
  local W=$1/corpora
  mkdir -p "$W"
  local EX_PY="$(command -v "${PYTHON:-python3}")"; [ -x "$EX_PY" ] || EX_PY="$PY"
  for pair in "wordnet:wordnet" "omw-1.4:omw-1.4"; do
    local zip=${pair%%:*} name=${pair##*:}
    if [ ! -d "$W/$name" ]; then
      echo "[INFO] 下载 nltk $name -> $W"
      curl -ksSL -o "$W/$name.zip" "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/$zip.zip" \
        && "$EX_PY" -c "import zipfile;zipfile.ZipFile('$W/$name.zip').extractall('$W')" && rm -f "$W/$name.zip" \
        || echo "[WARN] 下载 $name 失败 (METEOR 将退化到 0)"
    fi
  done
}
ensure_wordnet .cache/nltk_data
ensure_wordnet "$NLTK_DATA_PERSIST"

check_texlive () {
  if [ -x "$CDM_TEXLIVE_ROOT/bin/x86_64-linux/pdflatex" ]; then
    echo "[OK] pdflatex 可用 -> CDM 公式评测将正常打分"; return 0
  fi
  echo "[WARN] 未检测到 pdflatex (TeX Live)。"; return 1
}

TL_TAR=tools_archive/texlive.tar.gz
EXTRA_DEB=tools_archive/latex-extra.deb
ensure_texlive () {
  if [ -x "$CDM_TEXLIVE_ROOT/bin/x86_64-linux/pdflatex" ]; then return 0; fi
  if [ ! -f "$TL_TAR" ]; then echo "[FATAL] 缺 TeX Live 且找不到便携包: $TL_TAR" >&2; return 1; fi
  echo "[INFO] (重建) 解压便携 TeX Live -> $CDM_TEXLIVE_ROOT ..."
  mkdir -p "${CDM_TEXLIVE_ROOT%/*}"
  tar xzf "$TL_TAR" -C "${CDM_TEXLIVE_ROOT%/*}" || { echo "[FATAL] texlive 解压失败" >&2; return 1; }
  if [ -f "$EXTRA_DEB" ] && command -v ar >/dev/null 2>&1; then
    local tmpd; tmpd="$(mktemp -d)"; cd "$tmpd"
    ar x "$EXTRA_DEB" data.tar.xz 2>/dev/null && tar xf data.tar.xz 2>/dev/null \
      && cp -rn ./usr/share/texlive/texmf-dist/. "$CDM_TEXLIVE_ROOT/texmf-dist/" 2>/dev/null
    cd - >/dev/null; rm -rf "$tmpd"
  fi
  local bin="$CDM_TEXLIVE_ROOT/bin/x86_64-linux"
  "$bin/mktexlsr" "$CDM_TEXLIVE_ROOT/texmf-dist" >/dev/null 2>&1 || true
  [ -x "$bin/pdflatex" ] && echo "[OK] TeX Live 就绪: $bin/pdflatex" || { echo "[FATAL] pdflatex 仍不可用" >&2; return 1; }
}

LOCAL_PKGS="${VLMEVAL_LOCAL_PKGS:-$WORKDIR/.cache/localpkgs}"
SHADOW="${VLMEVAL_SHADOW_DIR:-$WORKDIR/.cache/shadow}"
LOCAL_PY="$(command -v "${PYTHON:-python3}")"
ensure_localpkgs () {
  if [ -d "$LOCAL_PKGS/nltk" ] && [ -d "$LOCAL_PKGS/numpy" ]; then return 0; fi
  echo "[INFO] (重建) pip 安装打分重依赖到本地 overlay -> $LOCAL_PKGS"
  "$PY" -m pip install --no-input --target="$LOCAL_PKGS" \
    numpy pandas nltk jieba tqdm openpyxl pillow lxml python-Levenshtein pylatexenc scipy beautifulsoup4 || \
    echo "[WARN] 部分包安装失败, 若 NFS 健康可忽略 (改走 .venv 原路径)"
}
ensure_shadow () {
  mkdir -p "$SHADOW/vlmeval/dataset/utils"
  cat > "$SHADOW/vlmeval/__init__.py" <<'EOF'
NFS_VLMEVAL = str(__import__("pathlib").Path(__import__("os").environ["VLMEVAL_ROOT"]) / './vlmeval')
if NFS_VLMEVAL not in __path__: __path__.append(NFS_VLMEVAL)
__version__ = '0.2rc1'
EOF
  cat > "$SHADOW/vlmeval/dataset/__init__.py" <<'EOF'
NFS = str(__import__("pathlib").Path(__import__("os").environ["VLMEVAL_ROOT"]) / './vlmeval/dataset')
if NFS not in __path__: __path__.append(NFS)
EOF
  cat > "$SHADOW/vlmeval/dataset/utils/__init__.py" <<'EOF'
NFS = str(__import__("pathlib").Path(__import__("os").environ["VLMEVAL_ROOT"]) / './vlmeval/dataset/utils')
if NFS not in __path__: __path__.append(NFS)
EOF
}

latest_pred_xlsx () {
  local eval_name=$1
  local d="$OUTROOT/$eval_name"
  [ -d "$d" ] || { echo ""; return; }
  find "$d" -type f -name "${eval_name}_ocrbench_v21.xlsx" -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | awk '{print $2}'
}

# ---------- STAGE: 推理单组 ----------
run_infer_group () {
  local eval_name=$1 cfg=$2 types=$3 label=$4
  echo "============ STAGE=推理 [$label] ($eval_name) ============"
  echo "[INFO] OCRBENCH_V2_EVAL_TYPE=[$types]"
  unset VLMEVAL_NO_TORCH || true
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  export OCRBENCH_V2_EVAL_TYPE="$types"
  # 目标 QA 子类的 question 末尾追加 <answer>/<think> 格式约束 (组内不相交, parse 组无匹配子类 -> no-op)
  export OCRBENCH_V2_ANSWER_FORMAT_TYPES="$ANSWER_FMT_TYPES"
  echo "[INFO] OCRBENCH_V2_ANSWER_FORMAT_TYPES=[$ANSWER_FMT_TYPES]"
  # --mode infer: 只推理写 xlsx, 跳过内置 evaluate, 推理阶段不跑 CDM/judge。
  "$PY" run.py --mode infer --config "$cfg" --work-dir "$OUTROOT" \
    --judge "$JUDGE_MODEL" --judge-base-url "$JUDGE_BASE_URL" --judge-key "$JUDGE_KEY" \
    --judge-retry "$JUDGE_RETRY" \
    --verbose --reuse
  unset OCRBENCH_V2_EVAL_TYPE
  unset OCRBENCH_V2_ANSWER_FORMAT_TYPES
  local cfg_base; cfg_base="$(basename "$cfg")"
  local i=0
  while [ $i -lt 30 ]; do
    if ! pgrep -f "run.py .*--config .*${cfg_base}" >/dev/null 2>&1; then break; fi
    echo "[INFO] 等待 ${eval_name} 的 run.py 退出... ($i)"; sleep 20; i=$((i+1))
  done
  sleep 5
  local pred; pred="$(latest_pred_xlsx "$eval_name")"
  if [ -z "$pred" ] || [ ! -f "$pred" ]; then
    echo "[FATAL] $eval_name 推理结束但未找到预测 xlsx" >&2; exit 1
  fi
  echo "[OK] $eval_name 预测 xlsx: $pred ($(python3 -c "import pandas as pd;print(len(pd.read_excel('$pred')))" 2>/dev/null || echo ?) 行)"
}

# ---------- STAGE: 合并 xlsx + 统一打分 ----------
run_score () {
  echo "============ STAGE=统一打分 (合并两组 xlsx; QA 目标子类抽 <answer> 内文) ============"
  echo "[INFO] judge : $JUDGE_MODEL @ $JUDGE_BASE_URL (key=$JUDGE_KEY, RPM=$OCRBENCH_JUDGE_RPM, cache=$OCRBENCH_JUDGE_CACHE_DIR)"
  ensure_texlive || check_texlive
  export PATH="$CDM_TEXLIVE_ROOT/bin/x86_64-linux:$PATH"
  export NLTK_DATA="${NLTK_DATA:-.cache/nltk_data}"
  if [ "$JUDGE_ENABLE" = "1" ]; then
    export OCRBENCH_SKIP_LLM_JUDGE=''
  else
    export OCRBENCH_SKIP_LLM_JUDGE="${OCRBENCH_SKIP_LLM_JUDGE:-1}"
  fi
  export OCRBENCH_QA_JUDGE_TYPES="${OCRBENCH_QA_JUDGE_TYPES:-reasoning VQA cn,science QA en}"
  export OCRBENCH_QA_JUDGE="${OCRBENCH_QA_JUDGE:-rescue}"
  export OCRBENCH_TEDS_NJOBS="${OCRBENCH_TEDS_NJOBS:-1}"
  export OCRBENCH_PARALLEL_CHUNKS="${OCRBENCH_PARALLEL_CHUNKS:-1}"
  # 打分侧: 对目标 QA 子类只取 <answer>...</answer> 内文作为 predict
  export OCRBENCH_ANSWER_BOX_TYPES="${OCRBENCH_ANSWER_BOX_TYPES:-$ANSWER_FMT_TYPES}"
  export VLMEVAL_NO_TORCH=1
  rm -f /tmp/ocr_judge_ratelimit.state 2>/dev/null || true
  if [ "${SCORE_USE_OVERLAY:-0}" = "1" ] && [ -x "$LOCAL_PY" ]; then
    ensure_localpkgs; ensure_shadow
    export VLMEVAL_LOCAL_PKGS="$LOCAL_PKGS" VLMEVAL_SHADOW_DIR="$SHADOW"
    RUN_PY="$LOCAL_PY"; echo "[INFO] SCORE_USE_OVERLAY=1 -> 本地 overlay 兜底"
  else
    RUN_PY="$PY"
    unset VLMEVAL_LOCAL_PKGS VLMEVAL_SHADOW_DIR 2>/dev/null || true
    echo "[INFO] 用 .venv python (已固化 CDM + judge 依赖, 需 NFS 健康)"
  fi

  parse_xlsx="$(latest_pred_xlsx "$RUN_PARSE_NAME")"
  qa_xlsx="$(latest_pred_xlsx "$RUN_QA_NAME")"
  if [ -z "$parse_xlsx" ] || [ ! -f "$parse_xlsx" ]; then
    echo "[FATAL] 找不到 parse_no_think 组预测 xlsx (先跑 STAGE=infer_parse 或 all)。搜索根: $OUTROOT/$RUN_PARSE_NAME" >&2; exit 1
  fi
  if [ -z "$qa_xlsx" ] || [ ! -f "$qa_xlsx" ]; then
    echo "[FATAL] 找不到 qa_think 组预测 xlsx (先跑 STAGE=infer_qa 或 all)。搜索根: $OUTROOT/$RUN_QA_NAME" >&2; exit 1
  fi
  echo "[INFO] parse xlsx: $parse_xlsx"
  echo "[INFO] qa   xlsx: $qa_xlsx"
  combined="$OUTROOT/${BASE}_combined_ocrbench_v21.xlsx"

  echo "[INFO] 合并两组 xlsx -> $combined ..."
  "$RUN_PY" - "$parse_xlsx" "$qa_xlsx" "$combined" <<'MERGE_PY'
import sys
import pandas as pd
p, q, out = sys.argv[1], sys.argv[2], sys.argv[3]
a = pd.read_excel(p); b = pd.read_excel(q)
need = ['index', 'orig_idx', 'category', 'question', 'answer', 'eval', 'bbox', 'content', 'prediction']
for df in (a, b):
    if 'orig_idx' not in df.columns:
        df['orig_idx'] = df['index'] if 'index' in df.columns else range(len(df))
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f'[FATAL] xlsx 缺列 {miss}; 列: {list(df.columns)}')
c = pd.concat([a[need], b[need]], ignore_index=True).sort_values('index').reset_index(drop=True)
c['orig_idx'] = range(len(c))
c.to_excel(out, index=False)
print(f'[OK] 合并 {len(a)} + {len(b)} -> {len(c)} 行, 子类 {c["category"].nunique()} 类')
print('    parse cats:', sorted(a["category"].unique()))
print('    qa    cats:', sorted(b["category"].unique()))
MERGE_PY

  echo "[INFO] 调用 score_ocrbench_v2_org.py 打分 (answer-box 子类: ${OCRBENCH_ANSWER_BOX_TYPES}; QA judge 白名单: ${OCRBENCH_QA_JUDGE_TYPES}; mode=$OCRBENCH_QA_JUDGE; nproc=$SCORE_NPROC) ..."
  if [ "$JUDGE_ENABLE" = "1" ]; then
    "$RUN_PY" "$WORKDIR/score_ocrbench_v2_org.py" --pred "$combined" --out-dir "$OUTROOT" --nproc "$SCORE_NPROC" \
      --judge "$JUDGE_MODEL" --judge-base-url "$JUDGE_BASE_URL" --judge-key "$JUDGE_KEY" \
      --judge-timeout "$JUDGE_TIMEOUT" --judge-retry "$JUDGE_RETRY"
  else
    "$RUN_PY" "$WORKDIR/score_ocrbench_v2_org.py" --pred "$combined" --out-dir "$OUTROOT" --nproc "$SCORE_NPROC"
  fi
  echo "[DONE] 产物 (在 $OUTROOT):"
  echo "  - ${BASE}_combined_ocrbench_v21_score.json            (13 task type + EN/CN Overall, 官方口径)"
  echo "  - ${BASE}_combined_ocrbench_v21_subcategory_score.json  (30 子类小分)"
  echo "  - ${BASE}_combined_ocrbench_v21_scored.jsonl            (逐条)"
}

cd "$WORKDIR"

case "$STAGE" in
  infer_parse) run_infer_group "$RUN_PARSE_NAME" "$CFG_PARSE" "$PARSE_TYPES" "parse_no_think (enable_thinking=false, 19 类)" ;;
  infer_qa)    run_infer_group "$RUN_QA_NAME"    "$CFG_QA"    "$QA_TYPES"    "qa_think (enable_thinking=true, 11 类)" ;;
  score)       run_score ;;
  all)         run_infer_group "$RUN_PARSE_NAME" "$CFG_PARSE" "$PARSE_TYPES" "parse_no_think (enable_thinking=false, 19 类)"
               run_infer_group "$RUN_QA_NAME"    "$CFG_QA"    "$QA_TYPES"    "qa_think (enable_thinking=true, 11 类)"
               run_score ;;
  *) echo "STAGE 必须是 infer_parse|infer_qa|score|all"; exit 1 ;;
esac
