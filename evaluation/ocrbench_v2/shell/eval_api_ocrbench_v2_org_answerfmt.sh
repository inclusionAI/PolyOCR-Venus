#!/usr/bin/env bash

# Resolve paths from this checkout; override VLMEVAL_ROOT to relocate it.
_EVAL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export VLMEVAL_ROOT="${VLMEVAL_ROOT:-$(cd -- "$_EVAL_SCRIPT_DIR/.." && pwd)}"
WORKDIR="$VLMEVAL_ROOT"
export CDM_TEXLIVE_ROOT="${CDM_TEXLIVE_ROOT:-$WORKDIR/.tools/texlive/2025}"
export NLTK_DATA="${NLTK_DATA:-$WORKDIR/.cache/nltk_data}"
cd -- "$WORKDIR" || exit $?

# =============================================================================
# ocrbench_v21 单阶段评测 (网关 API 模型) —— antalchemy 推理 + <answer> 格式约束算分
# 模型 : 由 MODEL_API 决定, 默认 doubao-seed-2-0-lite-260428
#        (要跑 claude 就 MODEL_API=claude-opus-4-8; 两者都走 antalchemy /v1/chat/completions, GPT4V 类)
# 数据 : data/ocrbench_v21.tsv (10000 行 / 30 子类, LMUData 软链)
# 命名 : config / outputs / 产物都按模型自动取名
#        eval_<model_slug>_api_ocrbench_v2_org_0916_answerfmt —— 换模型不会和别的模型混在一个目录
# 注意 : config/<run>.json 是每次从 config/eval_api_ocrbench_v2_org_answerfmt.template.json 现场生成的,
#        别手改它 (改模板, 或用 MODEL_API / API_BASE / MODEL_API_KEY 覆盖)
#
# 推理侧参照: shell/eval_gpt5_6_sol_api_ocrbench_v2_org_0822.sh
#   单阶段一次推完 30 子类 (API 模型无 enable_thinking 开关, 不做 parse/qa 两组切分),
#   全局 RPM 门 (API_RPM) 封顶速率, 不撞网关 ~60 RPM 租户配额。
# 算分侧参照: shell/eval_sft0909_v2-20260911-134014_ckpt10_0909_org_answerfmt_two_stage.sh
#   1) 7 个 QA 子类 (cognition/reasoning VQA cn+en, math QA en, diagram QA en, science QA en)
#      question 末尾追加统一输出格式约束 (OCRBENCH_V2_ANSWER_FORMAT_TYPES);
#   2) 打分只取这些子类 <answer>...</answer> 内文 (OCRBENCH_ANSWER_BOX_TYPES, 规则与 judge 同口径);
#   3) QA judge 只送 reasoning VQA cn / science QA en 两类, mode=rescue (规则没过才兜底);
#   4) text translation cn 全量 judge; 公式识别 cn/en 走 CDM; 其余确定性。
#
# judge 口径与 sft0909 answerfmt 完全一致 (同一个部署):
#   Qwen3.5-397B-A17B-FP8 @ https://aistudio.alipay.com/proxy/workflow_62110137:8000/v1
#   -> 30 子类小分与 outputs/eval_sft0909_v{1,2,3}-*_answerfmt 可直接对比。
#   注意这是个临时 workflow, 会休眠 (2026-09-16: 11:14 探活 200, 13:44 就 502 了)。
#   infer 不需要它 (preflight 只警告不拦); score 阶段会先探活, 502 直接 FATAL 退出,
#   避免白跑一小时把 text translation cn 打成 0。拉起 judge 后重跑 score 即可。
#
# 关键 (踩过的坑):
#   - vlmeval 自动加载 <repo>/.env 并覆盖 env; 共享存储 那份 .env 里是错 key (<OLD_API_KEY>)。
#     跑前挪开 (避免覆盖 共享存储 的 <API_KEY>), EXIT 时 trap 还原, 并加锁防两个评测互相踩。
#   - Anthropic 对 thinking 模型收到 sampling 参数直接 400, 网关直传 temperature;
#     vlmeval/api/gpt.py 已加 claude 分支剥离 temperature (CLAUDE_KEEP_TEMPERATURE=1 可关闭)。
#     doubao 走普通 temperature, 不受该分支影响。
#   - thinking 不做开关控制: API 侧无 enable_thinking, 全走网关/模型默认 (故不分 parse/qa 两组)。
#   - preflight 的"图片"那一项是硬门槛: OCRBench 是多模态榜单, 若模型是纯文本的 (或网关不吃
#     OpenAI 格式 image_url), 这一步会 4xx 并把 10000 条推理挡在门外 —— 别让 3.5h 白跑。
#   - antalchemy 域名在部分容器网段解析不到 (内网 DNS 是 ACL 视图; ide-sidecar pod 自 2026-09-15 18:47
#     起持续 NXDOMAIN, 同期 aistudio/matrixllm 正常) -> 先 STAGE=preflight, 在能解析的环境里跑。
#
# STAGE:
#   preflight | infer | score | all (默认 all: preflight -> infer -> score)
#   bash shell/run_api_answerfmt.sh preflight          # 推荐入口 (自动后台化/日志/停止都是它管)
#   API_NPROC=12 API_RPM=48 STAGE=all bash shell/eval_api_ocrbench_v2_org_answerfmt.sh
#   MODEL_API=claude-opus-4-8 bash shell/run_api_answerfmt.sh full   # 换模型
#   OCRBENCH_LLM_JUDGE=0 STAGE=score bash ...   (跳过 judge: translation 置 skip, QA 纯规则)
#   ONLY_TYPES='math QA en' bash ...            (冒烟: 只跑指定子类, 不是全量 30 类)
# =============================================================================
set -euo pipefail

STAGE="${STAGE:-all}"                    # preflight | infer | score | all
WORKDIR="$VLMEVAL_ROOT"
VENV="${VENV:-$WORKDIR/.venv}"
PY="${PYTHON:-$VENV/bin/python}"
ORG_TSV="${ORG_TSV:-$WORKDIR/data/ocrbench_v21.tsv}"
case "$ORG_TSV" in /*) ;; *) ORG_TSV="$WORKDIR/$ORG_TSV" ;; esac

# ---------- 被测模型 / 网关 (都可用 env 覆盖) ----------
MODEL_API="${MODEL_API:-doubao-seed-2-0-lite-260428}"
GW="${GW:-https://antalchemy.alipay.com/v1}"
API_BASE="${API_BASE:-$GW/chat/completions}"

# 运行名 (= config 里的 model key = outputs 子目录名 = 产物文件名前缀), 按模型自动取名
MODEL_SLUG="$(printf '%s' "$MODEL_API" | tr -c 'A-Za-z0-9' '_' | sed 's/_\{2,\}/_/g;s/^_//;s/_$//')"
BASE="${BASE:-eval_${MODEL_SLUG}_api_ocrbench_v2_org_0916_answerfmt}"
EVAL_NAME="$BASE"
OUTROOT="$WORKDIR/outputs/$EVAL_NAME"

# config 每次从模板现场生成: 换模型/换网关都不用改仓库里的 config, 也不会把两个模型的预测
# 混进同一个 xlsx (--reuse 是按 run_key 找 xlsx 续跑的)
CFG_TMPL="$WORKDIR/config/eval_api_ocrbench_v2_org_answerfmt.template.json"
CFG="$WORKDIR/config/${EVAL_NAME}.json"
[ -f "$CFG_TMPL" ] || { echo "[FATAL] 缺 config 模板: $CFG_TMPL" >&2; exit 1; }
RUN_KEY="$EVAL_NAME" NEW_MODEL="$MODEL_API" NEW_API_BASE="$API_BASE" "$PY" - "$CFG_TMPL" "$CFG" <<'GEN_PY'
import json, os, sys
tmpl, dst = sys.argv[1], sys.argv[2]
c = json.load(open(tmpl))
m = c['model'][list(c['model'])[0]]
m['model'] = os.environ['NEW_MODEL']
m['api_base'] = os.environ['NEW_API_BASE']
key = os.environ['RUN_KEY']
c['model'] = {key: m}
json.dump(c, open(dst, 'w'), ensure_ascii=False, indent=4)
print(f'[INFO] 生成 config -> {dst}')
print(f"       run_key={key}\n       model={m['model']}\n       api_base={m['api_base']}")
GEN_PY

# <answer> 格式约束目标子类 (推理侧追加 prompt 后缀; 打分侧抽 <answer> 内文) —— 与 sft0909 answerfmt 完全一致
ANSWER_FMT_TYPES="${ANSWER_FMT_TYPES:-cognition VQA cn,cognition VQA en,reasoning VQA cn,reasoning VQA en,math QA en,diagram QA en,science QA en}"

# ---------- key: source ./.env (<API_KEY>); 挪开 <repo>/.env (错 key <OLD_API_KEY>); trap 还原 ----------
if [ -f "$WORKDIR/.env" ]; then set -a; source "$WORKDIR/.env"; set +a; fi
export OPENAI_API_KEY
# 换网关一般也要换 key。上面的 source 会覆盖外部传入的 OPENAI_API_KEY, 所以在这里再兜一层:
#   MODEL_API_KEY=xxx GW=https://别的网关/v1 bash shell/run_api_answerfmt.sh preflight
if [ -n "${MODEL_API_KEY:-}" ]; then
  export OPENAI_API_KEY="$MODEL_API_KEY"
  echo "[INFO] 用 MODEL_API_KEY 覆盖 .env 里的 OPENAI_API_KEY (换网关时用)"
fi
# 注意: 不把 antalchemy 的 AK 当 judge key (judge 在 aistudio, 别处不外发凭据); judge 见下 JUDGE_KEY
ENV_MOVED=0
# 防呆: .env 会被临时挪开 (vlmeval 自动加载它且里面是错 key <OLD_API_KEY>)。备份已存在 => 另一个同款评测在跑,
# 两边同时挪会互相踩 (谁先还原谁把对方的 key 换掉)。
if [ -f "$WORKDIR/.env.bak_answerfmt" ]; then
  echo "[FATAL] 已存在 $WORKDIR/.env.bak_answerfmt -> 另一个同款评测可能正在跑。" >&2
  echo "        确认无并发后手动删掉再跑: rm -f $WORKDIR/.env.bak_answerfmt" >&2
  exit 1
fi
if [ -f "$WORKDIR/.env" ]; then mv "$WORKDIR/.env" "$WORKDIR/.env.bak_answerfmt"; ENV_MOVED=1; fi
restore_env () {
  if [ "$ENV_MOVED" = "1" ] && [ -f "$WORKDIR/.env.bak_answerfmt" ] && [ ! -f "$WORKDIR/.env" ]; then
    mv "$WORKDIR/.env.bak_answerfmt" "$WORKDIR/.env" && echo "[INFO] $WORKDIR/.env 已还原"
  fi
}
trap restore_env EXIT

# ---------- 并发 + 全局 RPM 限速 (antalchemy 租户 ~60 RPM) ----------
API_NPROC="${API_NPROC:-12}"
export API_RPM="${API_RPM:-48}"          # gpt.py _rpm_gate 全局门; nproc 开大只是填慢条目

# ---------- LLM-judge (text translation cn 全量 + QA 白名单 rescue) —— 同 sft0909 answerfmt ----------
JUDGE_ENABLE="${OCRBENCH_LLM_JUDGE:-1}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen3.5-397B-A17B-FP8}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://aistudio.alipay.com/proxy/workflow_62110137:8000/v1}"
JUDGE_KEY="${JUDGE_KEY:-dummy}"; export JUDGE_KEY          # 自建 judge 无需真 key
JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-180}"
JUDGE_RETRY="${JUDGE_RETRY:-3}"
# 自建 judge 无租户配额, 沿用 sft0909 的 2000; 若改走 antalchemy judge 必须压到 55 (租户 ~60 RPM)
export OCRBENCH_JUDGE_RPM="${OCRBENCH_JUDGE_RPM:-2000}"
# judge 缓存按 run_key 分开: 不同模型的预测文本不同, 混用缓存目录没意义还容易看错
export OCRBENCH_JUDGE_CACHE_DIR="${OCRBENCH_JUDGE_CACHE_DIR:-$WORKDIR/.cache/ocr_judge_cache_${EVAL_NAME}}"
mkdir -p "$OCRBENCH_JUDGE_CACHE_DIR"
SCORE_NPROC="${SCORE_NPROC:-64}"                           # 同 sft0909 (gpt5 那版是 32)

# ---------- 环境固化 ----------
# 推理流量走 openai SDK (httpx): 本环境 requests 走系统 DNS 对 antalchemy 会 gaierror, SDK 通。
# gpt.py 的 generate_inner 读这个 env 切到 _generate_via_openai_sdk。
export VLMEVAL_OPENAI_SDK=1
export PATH="$CDM_TEXLIVE_ROOT/bin/x86_64-linux:/usr/local/bin:$PATH"
export CDM_TEXLIVE_ROOT="${CDM_TEXLIVE_ROOT:-.tools/texlive/2025}"
export NLTK_DATA="${NLTK_DATA:-.cache/nltk_data}"
export NLTK_DATA_PERSIST="${NLTK_DATA_PERSIST:-$WORKDIR/.cache/nltk_data}"
if [ "$STAGE" = "score" ]; then export VLMEVAL_NO_TORCH=1; fi
export OCRBENCH_SKIP_LLM_JUDGE="${OCRBENCH_SKIP_LLM_JUDGE:-1}"   # infer 阶段跳内置 judge
export LMUData="${LMUData:-$WORKDIR/data/LMUData}"
mkdir -p "$LMUData"
export SPLIT_THINK=1
export VLLM_LOGGING_LEVEL=ERROR VLLM_CONFIGURE_LOGGING=0

# ---------- 前置检查 ----------
[ -f "$CFG" ] || { echo "[FATAL] 缺推理配置: $CFG" >&2; exit 1; }
grep -q "$MODEL_API" "$CFG" || { echo "[FATAL] $CFG 里的 model 不是 $MODEL_API" >&2; exit 1; }

# ---------- 数据集软链 (infer 需要) ----------
if [ "$STAGE" != "score" ]; then
  [ -f "$ORG_TSV" ] || { echo "[FATAL] 数据集不存在: $ORG_TSV" >&2; exit 1; }
  ln -sf "$ORG_TSV" "$LMUData/ocrbench_v21.tsv"
  LOCAL_TSV="$LMUData/ocrbench_v21_local.tsv"
  [ -f "$LOCAL_TSV" ] && [ "$ORG_TSV" -nt "$LOCAL_TSV" ] && rm -f "$LOCAL_TSV"
fi

# ---------- 工具函数 ----------
ensure_wordnet () {
  local W=$1/corpora; mkdir -p "$W"
  local EX_PY="$(command -v "${PYTHON:-python3}")"; [ -x "$EX_PY" ] || EX_PY="$PY"
  for pair in "wordnet:wordnet" "omw-1.4:omw-1.4"; do
    local zip=${pair%%:*} name=${pair##*:}
    if [ ! -d "$W/$name" ]; then
      curl -ksSL -o "$W/$name.zip" "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/$zip.zip" \
        && "$EX_PY" -c "import zipfile;zipfile.ZipFile('$W/$name.zip').extractall('$W')" && rm -f "$W/$name.zip" \
        || echo "[WARN] 下载 $name 失败 (METEOR 退化到 0)"
    fi
  done
}
if [ "$STAGE" != "score" ] || [ "${OCRBENCH_SCORE_SKIP_WN:-1}" = "0" ]; then
  ensure_wordnet .cache/nltk_data; ensure_wordnet "$NLTK_DATA_PERSIST"
fi
check_texlive () {
  [ -x "$CDM_TEXLIVE_ROOT/bin/x86_64-linux/pdflatex" ] && { echo "[OK] pdflatex 可用"; return 0; } || { echo "[WARN] 无 pdflatex"; return 1; }
}
ensure_texlive () {
  [ -x "$CDM_TEXLIVE_ROOT/bin/x86_64-linux/pdflatex" ] && return 0
  local TL_TAR=tools_archive/texlive.tar.gz
  local EXTRA_DEB=tools_archive/latex-extra.deb
  [ -f "$TL_TAR" ] || { echo "[FATAL] 缺 TeX Live 便携包: $TL_TAR" >&2; return 1; }
  echo "[INFO] 解压便携 TeX Live -> $CDM_TEXLIVE_ROOT ..."
  mkdir -p "${CDM_TEXLIVE_ROOT%/*}"; tar xzf "$TL_TAR" -C "${CDM_TEXLIVE_ROOT%/*}" || { echo "[FATAL] texlive 解压失败" >&2; return 1; }
  if [ -f "$EXTRA_DEB" ] && command -v ar >/dev/null 2>&1; then
    local tmpd; tmpd="$(mktemp -d)"; (cd "$tmpd" && ar x "$EXTRA_DEB" data.tar.xz 2>/dev/null && tar xf data.tar.xz 2>/dev/null \
      && cp -rn ./usr/share/texlive/texmf-dist/. "$CDM_TEXLIVE_ROOT/texmf-dist/" 2>/dev/null)
    rm -rf "$tmpd"
  fi
  local bin="$CDM_TEXLIVE_ROOT/bin/x86_64-linux"; "$bin/mktexlsr" "$CDM_TEXLIVE_ROOT/texmf-dist" >/dev/null 2>&1 || true
  [ -x "$bin/pdflatex" ] && echo "[OK] TeX Live 就绪" || { echo "[FATAL] pdflatex 仍不可用" >&2; return 1; }
}

latest_pred_xlsx () {
  local d="$OUTROOT/$1"; [ -d "$d" ] || { echo ""; return; }
  find "$d" -type f -name "${1}_ocrbench_v21.xlsx" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | awk '{print $2}'
}

# ---------- STAGE: preflight (judge / 网关 DNS / 模型 文本 / 图片 / answerfmt) ----------
run_preflight () {
  echo "============ STAGE=preflight [judge 连通性 + 网关 DNS + $MODEL_API 文本/图片/answerfmt] ============"
  [ "$JUDGE_ENABLE" = "1" ] || echo "[INFO] OCRBENCH_LLM_JUDGE=0 -> 跳过 judge 连通性检查"
  # 从生成的 config 里取真实 max_tokens, 让 preflight 用它打一发 (超大值被网关拒的情况得在这儿暴露)
  local max_tok; max_tok="$("$PY" -c "
import json
print(list(json.load(open('$CFG'))['model'].values())[0].get('max_tokens', 2048))")"
  VLMEVAL_NO_TORCH=1 API_BASE_URL="$API_BASE" MODEL_API="$MODEL_API" MAX_TOKENS="$max_tok" \
    JUDGE_MODEL="$JUDGE_MODEL" JUDGE_BASE_URL="$JUDGE_BASE_URL" JUDGE_KEY="$JUDGE_KEY" \
    JUDGE_ENABLE="$JUDGE_ENABLE" \
  "$PY" -u - <<'PREFLIGHT_PY'
import base64, io, json, os, socket, sys, time
import requests
from PIL import Image, ImageDraw

GW = os.environ['API_BASE_URL'].split('/chat/completions')[0].rstrip('/')   # 取 /v1 根, 即 SDK base_url
MODEL = os.environ['MODEL_API']
H = {'Content-Type': 'application/json', 'Authorization': f"Bearer {os.environ['OPENAI_API_KEY']}"}   # 只读 env, 不打印
HOST = GW.split('//', 1)[1].split('/', 1)[0]
fail = []

# 模型调用通道: openai SDK (与用户手测同一写法)。本环境实测: requests/curl 走系统 DNS 会
# gaierror, 而 SDK(httpx) 同秒能通 -> 模型侧一律走 SDK; judge 走 aistudio (DNS 正常) 用 requests。
from openai import OpenAI
# max_retries=4: 与用户手测差异的关键 —— 你的 test.py 默认 max_retries=2 带指数退避,
# 之前我探针全关 (0) 才显得"必挂"。链路是抖的, 重试才是正确姿势。
CLIENT = OpenAI(base_url=GW, api_key=os.environ['OPENAI_API_KEY'], max_retries=4, timeout=180.0)

def scall(tag, payload, timeout=180):
    t0 = time.time()
    try:
        resp = CLIENT.with_options(timeout=timeout).chat.completions.create(**payload)
    except Exception as e:
        fail.append(tag); print(f'-- {tag}: EXC {type(e).__name__}: {str(e)[:200]}'); return None
    dt = time.time() - t0
    msg = resp.choices[0].message
    content = msg.content or ''
    rc = getattr(msg, 'reasoning_content', None)
    u = resp.usage or None
    print(f"-- {tag}: SDK 200 ({dt:.1f}s) model={resp.model!r} finish={resp.choices[0].finish_reason} "
          f"out_tokens={getattr(u, 'completion_tokens', None)}" + (f" reasoning_content={len(rc)}ch" if rc else ''))
    print(f"   content[:240]: {str(content)[:240]!r}")
    if not content.strip():
        print('   [WARN] content 为空'); fail.append(tag + '(empty)')
    return resp

def call(base, key_hdr, tag, payload, timeout=180):
    t0 = time.time()
    try:
        r = requests.post(f"{base.rstrip('/')}/chat/completions", headers=key_hdr,
                          data=json.dumps(payload), timeout=timeout)
    except Exception as e:
        fail.append(tag); print(f'-- {tag}: EXC {type(e).__name__}: {str(e)[:160]}'); return None
    dt = time.time() - t0
    try:
        j = r.json()
    except Exception:
        print(f'-- {tag}: HTTP {r.status_code} ({dt:.1f}s) 非 JSON: {r.text[:200]}'); fail.append(tag); return None
    if r.status_code != 200:
        print(f'-- {tag}: HTTP {r.status_code} ({dt:.1f}s) {json.dumps(j, ensure_ascii=False)[:300]}')
        fail.append(tag); return None
    msg = j['choices'][0]['message']
    extra = []
    if msg.get('reasoning_content'): extra.append(f"reasoning_content={len(msg['reasoning_content'])}ch")
    u = j.get('usage') or {}
    print(f"-- {tag}: HTTP 200 ({dt:.1f}s) model={j.get('model')} finish={j['choices'][0].get('finish_reason')} "
          f"out_tokens={u.get('completion_tokens')} {' '.join(extra)}")
    print(f"   content[:240]: {str(msg.get('content'))[:240]!r}")
    if not str(msg.get('content') or '').strip():
        print('   [WARN] content 为空'); fail.append(tag + '(empty)')
    return j

# 0) judge (独立于网关; 502 = 那个临时 workflow 没起来, 打分阶段会缺 translation/QA rescue)
if os.environ.get('JUDGE_ENABLE') == '1':
    jm = os.environ['JUDGE_MODEL']
    jh = {'Content-Type': 'application/json', 'Authorization': f"Bearer {os.environ['JUDGE_KEY']}"}
    j = call(os.environ['JUDGE_BASE_URL'], jh, f'judge {jm}',
             {'model': jm, 'max_tokens': 32, 'messages': [{'role': 'user', 'content': 'Reply with exactly: OK'}]},
             timeout=120)
    if j is None:
        print('   [提示] judge 起不来可以先跑 STAGE=infer, 等它拉起后再 STAGE=score')

print(f'== DNS (仅提示, 不拦截: 模型流量走 SDK, 不依赖系统 DNS): {HOST}')
try:
    print('   ->', socket.getaddrinfo(HOST, 443, socket.AF_INET)[0][4][0])
except Exception:
    print('   -> 系统 DNS 解析不到 (requests/curl 会挂), 无碍: 模型调用走 openai SDK 通道')

img = Image.new('RGB', (420, 120), 'white')
d = ImageDraw.Draw(img); d.text((14, 50), 'OCRBench preflight 2026', fill='black')
buf = io.BytesIO(); img.save(buf, format='PNG')
b64 = base64.b64encode(buf.getvalue()).decode()

sys_msg = 'You are a helpful assistant.'
fmt = ('\n\nResponse format: Put the complete final answer in exactly one <answer>...</answer> block. '
       'If reasoning is provided, put it in <think>...</think>. Do not output any text outside these blocks.')
# 1) 纯文本 (验证 temperature/路径)
scall(f'{MODEL} text', {'model': MODEL, 'max_tokens': 64, 'temperature': 0, 'messages': [
    {'role': 'system', 'content': sys_msg}, {'role': 'user', 'content': 'Reply with exactly: OK'}]})
# 2) 图片 (验证网关接受 OpenAI 格式 image_url + base64)
scall(f'{MODEL} image', {'model': MODEL, 'max_tokens': 512, 'temperature': 0, 'messages': [
    {'role': 'system', 'content': sys_msg},
    {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}', 'detail': 'high'}},
                                 {'type': 'text', 'text': 'What text is in this image? Answer with the text only.'}]}]})
# 3) <answer> 格式约束是否被遵守 (answerfmt 算分依赖它)
scall(f'{MODEL} answerfmt', {'model': MODEL, 'max_tokens': 1024, 'temperature': 0, 'messages': [
    {'role': 'system', 'content': sys_msg},
    {'role': 'user', 'content': 'A shop sells 3 apples at 2 USD each. What is the total cost in USD?' + fmt}]})
# 4) 用 config 里真实的 max_tokens 打一发: 不少模型/网关会拒超大值 (前三发都是小值, 测不出来)。
#    不在这里拦住, 就是 10000 条推理每条都 400, 跑完 3.5h 拿一堆空预测。
MT = int(os.environ.get('MAX_TOKENS') or 0)
if MT > 1024:
    scall(f'{MODEL} max_tokens={MT}', {'model': MODEL, 'max_tokens': MT, 'temperature': 0, 'messages': [
        {'role': 'system', 'content': sys_msg},
        {'role': 'user', 'content': 'Reply with exactly: OK'}]}, timeout=300)

if fail:
    hints = []
    if any('image' in f for f in fail):
        hints.append('image 失败 -> 这个模型/网关可能不吃多模态 (OCRBench 是图文榜, 必须支持图片)')
    if any('max_tokens' in f for f in fail):
        hints.append('max_tokens 失败 -> 把模板里的 max_tokens 调到该模型上限 (如 8192/16384)')
    if 'claude' in MODEL.lower() and any('text' in f for f in fail):
        hints.append('text 就报 400 -> 试 CLAUDE_KEEP_TEMPERATURE=1 (网关自己吞 temperature 的情况)')
    sys.exit(f'[FATAL] preflight 失败项: {fail}. 先解决再开 10000 条推理。\n  ' + '\n  '.join(hints or ['看上面的状态码/报文定位']))
print('[OK] preflight 全通过')
PREFLIGHT_PY
}

# ---------- 预检: score/all 启用 judge 需 key ----------
case "$STAGE" in score|all)
  if [ "$JUDGE_ENABLE" = "1" ] && [ -z "${JUDGE_KEY:-}" ]; then
    echo "[FATAL] 启用 LLM-judge 但无 JUDGE_KEY。OCRBENCH_LLM_JUDGE=0 可跳过 judge。" >&2; exit 1; fi
  ;; esac

# ---------- STAGE: API 推理 (全 30 类, 7 个 QA 子类带 <answer> 格式约束) ----------
run_infer () {
  echo "============ STAGE=API推理 [$MODEL_API 单阶段 全30类 + answerfmt] ($EVAL_NAME) ============"
  echo "[INFO] 调用通道=openai SDK (用户 test.py 同款)  workers=${API_WORKERS:-8}  rpm=$API_RPM  max_tokens=${MAX_TOKENS:-32768}"
  echo "[INFO] 链路断着会指数退避无限重试 (心跳打日志), 好窗口自动续跑; JSONL 逐条落盘可断点续跑"
  # 防呆: 同 run_key 的推理进程已在跑就别再起第二个 (两个进程写同一 ckpt/xlsx 会互相踩)
  local running; running="$(pgrep -af "infer_api_sdk_ocrbench_v2.py .*${EVAL_NAME}" | head -1 || true)"
  if [ -n "$running" ]; then
    echo "[FATAL] 已有同 run_key 的推理进程在跑:" >&2; echo "        $running" >&2
    echo "        等它结束或 kill 后再来; 重跑本命令会从 ckpt 续跑。" >&2; exit 1
  fi
  if [ -n "${ONLY_TYPES:-}" ]; then
    # 冒烟用: 只跑指定子类 (产物也只有这些行, 分数不可与全量结果对比)
    export OCRBENCH_V2_EVAL_TYPE="$ONLY_TYPES"
    echo "[WARN] ONLY_TYPES=[$ONLY_TYPES] -> 只推理这些子类, 不是全量 30 类"
  else
    unset OCRBENCH_V2_EVAL_TYPE        # 不过滤 -> 跑全部 30 子类
  fi
  unset VLMEVAL_NO_TORCH || true
  # 目标 QA 子类的 question 末尾追加 <answer>/<think> 格式约束 (其余 23 类 prompt 不变)
  export OCRBENCH_V2_ANSWER_FORMAT_TYPES="$ANSWER_FMT_TYPES"
  echo "[INFO] OCRBENCH_V2_ANSWER_FORMAT_TYPES=[$ANSWER_FMT_TYPES]"
  # 推理走独立驱动 infer_api_sdk_ocrbench_v2.py (不再用 vlmeval GPT4V/requests 通道):
  # 调用即用户 test.py 的 openai SDK 方式; 7 个 QA 子类 <answer> 后缀由驱动内追加 (与打分侧对齐)
  local only_args=""
  [ -n "${ONLY_TYPES:-}" ] && only_args="--only-types $ONLY_TYPES"
  "$PY" infer_api_sdk_ocrbench_v2.py \
    --tsv "$ORG_TSV" \
    --out-dir "$OUTROOT/$EVAL_NAME" \
    --xlsx-name "${EVAL_NAME}_ocrbench_v21.xlsx" \
    --model "$MODEL_API" \
    --base-url "${API_BASE%/chat/completions}" \
    --workers "${API_WORKERS:-8}" \
    --rpm "$API_RPM" \
    --max-tokens "${MAX_TOKENS:-32768}" \
    --answer-fmt-types "${ANSWER_FMT_TYPES:-cognition VQA cn,cognition VQA en,reasoning VQA cn,reasoning VQA en,math QA en,diagram QA en,science QA en}" \
    $only_args
  local pred; pred="$(latest_pred_xlsx "$EVAL_NAME")"
  [ -n "$pred" ] && [ -f "$pred" ] || { echo "[FATAL] 推理结束但无预测 xlsx" >&2; exit 1; }
  echo "[OK] 预测 xlsx: $pred ($("$PY" -c "import pandas as pd;print(len(pd.read_excel('$pred')))" 2>/dev/null || echo ?) 行)"
}

# ---------- STAGE: 归一化单 xlsx + 离线打分 (口径同 sft0909 answerfmt) ----------
run_score () {
  echo "============ STAGE=统一打分 (公式 CDM + translation 全量 judge + QA 白名单 rescue + answer-box 抽取) ============"
  if [ "$JUDGE_ENABLE" = "1" ]; then
    echo "[INFO] LLM-judge 启用: $JUDGE_MODEL @ $JUDGE_BASE_URL (限流 ${OCRBENCH_JUDGE_RPM} RPM, cache=$OCRBENCH_JUDGE_CACHE_DIR)"
    # 探活锁: 那个临时 workflow 会休眠, 冷启动第一次可能 502 (2026-09-16 13:44 探 502, 13:5x 再探就 200)。
    # 所以重试 4 次 / 间隔 30s (~2min) 再判死。不拦的话 text translation cn 会整类判 0
    # + reasoning VQA cn / science QA en 失去 rescue 兜底, 白跑一小时还得重来。
    local jcode='' i
    for i in 1 2 3 4; do
      jcode="$(timeout 70 curl -s -o /dev/null -w '%{http_code}' -m 60 -X POST "$JUDGE_BASE_URL/chat/completions" \
        -H 'Content-Type: application/json' -H "Authorization: Bearer $JUDGE_KEY" \
        -d "{\"model\":\"$JUDGE_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"reply OK\"}],\"max_tokens\":8}" 2>/dev/null)" || jcode=000
      echo "[INFO] judge 探活 第${i}次 -> HTTP $jcode"
      [ "$jcode" = "200" ] && break
      [ "$i" -lt 4 ] && { echo "[INFO] 可能是 workflow 冷启动, 30s 后重试 ..."; sleep 30; }
    done
    if [ "$jcode" != "200" ]; then
      if [ "${FORCE_SCORE:-0}" = "1" ]; then
        echo "[WARN] judge 回 $jcode 但 FORCE_SCORE=1 -> 继续 (translation/QA rescue 分数不可信)"
      else
        echo "[FATAL] judge ($JUDGE_MODEL) 回 $jcode -> workflow_62110137 没起来, 打分不可信。" >&2
        echo "        拉起 judge 后再跑: bash shell/run_api_answerfmt.sh score" >&2
        echo "        或明确不要 judge : OCRBENCH_LLM_JUDGE=0 bash shell/run_api_answerfmt.sh score" >&2
        echo "        或强行打分       : FORCE_SCORE=1 bash shell/run_api_answerfmt.sh score" >&2
        exit 1
      fi
    else
      echo "[OK] judge 可用 (HTTP 200)"
    fi
  else
    echo "[INFO] LLM-judge 跳过 (OCRBENCH_LLM_JUDGE=0): text translation cn 置 skip; QA 纯规则"
  fi
  ensure_texlive || check_texlive
  export PATH="$CDM_TEXLIVE_ROOT/bin/x86_64-linux:$PATH"
  export NLTK_DATA="${NLTK_DATA:-.cache/nltk_data}"
  if [ "$JUDGE_ENABLE" = "1" ]; then export OCRBENCH_SKIP_LLM_JUDGE=''; else export OCRBENCH_SKIP_LLM_JUDGE="${OCRBENCH_SKIP_LLM_JUDGE:-1}"; fi
  # ---- 与 sft0909 answerfmt 一致的打分口径 ----
  export OCRBENCH_ANSWER_BOX_TYPES="${OCRBENCH_ANSWER_BOX_TYPES:-$ANSWER_FMT_TYPES}"   # 只取 <answer> 内文
  export OCRBENCH_QA_JUDGE_TYPES="${OCRBENCH_QA_JUDGE_TYPES:-reasoning VQA cn,science QA en}"
  export OCRBENCH_QA_JUDGE="${OCRBENCH_QA_JUDGE:-rescue}"                              # 规则没过才兜底 judge
  export OCRBENCH_TEDS_NJOBS="${OCRBENCH_TEDS_NJOBS:-1}"
  # 只影响并行度不影响分数 (sft0909 那版是 1, gpt5 那版是 32; 经验文档推荐 8)
  export OCRBENCH_PARALLEL_CHUNKS="${OCRBENCH_PARALLEL_CHUNKS:-8}"
  export VLMEVAL_NO_TORCH=1
  unset API_RPM                    # judge 自有 OCRBENCH_JUDGE_RPM 节流, 不要推理限速门干扰
  rm -f /tmp/ocr_judge_ratelimit.state 2>/dev/null || true

  local pred; pred="$(latest_pred_xlsx "$EVAL_NAME")"
  [ -n "$pred" ] && [ -f "$pred" ] || { echo "[FATAL] 无预测 xlsx, 先跑 STAGE=infer 或 all。搜索根: $OUTROOT/$EVAL_NAME" >&2; exit 1; }
  echo "[INFO] 单 xlsx: $pred"
  local combined="$OUTROOT/${EVAL_NAME}_combined_ocrbench_v21.xlsx"

  # 单输入归一化: 确保 orig_idx 列 (score 脚本容错缺列用行号), 写 combined
  "$PY" - "$pred" "$combined" <<'NORM_PY'
import sys, pandas as pd
src, out = sys.argv[1], sys.argv[2]
df = pd.read_excel(src)
hard = ['index','category','question','answer','eval','bbox','content','prediction']
miss = [c for c in hard if c not in df.columns]
if miss: raise SystemExit(f'[FATAL] xlsx 缺列 {miss}; 列: {list(df.columns)}')
df = df.sort_values('index').reset_index(drop=True)
if 'orig_idx' not in df.columns:
    df['orig_idx'] = range(len(df))
df.to_excel(out, index=False)
print(f'[OK] 归一化 {len(df)} 行, 子类 {df["category"].nunique()} 类 -> {out}')
NORM_PY

  echo "[INFO] 调用 score_ocrbench_v2_org.py 打分 (answer-box 子类: ${OCRBENCH_ANSWER_BOX_TYPES}; QA judge 白名单: ${OCRBENCH_QA_JUDGE_TYPES}; mode=$OCRBENCH_QA_JUDGE; nproc=$SCORE_NPROC) ..."
  if [ "$JUDGE_ENABLE" = "1" ]; then
    "$PY" "$WORKDIR/score_ocrbench_v2_org.py" --pred "$combined" --out-dir "$OUTROOT" --nproc "$SCORE_NPROC" \
      --judge "$JUDGE_MODEL" --judge-base-url "$JUDGE_BASE_URL" --judge-key "$JUDGE_KEY" \
      --judge-timeout "$JUDGE_TIMEOUT" --judge-retry "$JUDGE_RETRY"
  else
    "$PY" "$WORKDIR/score_ocrbench_v2_org.py" --pred "$combined" --out-dir "$OUTROOT" --nproc "$SCORE_NPROC"
  fi
  echo "[DONE] 产物 (在 $OUTROOT):"
  echo "  - ${EVAL_NAME}_combined_ocrbench_v21_score.json            (13 task type + EN/CN Overall)"
  echo "  - ${EVAL_NAME}_combined_ocrbench_v21_subcategory_score.json  (30 子类小分)"
  echo "  - ${EVAL_NAME}_combined_ocrbench_v21_scored.jsonl            (逐条)"
}

cd "$WORKDIR"
mkdir -p "$OUTROOT" "$WORKDIR/logs"

case "$STAGE" in
  preflight) run_preflight ;;
  infer)     run_infer ;;
  score)     run_score ;;
  all)       run_preflight; run_infer; run_score ;;
  *) echo "STAGE 必须是 preflight|infer|score|all"; exit 1 ;;
esac

restore_env
echo "[run $(date '+%F %T')] ALL DONE."
