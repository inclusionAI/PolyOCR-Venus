# OCRBench v2.1 评测

本目录是 PolyOCR-Venus 项目的评测模块，完整项目介绍与技术报告见[仓库首页](../../README.md)。报告中的 **OCRBench v2.1** 指本项目修订标注和评分流程后的协议，数据集名与数据文件统一命名为 `ocrbench_v21`。复现报告结果需使用相同的修订标注和评分 / judge 配置，不能直接与原始 OCRBench v2 分数比较。

导入时清理了硬编码 API 凭据，并将机器相关的路径改成项目相对路径或环境变量。逐文件来源、SHA-256 和修改类型记录在 [`SOURCE_MANIFEST.json`](SOURCE_MANIFEST.json)。VLMEvalKit 的模型与数据集注册表会导入其他模块，因此保留完整 `vlmeval/` 源码以满足导入依赖；仅收录与 OCRBench v2 有关的项目级配置、运行脚本和工具。

## 入口

| 路径 | 用途 |
| --- | --- |
| `run.py` | VLMEvalKit 配置式推理入口 |
| `infer_api_sdk_ocrbench_v2.py` | OpenAI SDK 推理、断点续跑、导出 XLSX |
| `score_ocrbench_v2_org.py` | 对已有预测 XLSX 打分 |
| `check_cdm_deps.py` | CDM Python / TeX Live 依赖自检 |
| `vlmeval/dataset/image_vqa.py` | OCRBench v2 数据集及提示词处理 |
| `vlmeval/dataset/utils/ocrbrnch_v2_eval.py` | 子类评分、judge 和汇总逻辑 |
| `vlmeval/dataset/utils/Ocrbench_v2/` | OCR / VQA / TEDS / IoU / spotting 等指标 |
| `vlmeval/dataset/utils/cdm/` | CDM 公式渲染与匹配 |
| `shell/`、`config/` | 参考运行脚本与配套配置 |
| `build_qa_subset.py`、`merge_qa_rescore.py`、`finalize_qa_rescore.py` | QA 子集重评分及结果合并 |
| `tools/` | 数据修正、单次评分和合并评分工具 |

评测参考脚本只有两份：

- **本地模型**（两阶段）：[`shell/eval_polyocr_v0-20260915-225503_ckpt33528_0919_org_answerfmt_two_stage.sh`](shell/eval_polyocr_v0-20260915-225503_ckpt33528_0919_org_answerfmt_two_stage.sh)，配套 `config/` 中同前缀的 `parse_no_think.json` 和 `qa_think.json`。解析 / 识别类为 19 类，问答 / 推理类为 11 类。
- **闭源 API 模型**：[`shell/eval_api_ocrbench_v2_org_answerfmt.sh`](shell/eval_api_ocrbench_v2_org_answerfmt.sh)，每次运行从 `config/eval_api_ocrbench_v2_org_answerfmt.template.json` 现场生成 config。

两者打分口径一致（Gen 3：两阶段分 think 组 + 7 个 QA 子类 `<answer>` 格式约束 + rescue judge 白名单收窄至 reasoning VQA cn / science QA en）。

## 环境与数据

以下命令均在本目录执行。完整评测面向 Linux 环境，使用 Python 3.11 或更高版本；本次导入用 Python 3.11 做了语法检查。

```bash
# 在仓库根目录下载数据（需先安装 Git LFS）
git lfs install
git lfs pull

cd evaluation/ocrbench_v2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_ocrbench_v2.txt
```

按运行方式选择依赖：

| 文件 | 范围 |
| --- | --- |
| 仓库根目录 `requirements.txt` | 默认安装入口，转到本目录的完整评测清单 |
| `requirements_ocrbench_v2.txt` | API 推理、完整评分及 VLMEvalKit 框架导入依赖；补充 CDM 所需的 `beautifulsoup4`、`PyMuPDF`、`scipy` 等 |
| `requirements_api.txt` | 仅独立 API 推理驱动，含 OpenAI SDK、Pandas、Pillow、Excel 导出依赖等 |
| `requirements_vllm.txt` | 完整评测清单加 vLLM，供 `use_vllm=true` 的本地模型配置使用 |

本目录原有 `requirements.txt`（上游完整框架清单）、`requirements_eval.txt`（源目录精简清单）和 `requirements_frozen.txt`（已有环境记录）均原样保留。不要仅凭原有精简清单安装完整评分环境：其缺少部分直接依赖。`VLMEVAL_NO_TORCH=1` 也不会绕过框架注册表的所有导入。

可选安装命令（仍在本目录）：

```bash
# 仅 API 推理，不含完整评分环境
python -m pip install -r requirements_api.txt
# 或：需要 vLLM 的本地 GPU 推理及评分
python -m pip install -r requirements_vllm.txt
```

新清单沿用源代码已有的版本约束，没有给缺少记录的包编造锁定版本。源环境记录未包含 vLLM 版本，部署时需按实际 GPU/CUDA 与模型选择兼容版本。本次仅核验依赖声明和导入覆盖，未执行完整安装或重建 GPU 环境。

公式评分需 TeX Live 的 `pdflatex`、`kpsewhich`、CJK 和 gkai 字体。手写答案指标需 NLTK 的 `wordnet`、`omw-1.4` 语料。配置 `CDM_TEXLIVE_ROOT` 和 `NLTK_DATA` 后检查：

```bash
export NLTK_DATA="$PWD/.cache/nltk_data"
python -m nltk.downloader -d "$NLTK_DATA" wordnet omw-1.4
# CDM_TEXLIVE_ROOT 指向已安装的 TeX Live 根目录
python check_cdm_deps.py
```

TeX Live 需包含 `collection-latexrecommended`、`collection-latexextra`、`collection-langcjk`、`collection-langchinese`。这些系统组件和 NLTK 语料不是 pip 包，不能通过 requirements 安装；`check_cdm_deps.py` 会检查实际可用性。

评测数据已放在 `data/ocrbench_v21.tsv`（10000 行 / 30 子类，图像以 base64 内嵌；约 1.4 GB，使用 Git LFS 存储，克隆后请运行 `git lfs pull`）。也可通过 `--tsv` / `ORG_TSV` 指定其他数据。使用 `run.py` 时，在 `LMUData` 中创建链接：

```bash
export LMUData="$PWD/data/LMUData"
mkdir -p "$LMUData"
ln -s "$PWD/data/ocrbench_v21.tsv" "$LMUData/ocrbench_v21.tsv"
```

## API 推理

在环境中设置 `OPENAI_API_KEY`；可复制 `.env.example` 并填写，再自行加载。密钥文件不提交。

```bash
python infer_api_sdk_ocrbench_v2.py \
  --tsv data/ocrbench_v21.tsv \
  --model YOUR_MODEL \
  --base-url https://YOUR_API_HOST/v1 \
  --out-dir outputs/api_run \
  --xlsx-name api_run_ocrbench_v21.xlsx \
  --workers 8 --rpm 48
```

该驱动会写 JSONL 断点和 XLSX，默认对 7 个 QA 子类追加 `<answer>...</answer>` 格式提示。使用此提示时，评分侧需同步设置 `OCRBENCH_ANSWER_BOX_TYPES`（见下方示例）。

## 离线评分

对已有预测执行规则评分：

```bash
python score_ocrbench_v2_org.py \
  --pred outputs/api_run/api_run_ocrbench_v21.xlsx \
  --out-dir outputs/api_run --nproc 8
```

未提供 `--judge` 时，`text translation cn` 会被跳过，因此这不是包含全部 judge 子类的完整评分。源代码会在 CDM 依赖缺失时报错；以实际自检行为为准。

启用 judge 并与上述 API 驱动的默认答案格式对齐：

```bash
export OCRBENCH_ANSWER_BOX_TYPES='cognition VQA cn,cognition VQA en,reasoning VQA cn,reasoning VQA en,math QA en,diagram QA en,science QA en'
export OCRBENCH_QA_JUDGE=rescue
export OCRBENCH_JUDGE_RPM=55
# 预先在环境中设置 JUDGE_MODEL、JUDGE_BASE_URL、JUDGE_KEY
python score_ocrbench_v2_org.py \
  --pred outputs/api_run/api_run_ocrbench_v21.xlsx \
  --out-dir outputs/api_run --nproc 8 \
  --judge "$JUDGE_MODEL" \
  --judge-base-url "$JUDGE_BASE_URL" \
  --judge-key "$JUDGE_KEY"
```

输出为 `*_score.json`、`*_subcategory_score.json` 和 `*_scored.jsonl`。`OCRBENCH_QA_JUDGE_TYPES` 可限定 QA judge 子类；执行子集抽取、重评分和合并时须使用相同设置。具体计分策略及参数以导入代码为准。

## 运行路径

Shell 脚本按自身位置定位本评测目录，并切换到该目录执行。Python 的数据、配置和辅助路径按脚本位置解析；可用 `VLMEVAL_ROOT` 覆盖评测根目录。

| 配置 | 默认位置 / 用途 |
| --- | --- |
| `VLMEVAL_ROOT` | 自动定位本评测目录 |
| `PYTHON` / `VENV` | 覆盖 Python 解释器 / 虚拟环境，默认 `.venv` |
| `ORG_TSV` / `--tsv` | 数据文件，默认 `data/ocrbench_v21.tsv` |
| `LMUData` | 数据集索引目录，默认 `data/LMUData` |
| `config/*.json` 的 `model_path` | `models/` 下的模型目录；与对应 Shell 的 `MODEL_PATH` 保持一致 |
| `NLTK_DATA` | `.cache/nltk_data` |
| `CDM_TEXLIVE_ROOT` | `.tools/texlive/2025`，可指向已有 TeX Live |
| `VLMEVAL_LOCAL_PKGS` / `VLMEVAL_SHADOW_DIR` | 可选离线评分依赖及包导入目录，默认 `.cache/localpkgs` / `.cache/shadow` |

历史脚本中的模型名称、两阶段配置及 judge 参数保留。运行前准备对应模型，并按当前服务修改 judge 地址。TeX Live 的可选便携安装包放在 `tools_archive/`。这些本地资源目录已加入忽略规则。

历史脚本中清理掉的真实凭据改为读取 `JUDGE_KEY` 或 `ANTCHEMY_KEY` 环境变量。`dummy` 等原有本地服务占位值保留。

## 导入检查

- 724 个 Python 文件通过 Python 3.11 编译语法检查。
- 44 个 Shell 文件通过 `bash -n`。
- 82 个 JSON 文件（含来源清单）通过解析检查。
- 114 段内嵌 Python 通过语法检查；44 个 Shell 入口通过迁移目录定位检查。
- 已扫描待发布文件和提交历史，清理机器相关路径。
- 未运行模型推理或全量 OCRBench 评分；数据通过 Git LFS 下载，模型和外部 judge 服务需自行准备。

上游说明见 [`README.upstream.md`](README.upstream.md)，许可证见 [`LICENSE`](LICENSE)。

## 开源协议

本项目的开源代码采用 **GPL-2.0** 协议；**OCRBenchV2.1** 采用 **MIT** 协议。随附的 VLMEvalKit 框架保留其 [Apache-2.0 协议](LICENSE)。
