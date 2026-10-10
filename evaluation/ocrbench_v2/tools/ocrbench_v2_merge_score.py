#!/usr/bin/env python3
"""OCRBench_v2 两组串行评测完成后的统一算分：
  - Step 7: 合并 30 类子类小分表（19 类来自 parse_no_think，11 类来自 qa_think），缺样本/缺失子类标 MISSING。
  - Step 8: 官方聚合分（13 个 task type + English/Chinese Overall），口径对齐
            vlmeval/dataset/utils/ocrbrnch_v2_eval.py 的 ocrbench_v2_aggregate_accuracy（sample-based）。

通用脚本（跨 checkpoint 复用同一份，不要为每个 checkpoint 重新创建算分脚本）：
  在 VLMEvalKit 工作目录下执行：
    `python3 tools/ocrbench_v2_merge_score.py <BASE_EVAL_NAME> [--work-dir DIR] [--out-dir DIR]`
  --work-dir：两组产物的查找根（默认 ./outputs）；会在 <work-dir>/<eval_name>/ 下读组内小分 json 与 jsonl。
  --out-dir ：合并/官方分 json 落盘目录（默认 = --work-dir）。新布局两组收拢到 outputs/<BASE>/ 时，
              传 --work-dir ./outputs/<BASE> --out-dir ./outputs/<BASE>，两侧同指共享文件夹。
  不传任何可选参数时：work-dir=out-dir=./outputs，等价于旧布局（产物散落在 outputs/ 根与各组目录）。
  不依赖 venv，纯函数本地 python3 即可。
"""
import argparse, glob, json, os
from collections import defaultdict

PARSE_CATS = [
    "chart parsing en", "document classification en", "document parsing cn", "document parsing en",
    "fine-grained text recognition en", "formula recognition cn", "formula recognition en",
    "full-page OCR cn", "full-page OCR en", "handwritten answer extraction cn",
    "key information extraction cn", "key information extraction en", "key information mapping en",
    "table parsing cn", "table parsing en", "text grounding en", "text recognition en",
    "text spotting en", "text translation cn",
]
QA_CATS = [
    "ASCII art classification en", "APP agent en", "VQA with position en", "cognition VQA cn",
    "cognition VQA en", "diagram QA en", "math QA en", "reasoning VQA cn", "reasoning VQA en",
    "science QA en", "text counting en",
]


def load_group(work_dir, eval_name):
    """读组内 _subcategory_score.json，兼容新旧两种格式。返回 (avg_dict, count_dict)。"""
    p = os.path.join(work_dir, eval_name, os.path.basename(eval_name) + '_subcategory_score.json')
    if not os.path.exists(p):
        return {}, {}
    d = json.load(open(p))
    if isinstance(d, dict) and 'subcategory_avg_score' in d:
        return d.get('subcategory_avg_score', {}), d.get('subcategory_count', {})
    return d, {}  # 旧格式: 顶层 cat->score


def latest_jsonl(work_dir, eval_dir):
    js = glob.glob(os.path.join(work_dir, eval_dir, '**/*_OCRBench_v2.jsonl'), recursive=True)
    if not js:
        return None
    return max(js, key=os.path.getmtime)


def score_from_jsonl(work_dir, eval_dir):
    """从最新 jsonl 实时算该组(取最新一份, 按 (category,index) 去重), 兜底用。"""
    p = latest_jsonl(work_dir, eval_dir)
    if not p:
        return {}, {}
    cs = defaultdict(list); seen = set()
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        k = (d['category'], d.get('index'))
        if k in seen:
            continue
        seen.add(k)
        cs[d['category']].append(float(d['score']))
    return {c: sum(v)/len(v) for c, v in cs.items()}, {c: len(v) for c, v in cs.items()}


def load_dedup_samples(work_dir, eval_dir):
    """去重后返回该组最新 jsonl 的全部样本 list，供 Step 8 合池用。"""
    p = latest_jsonl(work_dir, eval_dir)
    if not p:
        return []
    seen = set(); out = []
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        k = (d['category'], d.get('index'))
        if k in seen:
            continue
        seen.add(k)
        out.append(d)
    return out


# ---- Step 8: 官方聚合口径(与 ocrbrnch_v2_eval.py 对齐) ----
def aggregate_accuracy(data_list):
    buckets = {
        "en_text_recognition": [], "en_text_detection": [], "en_text_spotting": [],
        "en_relationship_extraction": [], "en_element_parsing": [], "en_mathematical_calculation": [],
        "en_visual_text_understanding": [], "en_knowledge_reasoning": [],
        "cn_text_recognition": [], "cn_relationship_extraction": [], "cn_element_parsing": [],
        "cn_visual_text_understanding": [], "cn_knowledge_reasoning": [],
    }
    for item in data_list:
        if str(item.get("ignore")) == "True":
            continue
        t = item.get("category") or item.get("type")
        if t is None:
            continue
        s = float(item["score"])
        if   t in ("text recognition en", "fine-grained text recognition en", "full-page OCR en"): buckets["en_text_recognition"].append(s)
        elif t in ("text grounding en", "VQA with position en"): buckets["en_text_detection"].append(s)
        elif t == "text spotting en": buckets["en_text_spotting"].append(s)
        elif t in ("key information extraction en", "key information mapping en"): buckets["en_relationship_extraction"].append(s)
        elif t in ("document parsing en", "chart parsing en", "table parsing en", "formula recognition en"): buckets["en_element_parsing"].append(s)
        elif t in ("math QA en", "text counting en"): buckets["en_mathematical_calculation"].append(s)
        elif t in ("document classification en", "cognition VQA en", "diagram QA en"): buckets["en_visual_text_understanding"].append(s)
        elif t in ("reasoning VQA en", "science QA en", "APP agent en", "ASCII art classification en"): buckets["en_knowledge_reasoning"].append(s)
        elif t == "full-page OCR cn": buckets["cn_text_recognition"].append(s)
        elif t in ("key information extraction cn", "handwritten answer extraction cn"): buckets["cn_relationship_extraction"].append(s)
        elif t in ("document parsing cn", "table parsing cn", "formula recognition cn"): buckets["cn_element_parsing"].append(s)
        elif t == "cognition VQA cn": buckets["cn_visual_text_understanding"].append(s)
        elif t in ("reasoning VQA cn", "text translation cn"): buckets["cn_knowledge_reasoning"].append(s)
    averages = {k: sum(v)/len(v) for k, v in buckets.items() if len(v) > 0}
    counts = {k: len(v) for k, v in buckets.items() if len(v) > 0}
    en = {k: averages[k] for k in averages if k.startswith("en_")}
    cn = {k: averages[k] for k in averages if k.startswith("cn_")}
    return en, cn, counts


def main():
    ap = argparse.ArgumentParser(description="OCRBench_v2 两组串行评测后的统一算分(Step7 30类合并 + Step8 官方聚合分)。")
    ap.add_argument("base", help="BASE_EVAL_NAME（如 eval_full_sft_0709_ckpt5382），不包含 _parse_no_think/_qa_think 后缀")
    ap.add_argument("--work-dir", default="./outputs", help="两组产物的查找根，默认 ./outputs（旧布局）。新布局传 ./outputs/<BASE>")
    ap.add_argument("--out-dir", default=None, help="合并/官方分 json 落盘目录，默认 = --work-dir")
    args = ap.parse_args()

    base = args.base
    work_dir = args.work_dir
    out_dir = args.out_dir if args.out_dir is not None else work_dir
    os.makedirs(out_dir, exist_ok=True)
    parse_eval = base + "_parse_no_think"
    qa_eval = base + "_qa_think"
    print(f"[merge] base={base} work_dir={work_dir} out_dir={out_dir}")

    # ---- Step 7: 合并 30 类 ----
    PARSE_AVG, PARSE_COUNT = load_group(work_dir, parse_eval)
    QA_AVG, QA_COUNT = load_group(work_dir, qa_eval)
    if not PARSE_AVG:
        PARSE_AVG, PARSE_COUNT = score_from_jsonl(work_dir, base + "_parse_no_think")
    if not QA_AVG:
        QA_AVG, QA_COUNT = score_from_jsonl(work_dir, base + "_qa_think")

    merged_avg, merged_count, source = {}, {}, {}
    missing = []
    for c in PARSE_CATS:
        if PARSE_AVG.get(c) is not None and PARSE_COUNT.get(c, -1) != 0:
            merged_avg[c] = PARSE_AVG[c]; merged_count[c] = PARSE_COUNT.get(c); source[c] = "parse_no_think"
        else:
            missing.append(c)
    for c in QA_CATS:
        if QA_AVG.get(c) is not None and QA_COUNT.get(c, -1) != 0:
            merged_avg[c] = QA_AVG[c]; merged_count[c] = QA_COUNT.get(c); source[c] = "qa_think"
        else:
            missing.append(c)

    print(f"\n===== Step 7: 30-category merged subcategory scores =====\n")
    print(f'{"Category":<45} {"Score":>8} {"Count":>6} {"Source":>16}')
    print('-' * 80)
    present = 0
    for c in PARSE_CATS + QA_CATS:
        a = merged_avg.get(c); n = merged_count.get(c)
        if a is None:
            print(f"{c:<45} {'MISSING':>8} {'-':>6}")
        else:
            present += 1
            print(f"{c:<45} {a*100:>7.2f}% {str(n):>6} {source[c]:>16}")
    print('-' * 80)
    if present > 0:
        overall = sum(merged_avg.values()) / len(merged_avg)
        print(f'{"OVERALL (mean of present subcats)":<45} {overall*100:>7.2f}%   ({present}/30 present)')

    # 写 30 类合并表(即使有缺失也写, 缺的不进 dict)
    all_json = os.path.join(out_dir, base + "_all_subcategory_score.json")
    all_out = {
        "subcategory_avg_score": {c: round(merged_avg[c], 6) for c in (PARSE_CATS + QA_CATS) if c in merged_avg},
        "subcategory_count": {c: merged_count[c] for c in (PARSE_CATS + QA_CATS) if c in merged_count},
        "overall_avg_score": round(sum(merged_avg.values()) / len(merged_avg), 6) if merged_avg else 0.0,
        "source": source,
        "parse_eval_name": parse_eval,
        "qa_eval_name": qa_eval,
        "missing": missing,
    }
    json.dump(all_out, open(all_json, 'w'), indent=4, ensure_ascii=False)
    print(f"\nSaved 30-category merged -> {all_json}")

    if missing:
        print(f"\n!! {len(missing)} categories missing / 0 samples -> 需要补跑对应分组 (with --reuse):")
        parse_missing = [c for c in missing if c in PARSE_CATS]
        qa_missing = [c for c in missing if c in QA_CATS]
        if parse_missing:
            print(f"  重跑 parse_no_think 脚本补: {parse_missing}")
        if qa_missing:
            print(f"  重跑 qa_think 脚本补: {qa_missing}")
        print("  补跑后重跑本算分脚本即可。")

    # ---- Step 8: 官方聚合分(只在 30 类齐全时才算完整; 有缺失也会算但会提示) ----
    # 合池: parse 组取19类, qa 组取11类, 各自归属取数, 按 (category,index) 去重
    parse_set = set(PARSE_CATS); qa_set = set(QA_CATS)
    parse_samples = [d for d in load_dedup_samples(work_dir, base + "_parse_no_think") if d['category'] in parse_set]
    qa_samples = [d for d in load_dedup_samples(work_dir, base + "_qa_think") if d['category'] in qa_set]
    seen = set(); merged_samples = []
    for d in parse_samples + qa_samples:
        cat = d['category']
        src = "parse" if cat in parse_set else "qa"
        k = (cat, d.get('index'), src)
        if k in seen:
            continue
        seen.add(k)
        merged_samples.append(d)

    en, cn, ns = aggregate_accuracy(merged_samples)
    en_overall = sum(en.values()) / len(en) if en else 0
    cn_overall = sum(cn.values()) / len(cn) if cn else 0
    final = {**en, **cn, "English Overall Score": en_overall, "Chinese Overall Score": cn_overall}

    order = ["en_text_recognition", "en_text_detection", "en_text_spotting", "en_relationship_extraction",
             "en_element_parsing", "en_mathematical_calculation", "en_visual_text_understanding", "en_knowledge_reasoning",
             "cn_text_recognition", "cn_relationship_extraction", "cn_element_parsing", "cn_visual_text_understanding",
             "cn_knowledge_reasoning"]
    print(f"\n===== Step 8: Official aggregate (task type, sample-based) =====\n")
    print(f'{"Task Type":<38} {"Score":>8}  {"N samples":>10}')
    print('-' * 62)
    for tt in order:
        if tt in final:
            print(f"{tt:<38} {final[tt]*100:>7.2f}% {ns.get(tt, 0):>10}")
    print('-' * 62)
    print(f"{'English Overall Score':<38} {en_overall*100:>7.2f}%")
    print(f"{'Chinese Overall Score':<38} {cn_overall*100:>7.2f}%")

    off_json = os.path.join(out_dir, base + "_official_aggregate_score.json")
    json.dump({k: round(v, 6) for k, v in final.items()}, open(off_json, 'w'), indent=4, ensure_ascii=False)
    print(f"\nSaved official aggregate -> {off_json}")

    if missing:
        print(f"\n注意: 有 {len(missing)} 个子类缺失, 上述 Step 7/8 数值不完整 —— 补跑对应分组后重算。")
    else:
        print("\nAll 30 categories present. Step 7/8 complete.")


if __name__ == "__main__":
    main()