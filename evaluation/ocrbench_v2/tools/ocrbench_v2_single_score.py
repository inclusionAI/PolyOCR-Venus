#!/usr/bin/env python3
"""OCRBench_v2 单组全量评测(30 类一次跑完)后的统一算分。

用于「vLLM 部署 + 不分组、不开 thinking」的场景(如 HunyuanOCR, 仅一组 _all)。
与 tools/ocrbench_v2_merge_score.py 的口径完全一致, 只是把 parse_no_think + qa_think 两路收成一路 _all:

  - Step 7: 30 类子类小分表(取最新一份 jsonl, 按 (category,index) 去重, float(score) 求均值; 缺/0样本标 MISSING)。
  - Step 8: 官方聚合分(13 task type + English/Chinese Overall, sample-based), 口径对齐
            vlmeval/dataset/utils/ocrbrnch_v2_eval.py 的 ocrbench_v2_aggregate_accuracy。

用法(在 VLMEvalKit 工作目录下):
    python3 tools/ocrbench_v2_single_score.py <BASE_EVAL_NAME> --eval-name <BASE>_all \
        [--work-dir ./outputs/<BASE>] [--out-dir ./outputs/<BASE>]

不依赖 venv, 纯函数本地 python3 即可。
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
ALL_CATS = PARSE_CATS + QA_CATS


def latest_jsonl(work_dir, eval_dir):
    js = glob.glob(os.path.join(work_dir, eval_dir, '**/*_OCRBench_v2.jsonl'), recursive=True)
    if not js:
        return None
    return max(js, key=os.path.getmtime)


def score_from_jsonl(work_dir, eval_dir):
    """取最新一份 jsonl, 按 (category,index) 去重 -> (avg_dict, count_dict)。"""
    p = latest_jsonl(work_dir, eval_dir)
    if not p:
        return {}, {}, []
    cs = defaultdict(list); seen = set(); samples = []
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
        samples.append(d)
    return {c: sum(v) / len(v) for c, v in cs.items()}, {c: len(v) for c, v in cs.items()}, samples


def aggregate_accuracy(data_list):
    """与 ocrbrnch_v2_eval.py:ocrbench_v2_aggregate_accuracy 完全一致的纯函数。"""
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
    averages = {k: sum(v) / len(v) for k, v in buckets.items() if len(v) > 0}
    counts = {k: len(v) for k, v in buckets.items() if len(v) > 0}
    en = {k: averages[k] for k in averages if k.startswith("en_")}
    cn = {k: averages[k] for k in averages if k.startswith("cn_")}
    return en, cn, counts


def main():
    ap = argparse.ArgumentParser(description="OCRBench_v2 单组全量(30类)评测后的统一算分(Step7 30类合并 + Step8 官方聚合分)。")
    ap.add_argument("base", help="BASE_EVAL_NAME(如 eval_hunyuan_ocr_0722), 不含 _all 后缀")
    ap.add_argument("--eval-name", default=None, help="单组评测名, 默认 <base>_all")
    ap.add_argument("--work-dir", default="./outputs", help="产物查找根, 默认 ./outputs")
    ap.add_argument("--out-dir", default=None, help="合并/官方分 json 落盘目录, 默认 = --work-dir")
    args = ap.parse_args()

    base = args.base
    eval_name = args.eval_name or (base + "_all")
    work_dir = args.work_dir
    out_dir = args.out_dir if args.out_dir is not None else work_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"[single] base={base} eval_name={eval_name} work_dir={work_dir} out_dir={out_dir}")

    avg, count, samples = score_from_jsonl(work_dir, eval_name)

    # ---- Step 7: 30 类(缺失标 MISSING) ----
    missing = [c for c in ALL_CATS if avg.get(c) is None or count.get(c, -1) == 0]
    print(f"\n===== Step 7: 30-category subcategory scores (source: {eval_name}) =====\n")
    print(f'{"Category":<45} {"Score":>8} {"Count":>6}')
    print('-' * 65)
    present = 0
    for c in ALL_CATS:
        a = avg.get(c); n = count.get(c)
        if a is None or (n is not None and n == 0):
            print(f"{c:<45} {'MISSING':>8} {'-':>6}")
        else:
            present += 1
            print(f"{c:<45} {a*100:>7.2f}% {str(n):>6}")
    print('-' * 65)
    merged = {c: avg[c] for c in ALL_CATS if avg.get(c) is not None and count.get(c, -1) != 0}
    if merged:
        overall = sum(merged.values()) / len(merged)
        print(f'{"OVERALL (mean of present subcats)":<45} {overall*100:>7.2f}%   ({present}/30 present)')

    all_json = os.path.join(out_dir, base + "_all_subcategory_score.json")
    all_out = {
        "subcategory_avg_score": {c: round(avg[c], 6) for c in ALL_CATS if avg.get(c) is not None and count.get(c, -1) != 0},
        "subcategory_count": {c: count[c] for c in ALL_CATS if avg.get(c) is not None and count.get(c, -1) != 0},
        "overall_avg_score": round(sum(merged.values()) / len(merged), 6) if merged else 0.0,
        "eval_name": eval_name,
        "missing": missing,
    }
    json.dump(all_out, open(all_json, 'w'), indent=4, ensure_ascii=False)
    print(f"\nSaved 30-category merged -> {all_json}")
    if missing:
        print(f"\n!! {len(missing)} categories missing / 0 samples -> 建议带 --reuse 补跑后重算。")
        print(f"  缺失: {missing}")

    # ---- Step 8: 官方聚合分(单组 _all 已含 30 类全集, 直接合池) ----
    en, cn, ns = aggregate_accuracy(samples)
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
        print(f"\n注意: 有 {len(missing)} 个子类缺失, Step 7/8 数值不完整 —— 补跑后重算。")
    else:
        print("\nAll 30 categories present. Step 7/8 complete.")


if __name__ == "__main__":
    main()
