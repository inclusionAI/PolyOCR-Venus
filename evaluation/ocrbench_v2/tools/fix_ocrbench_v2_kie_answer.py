#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix OCRBench_v2 parsing-group prediction artifacts whose `answer` field got
mangled by pandas/Excel CSV quote-escaping, so that VLMEvalKit's
`process_predictions` hits `assert len(data_item["answers"]) == 1`
(ocrbrnch_v2_eval.py:274) and aborts scoring -> jsonl has no `score` column.

What happens: key information mapping en rows whose `answer` literal contains
nested double-quotes get an extra wrapping pair of quotes on write, so reading
back yields a long str whose len() != 1.

Strategy: rebuild the scoring artifact from authoritative sources, entirely
in memory, and write the final jsonl line-by-line via json.dumps (never csv/xlsx),
so no quote-escaping corruption can recur.

  - answer / bbox / content / eval / question / category / image  : from original OCRBench_v2.tsv (authoritative)
  - prediction                                                   : from existing run's jsonl (T20260719-214350)
  - score                                                        : computed by process_predictions (with gemini judge for formula/translation)

Output: <EVAL_DIR>/eval_full_sft_0715_ckpt6998_0719_parse_no_think_OCRBench_v2_fixed.jsonl
        + a copy of the per-task-type official aggregate score json.
"""
from pathlib import Path as _EvalPath
import os as _eval_os
_EVAL_ROOT = _EvalPath(_eval_os.environ.get("VLMEVAL_ROOT", str(_EvalPath(__file__).resolve().parents[1]))).resolve()

import ast
import json
import os

# Must be set before any vlmeval/nltk metric import that resolves corpus paths.
os.environ.setdefault('NLTK_DATA', str(_EVAL_ROOT / '.cache/nltk_data'))
os.environ.setdefault('LMUData', str(_EVAL_ROOT / 'data/LMUData'))

import pandas as pd

# ----- config -----
EVAL_NAME = "eval_full_sft_0715_ckpt6998_0719_parse_no_think"
BASE = "./outputs/eval_full_sft_0715_ckpt6998_0719"
SRC_TSV = str(_EVAL_ROOT / 'data/LMUData/OCRBench_v2.tsv')
SRC_RUN = "T20260720-113923"
PARSED_JSONL = os.path.join(BASE, EVAL_NAME, SRC_RUN, EVAL_NAME + "_OCRBench_v2.jsonl")
OUT_JSONL = os.path.join(BASE, EVAL_NAME, EVAL_NAME + "_OCRBench_v2_fixed.jsonl")
EVAL_TYPES = [
    'chart parsing en', 'document classification en', 'document parsing cn', 'document parsing en',
    'fine-grained text recognition en', 'formula recognition cn', 'formula recognition en',
    'full-page OCR cn', 'full-page OCR en', 'handwritten answer extraction cn',
    'key information extraction cn', 'key information extraction en', 'key information mapping en',
    'table parsing cn', 'table parsing en', 'text grounding en', 'text recognition en',
    'text spotting en', 'text translation cn',
]

EVAL_TYPES_SET = set(EVAL_TYPES)


def main():
    print(f"[1/4] loading source TSV: {SRC_TSV}")
    src = pd.read_csv(SRC_TSV, sep='\t').set_index('index')
    src_f = src[src['category'].isin(EVAL_TYPES_SET)].sort_index()
    print(f"      source rows (filtered to {len(EVAL_TYPES)} types): {len(src_f)}")

    print(f"[2/4] loading existing predictions: {PARSED_JSONL}")
    preds = {}
    with open(PARSED_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            preds[d['index']] = d
    print(f"      prediction rows: {len(preds)}")

    missing = [i for i in src_f.index.tolist() if i not in preds]
    assert not missing, f"{len(missing)} src indices missing predictions: {missing[:10]}"
    cat_mismatch = [i for i in src_f.index.tolist() if preds[i]['category'] != src_f.loc[i, 'category']]
    assert not cat_mismatch, f"{len(cat_mismatch)} category mismatches"
    print(f"      alignment OK (no missing, no category mismatch).")

    # Build predict_result exactly like image_vqa.OCRBench_v2.evaluate does.
    print(f"[3/4] building predict_result and scoring via process_predictions ...")
    predict_result = []
    for idx in src_f.index.tolist():
        row = src_f.loc[idx]
        pred = preds[idx]
        predict = str(pred['prediction']) if pred.get('prediction') not in (None, '') else ''
        answers = ast.literal_eval(row['answer'])
        questions = row['question']
        evals = row['eval']
        bbox_raw = row['bbox']
        content_raw = row['content']
        bbox = ast.literal_eval(bbox_raw) if bbox_raw != 'without bbox' else bbox_raw
        content = ast.literal_eval(content_raw) if content_raw != 'without content' else content_raw
        entry = {
            "index": int(idx),
            "image_path": pred.get('image_path', row.get('image', '')),
            "type": row['category'],
            "category": row['category'],
            "question": questions,
            "predict": predict,
            "answers": answers,
            "bbox": bbox,
            "content": content,
        }
        if pd.notna(evals) and evals != 'without eval':
            entry["eval"] = evals
        predict_result.append(entry)

    # Score
    from vlmeval.dataset.utils.ocrbrnch_v2_eval import (
        ocrbench_v2_aggregate_accuracy, process_predictions,
    )
    from vlmeval.dataset.utils import build_judge

    judge_kwargs = dict(
        model='gemini-3.5-flash',
        nproc=4,
        retry=8,
        timeout=600,
        verbose=True,
    )
    judge_model = build_judge(max_tokens=128, **judge_kwargs)
    print(f"      judge_model built: {type(judge_model).__name__}")

    res_data_list = process_predictions(predict_result, judge_model=judge_model, nproc=4)

    # Validate every row got a score
    no_score = [d['index'] for d in res_data_list if d.get('score') is None]
    if no_score:
        print(f"      WARNING: {len(no_score)} rows have score=None (formula/translation judge pending?)")
    print(f"      scored {len(res_data_list)} rows.")

    # Write clean jsonl line-by-line (json.dumps -> no csv quote-escaping issues)
    print(f"[4/4] writing clean fixed jsonl: {OUT_JSONL}")
    with open(OUT_JSONL, 'w') as f:
        for d in res_data_list:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    print(f"      wrote {len(res_data_list)} lines.")

    # Official aggregate score (13 task types + En/Cn overall), sample-based
    en_scores, cn_scores = ocrbench_v2_aggregate_accuracy(res_data_list)
    final = {**en_scores, **cn_scores}
    if en_scores:
        final["English Overall Score"] = sum(en_scores.values()) / len(en_scores)
    if cn_scores:
        final["Chinese Overall Score"] = sum(cn_scores.values()) / len(cn_scores)
    score_pth = OUT_JSONL.replace('.jsonl', '_score.json')
    with open(score_pth, 'w') as f:
        json.dump({k: round(v, 6) for k, v in final.items()}, f, indent=4, ensure_ascii=False)

    order = [
        "en_text_recognition", "en_text_detection", "en_text_spotting",
        "en_relationship_extraction", "en_element_parsing", "en_mathematical_calculation",
        "en_visual_text_understanding", "en_knowledge_reasoning",
        "cn_text_recognition", "cn_relationship_extraction", "cn_element_parsing",
        "cn_visual_text_understanding", "cn_knowledge_reasoning",
    ]
    print(f"\n===== Official aggregate score (sample-based, 13 task types) =====")
    print(f"{'Task Type':<38} {'Score':>8}")
    print('-' * 50)
    for tt in order:
        if tt in final:
            print(f"{tt:<38} {final[tt]*100:>7.2f}%")
    print('-' * 50)
    if "English Overall Score" in final:
        print(f"{'English Overall Score':<38} {final['English Overall Score']*100:>7.2f}%")
    if "Chinese Overall Score" in final:
        print(f"{'Chinese Overall Score':<38} {final['Chinese Overall Score']*100:>7.2f}%")
    print(f"\nSaved score json: {score_pth}")
    print("DONE.")


if __name__ == "__main__":
    main()
