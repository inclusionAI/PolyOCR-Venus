# flake8: noqa
import argparse
import ast
import json
import os
import re

import ipdb
import numpy as np
from tqdm import tqdm

from .Ocrbench_v2.IoUscore_metric import (calculate_iou, extract_coordinates,
                                          vqa_with_position_evaluation)
from .Ocrbench_v2.page_ocr_metric import cal_per_metrics, ocr_edit_score
from .Ocrbench_v2.spotting_metric import extract_bounding_boxes_robust, spotting_evaluation
from .Ocrbench_v2.TEDS_metric import (TEDS, compute_f1_score, convert_markdown_table_to_html,
                                      convert_str_to_dict, convert_str_to_multi_dict, dict_to_html,
                                      doc_parsing_edit_distance, doc_parsing_evaluation,
                                      generate_combinations, wrap_html_table)
from .Ocrbench_v2.doc_parsing_md2md import doc_parsing_md2md_score
from .Ocrbench_v2.vqa_metric import (cn_formula_vlm_judge_evaluation, cn_math_expression_evaluation,
                                     cn_vqa_evaluation, counting_evaluation,
                                     formula_cdm_score, formula_vlm_judge_evaluation,
                                     math_expression_evaluation,
                                     text_translation_vlm_judge_evaluation, vqa_evaluation,
                                     vqa_evaluation_case_sensitive)


def is_nan_value(value):
    if value is None:
        return True
    if isinstance(value, str) and value.lower() == 'nan':
        return True
    try:
        import pandas as pd
        if pd.isna(value):
            return True
    except:
        pass
    return False


def get_value_or_zero(value):
    return 0.0 if value is None else value


def calculate_average(scores_dict):
    averages = {key: sum(values) / len(values) for key, values in scores_dict.items() if len(values) > 0}
    return averages


def _formula_judge_worker(judge_model, predict, answers, lang):
    from .Ocrbench_v2.vqa_metric import (cn_formula_vlm_judge_evaluation,
                                         formula_vlm_judge_evaluation)
    if lang == 'en':
        return formula_vlm_judge_evaluation(judge_model, predict, answers)
    else:
        return cn_formula_vlm_judge_evaluation(judge_model, predict, answers)


def _text_translation_judge_worker(judge_model, predict, answers):
    from .Ocrbench_v2.vqa_metric import text_translation_vlm_judge_evaluation
    return text_translation_vlm_judge_evaluation(judge_model, predict, answers)


# QA 类自由问答 (与 ocrbench_clean/convert_ocrbench_v2_qa_to_grpo.py 的 QA_CLASS_PATTERN 口径一致,
# 排除 VQA with position en / text counting en; case sensitive 走确定性规则不进 judge;
# multiple choice 规则保底, 未满分时同样进 judge rescue 复核)。
# 全量候选: 只有这里的子类**可能**进 judge, OCRBENCH_QA_JUDGE_TYPES 再做白名单收窄。
_QA_JUDGE_TYPES = frozenset({
    "APP agent en", "cognition VQA en", "diagram QA en", "math QA en",
    "reasoning VQA en", "science QA en", "cognition VQA cn", "reasoning VQA cn",
})
# LLM-judge 的翻译子类 (单独口径, 不受 QA 白名单影响)
QA_TEXT_TRANSLATION = "text translation cn"


def _qa_judge_mode():
    """QA judge 默认开启 (有 judge_model 且未全局跳过时)。

    环境变量 OCRBENCH_QA_JUDGE:
      - rescue (默认): 规则分保底, 仅复核规则未满分的样本, 最终分 = max(规则, judge);
      - full:    QA 类一律由 judge 打分, 最终分 = judge;
      - off/0/false: 关闭, 退化为纯确定性规则。
    全局 OCRBENCH_SKIP_LLM_JUDGE 非空时与翻译子类一并跳过 (优先级最高)。
    """
    if os.environ.get('OCRBENCH_SKIP_LLM_JUDGE'):
        return 'off'
    raw = (os.environ.get('OCRBENCH_QA_JUDGE') or 'rescue').strip().lower()
    if raw in ('off', '0', 'false', 'no'):
        return 'off'
    if raw == 'full':
        return 'full'
    return 'rescue'


def _qa_judge_worker(judge_model, question, predict, answers):
    from .Ocrbench_v2.vqa_metric import qa_vlm_judge_evaluation
    return qa_vlm_judge_evaluation(judge_model, question, predict, answers)


def qa_judge_active_types():
    """本次运行**实际**会送 LLM judge 的 QA 子类集合。

    环境变量 OCRBENCH_QA_JUDGE_TYPES (逗号分隔, 每项 strip):
      - 未设置/为空串 -> 全部 8 个 QA 子类 (历史默认口径, 行为不变);
      - 显式列出      -> 仅列出的子类进 judge, 未列出的 QA 子类退化为纯确定性规则打分
                        (即 vqa_evaluation / cn_vqa_evaluation / 多选抠字母, 不复核)。

    只接受 _QA_JUDGE_TYPES 内的名字; 其余 (如 VQA with position en / text counting en)
    从来没有 judge 口径, 列出会告警并忽略。
    subset 抽取 (build_qa_subset.py) 与重测合并 (merge_qa_rescore.py /
    finalize_qa_rescore.py) 用同一函数对齐 AFFECTED, 避免打分口径与合并口径漂移。
    """
    raw = (os.environ.get('OCRBENCH_QA_JUDGE_TYPES') or '').strip()
    if not raw:
        return set(_QA_JUDGE_TYPES)
    want = {t.strip() for t in raw.split(',') if t.strip()}
    unknown = want - _QA_JUDGE_TYPES
    if unknown:
        print(f"[QA VLM Judge][WARN] 以下子类无 judge 口径, 已忽略: {sorted(unknown)}")
    return want & _QA_JUDGE_TYPES


def qa_rescore_affected_types():
    """受 LLM-judge 口径影响的子类 = QA 白名单(生效时) + 翻译子类(未全局跳过时)。

    给 subset 抽取 / merge 条数校验 / finalize 汇总用: 这些环节必须与实际送 judge
    的子类完全一致, 否则 merge 的"逐类条数一致"校验会 FATAL。
    """
    if os.environ.get('OCRBENCH_SKIP_LLM_JUDGE'):
        return set()                                    # 全局跳过: 无任何 judge 口径变化
    if _qa_judge_mode() == 'off':
        return {QA_TEXT_TRANSLATION}                     # 仅翻译子类受影响
    return set(qa_judge_active_types()) | {QA_TEXT_TRANSLATION}


def process_predictions(predict_file, judge_model=None, nproc=16):
    # track_progress_rich 仅在 (空的) LLM-judge 任务池里用到, 延迟导入以避免拉起
    # vlmeval.utils -> vlmeval.dataset.__init__ 的整包导入链 (NFS 上很重)。
    track_progress_rich = None

    # TEDS 内部对单个表格即 parallel_process(n_jobs=...) 起 32 worker 池, 每条 fork 32 进程
    # 的开销远大于并行收益 -> 默认 n_jobs=1 (顺序, 无 fork)。需要时可经环境变量调高。
    teds = TEDS(n_jobs=int(os.environ.get('OCRBENCH_TEDS_NJOBS', '1')))

    res_data_list = []

    formula_indices = []
    formula_tasks = []
    text_translation_indices = []
    text_translation_tasks = []

    # QA 类 judge (默认 rescue): 规则分保底, 规则未满分才交 judge 复核, 最终 max(规则, judge)。
    # 生效子类由 OCRBENCH_QA_JUDGE_TYPES 收窄, 白名单外的 QA 子类只留下面这行确定性规则分。
    qa_judge_mode = _qa_judge_mode() if judge_model is not None else 'off'
    qa_judge_types = qa_judge_active_types() if qa_judge_mode != 'off' else set()
    if qa_judge_mode != 'off':
        print(f"[QA VLM Judge] mode={qa_judge_mode}, 生效子类 {len(qa_judge_types)} 个: "
              f"{sorted(qa_judge_types)}")
        rule_only = sorted(_QA_JUDGE_TYPES - qa_judge_types)
        if rule_only:
            print(f"[QA VLM Judge] 以下 QA 子类本次纯规则打分 (不进 judge): {rule_only}")
    qa_indices = []
    qa_tasks = []
    qa_rule_scores = []

    def _qa_rule_then_judge(index, data_item, rule_score):
        """先落确定性规则分, 再按 mode + 子类白名单决定是否送 judge 复核。

        所有 QA 分支统一走这里: 子类是否在白名单内由本函数判, 调用点不再各自 if。
        """
        data_item["score"] = rule_score
        if qa_judge_mode == 'off':
            return
        if data_item["type"] not in qa_judge_types:
            return  # 该子类本次不启用 judge, 规则分即最终分
        if qa_judge_mode == 'rescue' and rule_score >= 1:
            return
        predict = data_item["predict"]
        if not isinstance(predict, str) or not predict.strip():
            return  # 空预测 judge 必判 0, 省一次调用
        data_item["score"] = None  # 待 judge 池跑完后回填
        qa_indices.append(index)
        qa_rule_scores.append(rule_score)
        qa_tasks.append(dict(
            judge_model=judge_model,
            question=data_item.get("question"),
            predict=predict,
            answers=data_item["answers"],
        ))

    for index, data_item in enumerate(tqdm(predict_file)):
        if data_item["type"] == "APP agent en" or data_item["type"] == "ASCII art classification en" or data_item["type"] == "math QA en" \
            or data_item["type"] == "reasoning VQA en" or data_item["type"] == "science QA en" \
            or data_item["type"] == "text recognition en" or data_item["type"] == "document classification en" \
            or data_item["type"] == "cognition VQA en" or data_item["type"] == "diagram QA en":
            if "eval" in data_item.keys():
                if data_item["eval"] == "multiple choice":
                    if not isinstance(data_item["answers"], list):
                        data_item["answers"] = [data_item["answers"]]
                    assert len(data_item["answers"]) == 1

                    if not isinstance(data_item["predict"], str):
                        data_item["score"] = 0
                    else:
                        predict = ''.join(c for c in data_item["predict"] if c.isalpha())

                        if predict == data_item["answers"][0]:
                            data_item["score"] = 1
                        else:
                            data_item["score"] = 0
                    # 规则只抠字母整串相等, 对带思维链/<answer>标签的输出必判 0:
                    # 规则未满分时走 QA judge rescue 复核 (与自由问答同一 prompt 池, 最终 max(规则, judge))。
                    # 是否在白名单内由 _qa_rule_then_judge 内部判定。
                    _qa_rule_then_judge(index, data_item, data_item["score"])
                elif data_item["eval"] == "case sensitive":
                    data_item["score"] = vqa_evaluation_case_sensitive(data_item["predict"], data_item["answers"])
                else:
                    raise ValueError("No such evaluation method")
            else:
                # QA 子类: 规则 + (白名单内则) judge 复核; text recognition en /
                # document classification en / ASCII art classification en 等非 QA 类: 纯规则。
                _qa_rule_then_judge(index, data_item, vqa_evaluation(data_item["predict"], data_item["answers"]))

        elif data_item["type"] == "cognition VQA cn" or data_item["type"] == "reasoning VQA cn":
            if "eval" in data_item.keys():
                if data_item["eval"] == "multiple choice":
                    # 个别多答案多选(如 ['A','B']) 会撞原 len==1 硬断言; 改为:
                    # 预测(仅留字母)命中任一正确选项即记对 —— 单答案项与原逻辑一致。
                    ans = data_item["answers"] if isinstance(data_item["answers"], list) else []
                    if not isinstance(data_item["predict"], str) or len(ans) == 0:
                        data_item["score"] = 0
                    else:
                        predict = ''.join(c for c in data_item["predict"] if c.isalpha())
                        data_item["score"] = 1 if predict in ans else 0
                    # 同 en: 抠字母规则对思维链输出误杀, 未满分交 judge rescue 复核。
                    _qa_rule_then_judge(index, data_item, data_item["score"])
                elif data_item["eval"] == "case sensitive":
                    data_item["score"] = vqa_evaluation_case_sensitive(data_item["predict"], data_item["answers"])
                else:
                    raise ValueError("No such evaluation method")
            else:
                # cognition VQA cn / reasoning VQA cn 均属 QA 类: 规则 + 默认 judge 复核
                _qa_rule_then_judge(index, data_item, cn_vqa_evaluation(data_item["predict"], data_item["answers"]))

        elif data_item["type"] == "handwritten answer extraction cn":
            if "简答" in data_item["question"]:
                ocr_metric = cal_per_metrics(data_item["predict"], data_item["answers"][0])
                data_item["score"] = (
                    get_value_or_zero(ocr_metric["bleu"]) +
                    get_value_or_zero(ocr_metric["meteor"]) +
                    get_value_or_zero(ocr_metric["f_measure"]) +
                    (1 - get_value_or_zero(ocr_metric["edit_dist"]))
                ) / 4
            else:
                assert len(data_item["answers"]) == 1
                answer = data_item["answers"][0]
                chars = list(answer)
                if len(answer) > 1:

                    answer_list = [
                        "".join(chars),
                        ".".join(chars),
                        ". ".join(chars),
                        ",".join(chars),
                        ", ".join(chars),
                        "、".join(chars),
                        ";".join(chars),
                        "; ".join(chars),
                        " ".join(chars),
                        "和".join(chars)
                    ]
                    max_score = 0
                    for answer in answer_list:
                        if answer in data_item["predict"]:
                            temp_score = 1
                        else:
                            temp_score = 0
                        if temp_score > max_score:
                            max_score = temp_score
                    data_item["score"] = max_score

                else:
                    if data_item["answers"][0] in data_item["predict"]:
                        data_item["score"] = 1
                    else:
                        data_item["score"] = 0

        elif data_item["type"] == "formula recognition cn":
            if is_nan_value(data_item["predict"]):
                data_item["score"] = 0
            else:
                # 改用 OmniDocBench CDM F1 评测（渲染成图后 token 匹配），去掉 LLM judge
                data_item["score"] = formula_cdm_score(data_item["predict"], data_item["answers"])

        elif data_item["type"] == "text counting en":
            data_item["score"] = counting_evaluation(data_item["predict"], data_item["answers"], data_item["eval"])

        elif data_item["type"] == "formula recognition en":
            # 改用 OmniDocBench CDM F1 评测（渲染成图后 token 匹配），去掉 LLM judge
            data_item["score"] = formula_cdm_score(data_item["predict"], data_item["answers"])

        elif data_item["type"] == "table parsing en":
            if type(data_item["answers"])==list and len(data_item["answers"]) == 1:
                if not isinstance(data_item["predict"], str):
                    data_item["score"] = 0
                elif not isinstance(data_item["question"], str):
                    data_item["ignore"] = "True"
                    data_item["score"] = 0

                elif "html" in data_item["question"].lower():
                    no_find = False
                    predict_table = data_item["predict"].replace('\n','')
                    if "<body" in predict_table:
                        predict_table = re.findall('<body.*', predict_table)[0]
                    elif "<table" in predict_table:
                        predict_table = re.findall('<table.*', predict_table)[0]
                    else:
                        no_find = True

                    if no_find:
                        data_item["score"] = 0
                    else:
                        pred_table_html = wrap_html_table(predict_table)
                        gold_table_html = wrap_html_table(data_item["answers"][0])
                        try:
                            data_item["score"] = teds.evaluate(pred_table_html, gold_table_html)
                        except:
                            data_item["score"] = 0

                elif "markdown" in data_item["question"].lower():
                    if not isinstance(data_item["predict"], str):

                        prediction = str(data_item["predict"])
                        pred_table_html = convert_markdown_table_to_html(prediction)
                        gt_table_html = convert_markdown_table_to_html(data_item["answers"][0])
                        data_item["score"] = teds.evaluate(pred_table_html, gt_table_html)

                    else:
                        pred_table_html = convert_markdown_table_to_html(data_item["predict"])
                        gt_table_html = convert_markdown_table_to_html(data_item["answers"][0])
                        data_item["score"] = teds.evaluate(pred_table_html, gt_table_html)
            else:
                raise ValueError

        elif data_item["type"] == "table parsing cn":
            if not isinstance(data_item["predict"], str):
                data_item["score"] = 0
            else:
                no_find = False
                predict_table = data_item["predict"].replace('\n','')
                if "<body" in predict_table:
                    predict_table = re.findall('<body.*', predict_table)[0]
                elif "<table" in predict_table:
                    predict_table = re.findall('<table.*', predict_table)[0]
                else:
                    no_find = True

                if no_find:
                    data_item["score"] = 0
                else:
                    pred_table_html = wrap_html_table(predict_table)
                    gold_table_html = wrap_html_table(data_item["answers"][0])
                    try:
                        data_item["score"] = teds.evaluate(pred_table_html, gold_table_html)
                    except:
                        data_item["score"] = 0
                        print("error")

        elif data_item["type"] == "chart parsing en":
            answer = data_item["answers"][0]
            if data_item["predict"]:

                pred_chart_dict = convert_str_to_multi_dict(data_item["predict"])
                if len(pred_chart_dict) == 0:
                    data_item["score"] = 0
                else:
                    pred_chart_html = dict_to_html(pred_chart_dict)
                    # GT `answer` is a dict-spec string (same format as predict);
                    # convert it via convert_str_to_multi_dict before dict_to_html,
                    # otherwise dict_to_html does data.items() on a str and crashes.
                    gt_chart_dict = convert_str_to_multi_dict(answer) if isinstance(answer, str) else answer
                    if not gt_chart_dict:
                        data_item["score"] = 0
                    else:
                        gt_chart_html = dict_to_html(gt_chart_dict)
                        try:
                            data_item["score"] = teds.evaluate(pred_chart_html, gt_chart_html)
                        except Exception:
                            data_item["score"] = 0
            else:
                data_item["score"] = 0

        elif data_item["type"] == "document parsing en":
            assert type(data_item["answers"])==list and len(data_item["answers"]) == 1
            # 对齐 OmniDocBench md2md：拆 纯文本/公式/表格 分别打分(Edit_dist/CDM/TEDS)再平均；
            # 旧的整段编辑距离 doc_parsing_edit_distance 留底，依赖缺失时由 doc_parsing_md2md_score 内部退化调用。
            data_item["score"] = doc_parsing_md2md_score(data_item["predict"], data_item["answers"][0])

        elif data_item["type"] == "document parsing cn":
            assert type(data_item["answers"])==list and len(data_item["answers"]) == 1
            data_item["score"] = doc_parsing_md2md_score(data_item["predict"], data_item["answers"][0])

        elif data_item["type"] == "key information extraction en" or data_item["type"] == "key information mapping en":
            assert len(data_item["answers"]) == 1
            answers = generate_combinations(data_item["answers"][0])

            if type(answers)==list and len(answers) == 1:
                if not isinstance(data_item["predict"], str):
                    data_item["score"] = 0
                else:
                    pred_kie_dict = convert_str_to_dict(data_item["predict"])
                    data_item["score"] = compute_f1_score(pred_kie_dict, answers[0])
            else:
                max_score = 0
                for answer in answers:
                    pred_kie_dict = convert_str_to_dict(data_item["predict"])
                    data_item["score"] = compute_f1_score(pred_kie_dict, answer)

                    if data_item["score"] > max_score:
                        max_score = data_item["score"]
                data_item["score"] = max_score

        elif data_item["type"] == "key information extraction cn":
            assert len(data_item["answers"]) == 1
            answers = ast.literal_eval(data_item["answers"][0])
            answers = {k: v if isinstance(v, list) else [v] for k, v in answers.items()}
            answers = generate_combinations(answers)
            if type(answers)==list and len(answers) == 1:
                if not isinstance(data_item["predict"], str):
                    data_item["score"] = 0
                else:
                    pred_kie_dict = convert_str_to_dict(data_item["predict"])
                    data_item["score"] = compute_f1_score(pred_kie_dict, answers[0])
            else:
                max_score = 0
                for answer in answers:
                    pred_kie_dict = convert_str_to_dict(data_item["predict"])
                    data_item["score"] = compute_f1_score(pred_kie_dict, answer)

                    if data_item["score"] > max_score:
                        max_score = data_item["score"]
                data_item["score"] = max_score

        elif data_item["type"] == "VQA with position en":
            if not isinstance(data_item["predict"], str):
                data_item["score"] = 0
            else:
                pred_dict = convert_str_to_dict(data_item["predict"])
                data_item["score"] = vqa_with_position_evaluation(pred_dict, data_item)

        elif data_item["type"] == "text translation cn":
            if os.environ.get('OCRBENCH_SKIP_LLM_JUDGE'):
                # LLM-judge 子类暂不测：置 None 并标记 ignore，使其从聚合中排除，
                # 避免依赖 gemini judge 链路。需要时 unset 该环境变量即可恢复 LLM 评测。
                data_item["score"] = None
                data_item["ignore"] = "True"
            elif not isinstance(data_item["predict"], str) or len(data_item["predict"]) == 0:
                data_item["score"] = 0
            elif len(data_item["answers"][0]) == 0:
                data_item["score"] = 0
                data_item["ignore"] = "True"
            elif judge_model is not None:
                data_item["score"] = None
                text_translation_indices.append(index)
                text_translation_tasks.append(dict(
                    judge_model=judge_model,
                    predict=data_item["predict"],
                    answers=data_item["answers"]
                ))
            else:
                ocr_metric = cal_per_metrics(data_item["predict"], data_item["answers"][0])
                data_item["score"] = (ocr_metric["bleu"] + ocr_metric["meteor"] + ocr_metric["f_measure"] + (1 - ocr_metric["edit_dist"])) / 4

        elif data_item["type"] == "fine-grained text recognition en":
            if not isinstance(data_item["predict"], str):
                data_item["score"] = 0
            elif len(data_item["predict"]) == 0:
                data_item["score"] = 0
            else:
                # 改用编辑距离打分 (1 - edit_dist/max(len))，口径同 OmniDocBench Edit_dist
                data_item["score"] = ocr_edit_score(data_item["predict"], data_item["answers"][0])
        elif data_item["type"] == "full-page OCR en":
            if not data_item["predict"]:
                data_item["score"] = 0
            else:
                # 改用编辑距离打分
                data_item["score"] = ocr_edit_score(data_item["predict"], data_item["answers"][0])

        elif data_item["type"] == "full-page OCR cn":
            if not isinstance(data_item["predict"], str):
                data_item["score"] = 0
            else:
                if len(data_item["predict"]) == 0:
                    data_item["score"] = 0
                else:
                    # 改用编辑距离打分
                    data_item["score"] = ocr_edit_score(data_item["predict"], data_item["answers"][0])

        elif data_item["type"] == "text grounding en":
            if not isinstance(data_item["predict"], str):
                data_item["score"] = 0
            else:
                predict_bbox = extract_coordinates(data_item["predict"])
                if not predict_bbox:
                    data_item["score"] = 0
                else:
                    data_item["score"] = calculate_iou(predict_bbox, data_item["answers"])

        elif data_item["type"] == "text spotting en":
            if not isinstance(data_item["predict"], str):
                data_item["score"] = 0
            else:
                predict_bbox = extract_bounding_boxes_robust(data_item["predict"])
                if not predict_bbox:
                    data_item["score"] = 0
                else:
                    data_item["score"] = spotting_evaluation(predict_bbox, data_item)

        else:
            raise ValueError("Unknown task type!")

        res_data_list.append(data_item)

    if qa_tasks:
        from vlmeval.utils import track_progress_rich  # 延迟导入 (见 process_predictions 顶部说明)
        total = len(qa_tasks)
        print(f"\n{'='*60}")
        print(f"[QA VLM Judge] Starting evaluation: {total} items (mode={qa_judge_mode}), {nproc} threads")
        print(f"{'='*60}")
        completed = [0]

        def _qa_judge_worker_with_progress(judge_model, question, predict, answers):
            result = _qa_judge_worker(judge_model, question, predict, answers)
            completed[0] += 1
            if completed[0] % 10 == 0 or completed[0] == total:
                print(f"[QA VLM Judge] Progress: {completed[0]}/{total} ({100*completed[0]//total}%)")
            return result

        qa_scores = track_progress_rich(_qa_judge_worker_with_progress, qa_tasks, nproc=nproc)
        rescued = 0
        for idx, rule_score, score in zip(qa_indices, qa_rule_scores, qa_scores):
            judge_score = score if score is not None else 0
            final = judge_score if qa_judge_mode == 'full' else max(rule_score, judge_score)
            if final > (rule_score or 0):
                rescued += 1
            res_data_list[idx]["score"] = final
        print(f"[QA VLM Judge] Done. Evaluated {total} items, rescued {rescued} (rule<final).")
        print(f"{'='*60}\n")

    if formula_tasks:
        from vlmeval.utils import track_progress_rich  # 延迟导入 (见 process_predictions 顶部说明)
        total = len(formula_tasks)
        print(f"\n{'='*60}")
        print(f"[Formula VLM Judge] Starting evaluation: {total} items, {nproc} threads")
        print(f"{'='*60}")
        completed = [0]

        def _formula_judge_worker_with_progress(judge_model, predict, answers, lang):
            result = _formula_judge_worker(judge_model, predict, answers, lang)
            completed[0] += 1
            if completed[0] % 10 == 0 or completed[0] == total:
                print(f"[Formula VLM Judge] Progress: {completed[0]}/{total} ({100*completed[0]//total}%)")
            return result

        formula_scores = track_progress_rich(_formula_judge_worker_with_progress, formula_tasks, nproc=nproc)
        for idx, score in zip(formula_indices, formula_scores):
            res_data_list[idx]["score"] = score if score is not None else 0
        print(f"[Formula VLM Judge] Done. Evaluated {total} items.")
        print(f"{'='*60}\n")

    if text_translation_tasks:
        from vlmeval.utils import track_progress_rich  # 延迟导入 (见 process_predictions 顶部说明)
        total = len(text_translation_tasks)
        print(f"\n{'='*60}")
        print(f"[Text Translation VLM Judge] Starting evaluation: {total} items, {nproc} threads")
        print(f"{'='*60}")
        completed = [0]

        def _text_translation_judge_worker_with_progress(judge_model, predict, answers):
            result = _text_translation_judge_worker(judge_model, predict, answers)
            completed[0] += 1
            if completed[0] % 10 == 0 or completed[0] == total:
                print(f"[Text Translation VLM Judge] Progress: {completed[0]}/{total} ({100*completed[0]//total}%)")
            return result

        text_translation_scores = track_progress_rich(
            _text_translation_judge_worker_with_progress,
            text_translation_tasks,
            nproc=nproc
        )
        for idx, score in zip(text_translation_indices, text_translation_scores):
            res_data_list[idx]["score"] = score if score is not None else 0
        print(f"[Text Translation VLM Judge] Done. Evaluated {total} items.")
        print(f"{'='*60}\n")

    return res_data_list



def ocrbench_v2_aggregate_accuracy(data_list):
    en_text_recognition_list, en_text_detection_list, en_text_spotting_list, en_relationship_extraction_list = [], [], [], []
    en_element_parsing_list, en_mathematical_calculation_list, en_visual_text_understanding_list = [], [], []
    en_knowledge_reasoning_list = []

    cn_text_recognition_list, cn_relationship_extraction_list = [], []
    cn_element_parsing_list, cn_visual_text_understanding_list = [], []
    cn_knowledge_reasoning_list = []

    res_list = []
    for item in data_list:
        if "ignore" in item.keys():
            assert item["ignore"] == "True"

        elif item["type"] == "text recognition en" or item["type"] == "fine-grained text recognition en" or item["type"] == "full-page OCR en":
            en_text_recognition_list.append(item["score"])

        elif item["type"] == "text grounding en" or item["type"] == "VQA with position en":
            en_text_detection_list.append(item["score"])

        elif item["type"] == "text spotting en":
            en_text_spotting_list.append(item["score"])

        elif item["type"] == "key information extraction en" or item["type"] == "key information mapping en":
            en_relationship_extraction_list.append(item["score"])

        elif item["type"] == "document parsing en" or item["type"] == "chart parsing en" \
        or item["type"] == "table parsing en" or item["type"] == "formula recognition en":
            en_element_parsing_list.append(item["score"])

        elif item["type"] == "math QA en" or item["type"] == "text counting en":
            en_mathematical_calculation_list.append(item["score"])

        elif item["type"] == "document classification en" \
        or item["type"] == "cognition VQA en" or item["type"] == "diagram QA en":
            en_visual_text_understanding_list.append(item["score"])

        elif item["type"] == "reasoning VQA en" or item["type"] == "science QA en" \
        or item["type"] == "APP agent en" or item["type"] == "ASCII art classification en":
            en_knowledge_reasoning_list.append(item["score"])

        elif item["type"] == "full-page OCR cn":
            cn_text_recognition_list.append(item["score"])

        elif item["type"] == "key information extraction cn" or item["type"] == "handwritten answer extraction cn":
            cn_relationship_extraction_list.append(item["score"])

        elif item["type"] == "document parsing cn" or item["type"] == "table parsing cn" or item["type"] == "formula recognition cn":
            cn_element_parsing_list.append(item["score"])

        elif item["type"] == "cognition VQA cn":
            cn_visual_text_understanding_list.append(item["score"])

        elif item["type"] == "reasoning VQA cn" or item["type"] == "text translation cn":
            cn_knowledge_reasoning_list.append(item["score"])

        else:
            raise ValueError("Unknown task type!")

    en_scores = {
        "en_text_recognition": en_text_recognition_list,
        "en_text_detection": en_text_detection_list,
        "en_text_spotting": en_text_spotting_list,
        "en_relationship_extraction": en_relationship_extraction_list,
        "en_element_parsing": en_element_parsing_list,
        "en_mathematical_calculation": en_mathematical_calculation_list,
        "en_visual_text_understanding": en_visual_text_understanding_list,
        "en_knowledge_reasoning": en_knowledge_reasoning_list
    }

    cn_scores = {
        "cn_text_recognition": cn_text_recognition_list,
        "cn_relationship_extraction": cn_relationship_extraction_list,
        "cn_element_parsing": cn_element_parsing_list,
        "cn_visual_text_understanding": cn_visual_text_understanding_list,
        "cn_knowledge_reasoning": cn_knowledge_reasoning_list
    }

    en_averages = calculate_average(en_scores)
    cn_averages = calculate_average(cn_scores)

    return en_averages,cn_averages
