import json
import math
import os
import re

import ipdb


_CDM_PDFLATEX_READY = None
def _cdm_pdflatex_ready():
    """CDM 公式评测依赖系统级 pdflatex; 缺失则提前返回 0, 避免逐条 subprocess 兜底。
    结果按进程缓存 (一次检测即可)。"""
    global _CDM_PDFLATEX_READY
    if _CDM_PDFLATEX_READY is not None:
        return _CDM_PDFLATEX_READY
    try:
        from vlmeval.dataset.utils.cdm.modules.texlive_env import resolve_tex_binary
        p = resolve_tex_binary('pdflatex')
        r = bool(p) and os.path.isfile(p) and os.access(p, os.X_OK)
    except Exception:
        r = False
    _CDM_PDFLATEX_READY = r
    return r


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def vqa_evaluation(predict, answers):
    score = 0
    if isinstance(answers, list):
        for j in range(len(answers)):
            if isinstance(answers[j], (int, float)):
                answers[j] = str(answers[j])
            try:
                answer = answers[j].lower().strip().replace("\n", " ")
            except Exception:
                ipdb.set_trace()
            if isinstance(predict, (int, float)):
                predict = str(predict)
            predict = predict.lower().strip().replace("\n", " ")
            if len(answer.split()) < 5:
                if answer in predict:
                    score = 1
            else:
                dist = levenshtein_distance(predict, answer)
                length = max(len(predict), len(answer))
                ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
                ANLS_value = 1 - ANLS_value

                if ANLS_value >= 0.5 and ANLS_value > score:
                    score = ANLS_value

    else:
        answers = answers.lower().strip().replace("\n", " ")
        predict = predict.lower().strip().replace("\n", " ")
        if len(answers.split()) < 5:
            if answers in predict:
                score = 1
            else:
                dist = levenshtein_distance(predict, answers)
                length = max(len(predict), len(answers))
                ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
                ANLS_value = 1 - ANLS_value

                if ANLS_value >= 0.5 and ANLS_value > score:
                    score = ANLS_value

    return score


def cn_vqa_evaluation(predict, answers):
    score = 0
    if isinstance(answers, list):
        for j in range(len(answers)):
            if isinstance(answers[j], (int, float)):
                answers[j] = str(answers[j])
            try:
                answer = answers[j].lower().strip().replace("\n", " ").replace(" ", "")
            except Exception:
                ipdb.set_trace()
            if isinstance(predict, (int, float)):
                predict = str(predict)
            predict = predict.lower().strip().replace("\n", " ").replace(" ", "")
            if len(answer.split(",")) < 4:
                if answer in predict:
                    score = 1
            else:
                dist = levenshtein_distance(predict, answer)
                length = max(len(predict), len(answer))
                ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
                ANLS_value = 1 - ANLS_value

                if ANLS_value >= 0.5 and ANLS_value > score:
                    score = ANLS_value

    else:
        answers = answers.lower().strip().replace("\n", " ").replace(" ", "")
        predict = predict.lower().strip().replace("\n", " ").replace(" ", "")
        if len(answer.split(",")) < 4:
            if answers in predict:
                score = 1
            else:
                dist = levenshtein_distance(predict, answers)
                length = max(len(predict), len(answers))
                ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
                ANLS_value = 1 - ANLS_value

                if ANLS_value >= 0.5 and ANLS_value > score:
                    score = ANLS_value

    return score


def vqa_evaluation_case_sensitive(predict, answers):
    score = 0
    if isinstance(answers, list):
        for j in range(len(answers)):
            if isinstance(answers[j], (int, float)):
                answers[j] = str(answers[j])
            try:
                answer = answers[j].strip().replace("\n", " ")
            except Exception:
                ipdb.set_trace()
            predict = predict.strip().replace("\n", " ")
            if len(answer.split()) < 5:
                if answer in predict:
                    score = 1
            else:
                dist = levenshtein_distance(predict, answer)
                length = max(len(predict), len(answer))
                ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
                ANLS_value = 1 - ANLS_value

                if ANLS_value >= 0.5 and ANLS_value > score:
                    score = ANLS_value

    else:
        answers = answers.strip().replace("\n", " ")
        predict = predict.strip().replace("\n", " ")
        if len(answers.split()) < 5:
            if answers in predict:
                score = 1
            else:
                dist = levenshtein_distance(predict, answers)
                length = max(len(predict), len(answers))
                ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
                ANLS_value = 1 - ANLS_value

                if ANLS_value >= 0.5 and ANLS_value > score:
                    score = ANLS_value

    return score


def extract_first_number(string):
    match = re.search(r'\d+', string)
    if match:
        return int(match.group())
    return None


def counting_evaluation(predict, answers, eval_method):
    score = 0

    if isinstance(predict, str):
        predict_processed = predict.lower().strip().replace("\n", " ")
    elif math.isnan(predict):
        return 0
    else:
        predict_processed = int(predict)
    if isinstance(answers, list):
        temp_score = 0
        for j in range(len(answers)):
            if isinstance(answers[j], (int, float)):
                answers[j] = str(answers[j])
            answer = answers[j].lower().strip().replace("\n", " ")
            if eval_method == "exact match":
                if answer in predict:
                    score = 1
                else:
                    score = 0
            elif eval_method == "regression":
                predict_number = extract_first_number(predict_processed)
                if predict_number:

                    answer = int(answer)

                    if predict_number <= 0 or predict_number >= 2 * answer:
                        score = 0
                    else:
                        iou = 1 - abs(predict_number - answer) / answer
                        if iou > 0.5:
                            score = iou
                        else:
                            score = 0
                else:
                    score = 0
            if score > temp_score:
                temp_score = score
        score = temp_score

    else:
        answers = answers.lower().strip().replace("\n", " ")
        predict = predict.lower().strip().replace("\n", " ")
        if eval_method == "exact match":
            if answer in predict:
                score = 1
            else:
                score = 0
        elif eval_method == "regression":
            predict = extract_first_number(predict)
            if predict:
                answer = int(answer)
                if predict <= 0 or predict >= 2 * answer:
                    score = 0
                else:
                    iou = 1 - abs(predict - answer) / answer

                    if iou > 0.5:
                        score = iou
                    else:
                        score = 0
            else:
                score = 0
    return score


def math_expression_evaluation_legacy(predict, answers):
    # 旧版：去空格后子串包含(answer in predict)，存在假阳性（如 answer="1" 命中 predict="100"）。保留用于回算对照。
    score = 0
    if isinstance(answers, list):
        for j in range(len(answers)):
            answer = answers[j].strip().replace("\n", " ").replace(" ", "")
            predict_proc = predict.strip().replace("\n", " ").replace(" ", "")
            if answer in predict_proc:
                score = 1
    else:
        answers = answers.strip().replace("\n", " ").replace(" ", "")
        predict_proc = predict.strip().replace("\n", " ").replace(" ", "")
        if answers in predict_proc:
            score = 1
    return score


def math_expression_evaluation(predict, answers):
    # 新版：去空格后整体相等才得 1，避免旧版子串匹配的假阳性。
    # （math QA en 也复用此函数；公式无 judge 时走这里。）
    if not isinstance(predict, str):
        return 0
    predict_proc = predict.strip().replace("\n", " ").replace(" ", "")
    if isinstance(answers, list):
        for j in range(len(answers)):
            answer = answers[j].strip().replace("\n", " ").replace(" ", "")
            if answer == predict_proc:
                return 1
    else:
        answers = answers.strip().replace("\n", " ").replace(" ", "")
        if answers == predict_proc:
            return 1
    return 0


FORMULA_JUDGE_PROMPT = """You are a formula recognition evaluator. Compare the predicted formula with the ground truth formula and judge whether they would look the same when rendered (e.g. by LaTeX or MathML).

Focus on whether the rendered visual output is identical, not mathematical equivalence. Minor differences in LaTeX commands that produce the same rendered result (e.g. \\frac vs \\dfrac with same visual, extra braces, etc.) should be considered the same.

Predicted formula:
{predict}

Ground truth formula:
{answer}

Please give a score between 0 and 1:
- 1.0: The two formulas render identically
- 0.5-0.9: Minor visual differences (e.g. slightly different spacing, optional symbols)
- 0.1-0.4: Major structural differences but some parts match
- 0.0: Completely different or unrecognizable

Output ONLY a single number between 0 and 1 (e.g. 0.8). Do not output anything else."""


def _parse_score(response):
    if not response:
        return 0.0
    response = response.strip()
    try:
        score = float(response)
        return max(0.0, min(1.0, score))
    except (ValueError, TypeError):
        # 只抠合法浮点字面量; 旧正则 ([\d.]+) 会把 "API." 的句号抠出来导致 float('.') 抛错。
        match = re.search(r'(\d+(?:\.\d+)?|\.\d+)', response)
        if match:
            try:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            except ValueError:
                pass
        return 0.0


def formula_vlm_judge_evaluation(judge_model, predict, answers):
    if not isinstance(predict, str) or len(predict.strip()) == 0:
        return 0.0

    if not isinstance(answers, list):
        answers = [answers]

    max_score = 0.0
    for answer in answers:
        prompt = FORMULA_JUDGE_PROMPT.format(predict=predict.strip(), answer=answer.strip())
        try:
            response = judge_model.generate(prompt)
            score = _parse_score(response)
            if score > max_score:
                max_score = score
        except Exception:
            pass
    return max_score


TEXT_TRANSLATION_JUDGE_PROMPT = """You are a translation evaluator for OCRBench v2. The task may be Chinese-to-English or English-to-Chinese. Compare the model's translated text with the reference translation and judge whether they express the same meaning in the intended target language.

Evaluation principles:
- Focus on semantic equivalence, not word-by-word matching.
- Natural paraphrases, synonym substitutions, and word-order differences should receive high scores if the meaning is preserved.
- Do not penalize harmless formatting or punctuation differences.
- Penalize mistranslation, missing key information, wrong entities/numbers/negation/relations, or using the wrong target language.
- If the model output contains extra explanation, score based on whether the reference meaning is still complete and not contradicted.

Model translation:
{predict}

Reference translation:
{answer}

Please give a score between 0 and 1:
- 1.0: Semantically equivalent, fluent, and complete
- 0.7-0.9: Mostly equivalent with only minor issues
- 0.4-0.6: Partially correct but with obvious omissions or mistranslations
- 0.1-0.3: Only a few fragments are relevant; overall meaning is wrong
- 0.0: Completely wrong, unrelated, empty, or unreadable

Output ONLY a single number between 0 and 1 (e.g. 0.8). Do not output anything else."""


def text_translation_vlm_judge_evaluation(judge_model, predict, answers):
    if not isinstance(predict, str) or len(predict.strip()) == 0:
        return 0.0

    if not isinstance(answers, list):
        answers = [answers]

    max_score = 0.0
    for answer in answers:
        if not isinstance(answer, str):
            answer = str(answer)
        if len(answer.strip()) == 0:
            continue
        prompt = TEXT_TRANSLATION_JUDGE_PROMPT.format(predict=predict.strip(), answer=answer.strip())
        try:
            response = judge_model.generate(prompt)
            score = _parse_score(response)
            if score > max_score:
                max_score = score
        except Exception:
            pass
    return max_score


# CharXiv 风格 QA judge 模板 (对齐 vlmeval/dataset/charxiv.py 的 grading_query 形态):
# 任务定义 + JSON 输出契约 (extracted_answer 先抽取, score 后打分) + 2 条判分规则
# + 2 个 few-shot 示例 (一错一对) + Your Turn。多候选 GT 以 "; " 拼接一次给出。
QA_JUDGE_PROMPT = """You will be given a question, one or more ground truth answers and a model response. You need to extract the final answer from the model response, compare it with the ground truth answer(s), and then assign a binary score. Avoid providing explanations in your response. If there is no provided model response, please leave the extracted answer empty and give a score of 0.

Your response must follow json formats with keys [extracted_answer, score] where the value of the score is an integer in [0, 1]. The question and answers may be in English or Chinese. You must follow the scoring rules:

    ### Rules ###
    * Give a score of 1 if and only if the final answer refers to the same term as ANY ONE of the ground truth answers. It's acceptable to have different grammar, letter case, punctuation, whitespace or date/time/number formats (e.g., α and alpha; "15 March" and "March 15"; "12:42 PM" and "12:42 p.m."), missing/extra units or currency symbols, and harmless extra words (e.g., "The answer is X"). It's also acceptable to have different orders of the terms when the question asks for multiple terms.
    * Give a score of 0 if the final answer differs from ALL ground truth answers on any term (e.g., ACC+ and ACC; P-101 and P=101; "APPVIA" and "APPV1A"), contradicts the ground truth, answers a different aspect of the question, or is empty/irrelevant.

    ### Example 1 Starts ###
    * Question: What is the coupon code?
    * Ground Truth(s): APPVIA
    * Response: After applying the discount, the coupon code shown on the page is APPV1A.

    {{
        "extracted_answer": "APPV1A",
        "score": 0
    }}
    ### Example 1 Ends ###


    ### Example 2 Starts ###
    * Question: What is the date of travel?
    * Ground Truth(s): 15 March; 15 MAR; March 15
    * Response: The date of travel is March 15th.

    {{
        "extracted_answer": "March 15th",
        "score": 1
    }}
    ### Example 2 Ends ###


    ### Your Turn ###
    * Question: {question}
    * Ground Truth(s): {answers}
    * Response: {predict}"""


def _parse_qa_judge_json(response):
    """解析 CharXiv 风格 JSON judge 输出 {"extracted_answer","score"}。

    兼容 ```json 代码块包裹与前后多余文字 (抠第一个 {...} 块); score 收敛到 [0,1]。
    解析失败返回 None (由调用方重试/兜底)。
    """
    if not response or not isinstance(response, str):
        return None
    text = response.strip()
    if text.startswith('```'):
        text = re.sub(r'^```[a-zA-Z]*\s*|\s*```$', '', text, flags=re.S).strip()
    try:
        obj = json.loads(text)
    except Exception:
        match = re.search(r'\{.*\}', text, flags=re.S)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
        except Exception:
            return None
    if not isinstance(obj, dict) or 'score' not in obj:
        return None
    try:
        return max(0.0, min(1.0, float(obj['score'])))
    except Exception:
        return None


# ---- judge 输入去思维链 ----
# qa_think 配置下模型常输出长篇结构化推理 (### Phase 1..4 + <answer>X</answer>), 整段喂 judge 有两个问题:
#   1) 慢: 单条 prompt 数千~上万 token, 实测 4~30s/条, 还会撞网关 504;
#   2) 偏: judge 要在一堆中间假设里找最终答案, 反而容易被推理噪声带偏。
# 规则分仍按原始预测算 (确定性基线不变), 只在进 judge 前把预测收敛成"最终答案"。
_ANSWER_TAG_RE = re.compile(r'<answer>(.*?)</answer>', re.S | re.I)
_FINAL_ANS_TAIL_RE = re.compile(
    r'(?:final\s+answer\s*(?:is|:)?|最终答案(?:是|:)?|答案是|正确答案(?:是|:)?)[\s:："\'`]*([^\n]+)',
    re.I)
_JUDGE_TAIL_CHARS = 400


def strip_reasoning_for_judge(predict, max_chars=_JUDGE_TAIL_CHARS):
    """把带思维链的预测收敛为最终答案文本; 已经是短答案时原样返回。

    优先级: <answer>标签内容(取最后一个) > "final answer is ..." 句式之后 > 最后一行 > 截断尾部。
    取最后一个标签: 模型可能中途改答案, 末次为准 (与人工核对口径一致)。
    """
    if not isinstance(predict, str):
        return predict
    p = predict.strip()
    if not p:
        return p
    tags = [t.strip() for t in _ANSWER_TAG_RE.findall(p)]
    tags = [t for t in tags if t]
    if tags:
        return tags[-1]
    if len(p) <= max_chars:
        return p
    m = _FINAL_ANS_TAIL_RE.search(p[-max_chars:])
    if m and m.group(1).strip():
        return m.group(1).strip()
    lines = [l.strip() for l in p.splitlines() if l.strip()]
    return lines[-1] if lines else p[-max_chars:]


def qa_vlm_judge_evaluation(judge_model, question, predict, answers, retries=3):
    """QA 类自由问答的 LLM-judge (CharXiv 风格): 一次调用带全部候选答案, JSON 判 0/1。

    JSON 解析失败重试 retries 次; 全部失败退化 _parse_score 抠数字 (抠不到则 0, fail-safe)。
    预测先经 strip_reasoning_for_judge 去掉思维链, 只把最终答案交给 judge。
    """
    if not isinstance(predict, str) or len(predict.strip()) == 0:
        return 0.0
    predict = strip_reasoning_for_judge(predict)
    if not predict.strip():
        return 0.0
    if not isinstance(answers, list):
        answers = [answers]
    answers = [str(a).strip() for a in answers if str(a).strip()]
    if not answers:
        return 0.0
    answers_text = "; ".join(answers)
    if not isinstance(question, str):
        question = str(question)
    prompt = QA_JUDGE_PROMPT.format(question=question, answers=answers_text, predict=predict.strip())
    last_response = None
    for _ in range(max(1, retries)):
        try:
            last_response = judge_model.generate(prompt)
            score = _parse_qa_judge_json(last_response)
            if score is not None:
                return score
        except Exception:
            pass
    return _parse_score(last_response)


def remove_text_tags(latex_str):
    """
    Removes LaTeX \text{...} tags while keeping their content.

    :param latex_str: A string containing LaTeX expressions
    :return: The processed string with \text{...} tags removed
    """

    pattern = r'\\text\{([^{}]*)\}'

    processed_str = re.sub(pattern, r'\1', latex_str)

    return processed_str


def cn_math_expression_evaluation_legacy(predict, answers):
    # 旧版：去\\text{}+去空格子串包含，有假阳性。保留用于回算对照。
    score = 0
    assert len(answers) == 1
    answers = [remove_text_tags(answers[0])]
    predict = remove_text_tags(predict)
    for j in range(len(answers)):
        answer = answers[j].strip().replace("\n", " ").replace(" ", "")
        predict_proc = predict.strip().replace("\n", " ").replace(" ", "")
        if answer in predict_proc:
            score = 1
    return score


def cn_math_expression_evaluation(predict, answers):
    # 新版：去\\text{}+去空格后整体相等才 1，修复子串假阳性。
    if not isinstance(predict, str):
        return 0
    assert len(answers) == 1
    answer = remove_text_tags(answers[0]).strip().replace("\n", " ").replace(" ", "")
    predict_proc = remove_text_tags(predict).strip().replace("\n", " ").replace(" ", "")
    return 1 if answer == predict_proc else 0


def cn_formula_vlm_judge_evaluation(judge_model, predict, answers):
    if not isinstance(predict, str) or len(predict.strip()) == 0:
        return 0.0

    assert len(answers) == 1
    answer = remove_text_tags(answers[0])
    predict = remove_text_tags(predict)

    prompt = FORMULA_JUDGE_PROMPT.format(predict=predict.strip(), answer=answer.strip())
    try:
        response = judge_model.generate(prompt)
        return _parse_score(response)
    except Exception:
        pass
    return 0.0


# ===================== CDM (OmniDocBench 公式评测) 适配 =====================
# 公式识别改用 CDM：渲染 LaTeX 成图 -> token 匈牙利匹配 -> RANSAC -> F1。
# 替代上面的 LLM judge 与子串匹配退化路径。需要系统级 pdflatex / convert / gs。

def _strip_cdm_math_wrappers(text):
    text = str(text or '').strip()
    # 去掉 $$...$$ / $...$ / \[...\] / \(...\) 公式定界符
    while text.startswith('$$') and text.endswith('$$') and len(text) > 4:
        text = text[2:-2].strip()
    while text.startswith('$') and text.endswith('$') and len(text) > 2:
        text = text[1:-1].strip()
    if text.startswith('\\[') and text.endswith('\\]'):
        text = text[2:-2].strip()
    if text.startswith('\\(') and text.endswith('\\)'):
        text = text[2:-2].strip()
    # 去 \text{} 标签，保留内容
    text = remove_text_tags(text).strip()
    return text


def _pred_latex_for_cdm(predict):
    """从模型输出中抠出 LaTeX：支持 ```latex...``` 代码块或裸 LaTeX。"""
    if not isinstance(predict, str):
        return ''
    if '```latex' in predict:
        predict = predict.split('```latex')[-1].split('```')[0]
    return _strip_cdm_math_wrappers(predict)


def formula_cdm_score(predict, answers):
    """用 OmniDocBench CDM 计算公式识别 F1。

    - predict: 模型输出的 LaTeX（可含 $$/代码块定界符，会自动剥离）
    - answers: list 形式的候选 GT LaTeX；对每个候选拿最高 F1
    - predict 非字符串/为空 或 CDM 依赖缺失/渲染异常 -> 0
    依赖系统级 pdflatex / convert(ImageMagick) / gs，缺任一则返回 0 并打印一次告警。
    """
    pred_latex = _pred_latex_for_cdm(predict)
    if not pred_latex:
        return 0.0
    # pdflatex 缺失 (TeX Live 未安装) -> 报错, 不静默置 0。(打分脚本会前置自检拦截; 直接调用也会报错)
    if not _cdm_pdflatex_ready():
        raise RuntimeError(
            'CDM 依赖缺失: 未检测到可执行的 pdflatex (TeX Live 未安装/不在 PATH)。'
            '公式识别需 TeX Live(含 CJK + gkai)。请装 scheme-basic + collection-latexrecommended '
            '+ collection-latexextra + collection-langcjk + collection-langchinese, '
            '装到 .tools/texlive/2025 或设 CDM_TEXLIVE_ROOT。先 python check_cdm_deps.py 看缺啥。')
    if not isinstance(answers, list):
        answers = [answers]

    try:
        from vlmeval.dataset.utils.cdm import cdm as _cdm
    except Exception as exc:
        # 仅首次缺失时提醒，避免逐条刷屏
        if not getattr(formula_cdm_score, '_warn_no_cdm', False):
            print(f'[CDM] 公式评测依赖未就绪, 公式得分置 0: {exc}')
            formula_cdm_score._warn_no_cdm = True
        return 0.0

    max_score = 0.0
    for ans in answers:
        gt_latex = _strip_cdm_math_wrappers(ans)
        if not gt_latex:
            continue
        try:
            score = _cdm(gt_latex, pred_latex)  # 返回 F1
        except Exception:
            score = 0.0
        if score > max_score:
            max_score = score
    return max_score


if __name__ == "__main__":
    test_predict = "apple pie and banana"
    test_answers = ["apple", "banana pie", "apple pie and orange"]

    vqa_score = vqa_evaluation(test_predict, test_answers)
    print(f"VQA evaluation score for predict '{test_predict}' and answers {test_answers}: {vqa_score}")
