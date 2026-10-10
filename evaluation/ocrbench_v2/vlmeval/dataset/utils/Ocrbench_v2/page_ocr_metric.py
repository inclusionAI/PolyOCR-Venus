import re

import jieba
import nltk
from nltk.metrics import f_measure, precision, recall
from nltk.translate import meteor_score


def _fast_edit_distance(a, b):
    """字符级编辑距离: 优先 C 实现的 Levenshtein (O(n*m) 但在 C 里, 大字符串快几个数量级),
    缺失则退回 nltk.edit_distance (纯 python, 仅用于短串/兜底)。"""
    try:
        import Levenshtein
        return Levenshtein.distance(a, b)
    except Exception:
        return nltk.edit_distance(a, b)


def contain_chinese_string(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')
    return bool(chinese_pattern.search(text))


def cal_per_metrics(pred, gt):
    metrics = {}

    if contain_chinese_string(gt) or contain_chinese_string(pred):
        reference = jieba.lcut(gt)
        hypothesis = jieba.lcut(pred)
    else:
        reference = gt.split()
        hypothesis = pred.split()

    metrics["bleu"] = nltk.translate.bleu([reference], hypothesis)
    try:
        metrics["meteor"] = meteor_score.meteor_score([reference], hypothesis)
    except Exception:
        # wordnet/omw corpus may be unavailable on this box (no network to fetch);
        # degrade to 0.0 instead of crashing the whole eval.
        metrics["meteor"] = 0.0

    reference = set(reference)
    hypothesis = set(hypothesis)
    metrics["f_measure"] = f_measure(reference, hypothesis)

    metrics["precision"] = precision(reference, hypothesis)
    metrics["recall"] = recall(reference, hypothesis)
    metrics["edit_dist"] = _fast_edit_distance(pred, gt) / max(len(pred), len(gt))
    return metrics


def ocr_edit_score(pred, gt):
    """OCR 文本识别的编辑距离打分（替换原 4 指标平均）。

    口径与 OmniDocBench Edit_dist 一致：
        score = 1 - edit_distance(pred, gt) / max(len(pred), len(gt))
    pred 或 gt 非字符串/为空 -> 0。
    旧的 4 指标平均实现保留下方 cal_per_metrics（不改），便于回算对照。
    """
    if not isinstance(pred, str) or not isinstance(gt, str):
        return 0.0
    if len(pred) == 0 and len(gt) == 0:
        return 1.0
    denom = max(len(pred), len(gt))
    if denom == 0:
        return 0.0
    edit = _fast_edit_distance(pred, gt)
    return max(0.0, 1.0 - float(edit) / float(denom))


if __name__ == "__main__":

    # Examples for region text recognition and read all text tasks
    predict_text = "metrics['edit_dist'] = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))"
    true_text = "metrics = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))"

    scores = cal_per_metrics(predict_text, true_text)

    predict_text = "metrics['edit_dist'] len(gt))"
    true_text = "metrics = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))"

    scores = cal_per_metrics(predict_text, true_text)
    print(scores)
