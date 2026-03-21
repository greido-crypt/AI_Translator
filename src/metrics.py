from __future__ import annotations

import re
from typing import Dict, List, Tuple

import evaluate
import numpy as np
import sacrebleu

_bleu_metric = evaluate.load("sacrebleu")

TECH_TERMS_RU = [
    "фронтенд",
    "бэкенд",
    "api",
    "http",
    "код состояния",
    "кэш",
    "модуль логирования",
    "ошибки аутентификации",
    "ошибки тайм-аута",
    "проблемы соединения",
    "база данных",
    "запрос",
    "ответ",
]


def _postprocess_text(preds: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
    preds = [p.strip() for p in preds]
    labels = [l.strip() for l in labels]
    return preds, labels


def _terminology_accuracy(preds: List[str], labels: List[str]) -> float:
    covered = 0
    correct = 0

    for pred, ref in zip(preds, labels):
        pred_low = pred.lower()
        ref_low = ref.lower()
        for term in TECH_TERMS_RU:
            if re.search(r"\b" + re.escape(term) + r"\b", ref_low):
                covered += 1
                if re.search(r"\b" + re.escape(term) + r"\b", pred_low):
                    correct += 1

    if covered == 0:
        return 1.0
    return correct / covered


def compute_metrics(eval_preds, tokenizer) -> Dict[str, float]:
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    if preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    labels = np.where(labels != -100, labels, pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = _postprocess_text(decoded_preds, decoded_labels)

    bleu_result = _bleu_metric.compute(
        predictions=decoded_preds,
        references=[[label] for label in decoded_labels],
    )
    chrf_result = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels])

    terminology_acc = _terminology_accuracy(decoded_preds, decoded_labels)

    gen_lens = [np.count_nonzero(pred != pad_token_id) for pred in preds]

    return {
        "bleu": round(float(bleu_result["score"]), 4),
        "chrf": round(float(chrf_result.score), 4),
        "terminology_accuracy": round(float(terminology_acc), 4),
        "gen_len": round(float(np.mean(gen_lens)), 4),
    }
