import argparse
import inspect

import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from data import load_tokenized_datasets
from metrics import compute_metrics
from utils import ensure_dir, load_config, save_json


def _extract_core_metrics(metrics_dict):
    return {
        "bleu": float(metrics_dict.get("test_bleu", metrics_dict.get("bleu", 0.0))),
        "chrf": float(metrics_dict.get("test_chrf", metrics_dict.get("chrf", 0.0))),
        "gen_len": float(metrics_dict.get("test_gen_len", metrics_dict.get("gen_len", 0.0))),
    }


def _decode_predictions(predictions, tokenizer):
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if predictions.ndim == 3:
        predictions = np.argmax(predictions, axis=-1)
    return tokenizer.batch_decode(predictions, skip_special_tokens=True)


def _trainer_processing_kwargs(tokenizer):
    trainer_signature = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}


def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation for EN->RU MT")
    parser.add_argument("--config", default="configs/train_config.json", help="Path to config JSON")
    args = parser.parse_args()

    config = load_config(args.config)

    ensure_dir("outputs")
    ensure_dir("outputs/baseline_eval_tmp")

    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    tokenized_splits, raw_splits = load_tokenized_datasets(config, tokenizer)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    eval_args = Seq2SeqTrainingArguments(
        output_dir="outputs/baseline_eval_tmp",
        per_device_eval_batch_size=int(config["per_device_eval_batch_size"]),
        predict_with_generate=True,
        do_train=False,
        do_eval=False,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=eval_args,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
        **_trainer_processing_kwargs(tokenizer),
    )

    predictions = trainer.predict(tokenized_splits["test"])
    metrics = _extract_core_metrics(predictions.metrics)
    metrics["num_test_samples"] = len(raw_splits["test"])

    decoded_predictions = _decode_predictions(predictions.predictions, tokenizer)

    samples = []
    for idx in range(min(20, len(raw_splits["test"]))):
        samples.append(
            {
                "src": raw_splits["test"][idx]["source"],
                "reference": raw_splits["test"][idx]["target"],
                "prediction": decoded_predictions[idx].strip(),
            }
        )

    save_json(metrics, "outputs/baseline_metrics.json")
    save_json(samples, "outputs/baseline_samples.json")

    print("Baseline evaluation completed.")
    print(f"BLEU: {metrics['bleu']:.4f}")
    print(f"chrF: {metrics['chrf']:.4f}")
    print(f"gen_len: {metrics['gen_len']:.4f}")


if __name__ == "__main__":
    main()
