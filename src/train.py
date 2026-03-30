import argparse
import inspect
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from data import load_tokenized_datasets
from metrics import compute_metrics
from utils import ensure_dir, load_config, save_json, set_seed


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


def _assert_cuda_or_fail() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Install NVIDIA GPU drivers + CUDA-enabled PyTorch build, then rerun.\n"
            "Example (CUDA 12.1): pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121"
        )

    gpu_name = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"CUDA device: {gpu_name} ({total_mem_gb:.2f} GB)")


def _build_precision_flags(config: dict) -> tuple[bool, bool]:
    use_bf16 = bool(config.get("bf16", False)) and torch.cuda.is_bf16_supported()
    use_fp16 = bool(config.get("fp16", True)) and not use_bf16
    return use_fp16, use_bf16


def _run_stabilization_phase(
    trainer: Seq2SeqTrainer,
    tokenizer,
    tokenized_train,
    tokenized_val,
    learning_rate: float,
    num_epochs: float,
    seed: int,
) -> Seq2SeqTrainer:
    if num_epochs <= 0:
        return trainer

    base_args = trainer.args
    stabilize_args = Seq2SeqTrainingArguments(
        output_dir=base_args.output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=base_args.per_device_train_batch_size,
        per_device_eval_batch_size=base_args.per_device_eval_batch_size,
        dataloader_num_workers=base_args.dataloader_num_workers,
        num_train_epochs=num_epochs,
        weight_decay=base_args.weight_decay,
        warmup_ratio=min(0.03, float(base_args.warmup_ratio)),
        logging_steps=base_args.logging_steps,
        eval_strategy=base_args.eval_strategy,
        save_strategy=base_args.save_strategy,
        predict_with_generate=True,
        fp16=base_args.fp16,
        bf16=base_args.bf16,
        use_cpu=False,
        seed=seed + 1,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        save_total_limit=2,
        report_to=[],
    )
    stabilized_trainer = Seq2SeqTrainer(
        model=trainer.model,
        args=stabilize_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=trainer.data_collator,
        compute_metrics=trainer.compute_metrics,
        **_trainer_processing_kwargs(tokenizer),
    )
    stabilized_trainer.train()
    return stabilized_trainer


def _predict_on_test(model, tokenizer, tokenized_test, per_device_eval_batch_size, output_dir):
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=per_device_eval_batch_size,
        dataloader_num_workers=0,
        predict_with_generate=True,
        do_train=False,
        do_eval=False,
        report_to=[],
        use_cpu=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
        **_trainer_processing_kwargs(tokenizer),
    )
    return trainer.predict(tokenized_test)


def _build_report(config, split_sizes, baseline_metrics, final_metrics):
    improved_bleu = final_metrics["bleu"] > baseline_metrics["bleu"]
    improved_chrf = final_metrics["chrf"] > baseline_metrics["chrf"]

    if improved_bleu and improved_chrf:
        conclusion = "Quality improved on both BLEU and chrF after fine-tuning."
    elif improved_bleu or improved_chrf:
        conclusion = "Quality improved partially: one metric increased after fine-tuning."
    else:
        conclusion = "No improvement on BLEU/chrF after fine-tuning with current setup."

    lines = [
        f"Model: {config['model_name']}",
        f"Dataset: {config['dataset_name']} ({config['lang_pair']})",
        (
            "Split sizes: "
            f"train={split_sizes['train']}, validation={split_sizes['validation']}, test={split_sizes['test']}"
        ),
        f"Baseline BLEU: {baseline_metrics['bleu']:.4f}",
        f"Baseline chrF: {baseline_metrics['chrf']:.4f}",
        f"Final BLEU: {final_metrics['bleu']:.4f}",
        f"Final chrF: {final_metrics['chrf']:.4f}",
        f"Conclusion: {conclusion}",
    ]
    return "\n".join(lines) + "\n"


def _merge_log_histories(main_history, stabilization_history, main_epochs: float):
    merged = list(main_history or [])
    if not stabilization_history:
        return merged

    for item in stabilization_history:
        if not isinstance(item, dict):
            continue
        shifted = dict(item)
        if "epoch" in shifted and isinstance(shifted["epoch"], (int, float)):
            shifted["epoch"] = float(shifted["epoch"]) + float(main_epochs)
        merged.append(shifted)
    return merged


def _extract_epoch_losses(log_history):
    train_bucket = {}
    eval_points = {}

    for row in log_history:
        if not isinstance(row, dict):
            continue
        epoch = row.get("epoch")
        if epoch is None:
            continue
        epoch_idx = max(1, int(np.ceil(float(epoch))))

        if "loss" in row and "eval_loss" not in row:
            train_bucket.setdefault(epoch_idx, []).append(float(row["loss"]))
        if "eval_loss" in row:
            eval_points[epoch_idx] = float(row["eval_loss"])

    epochs = sorted(set(train_bucket.keys()) | set(eval_points.keys()))
    train_losses = []
    eval_losses = []
    for ep in epochs:
        values = train_bucket.get(ep, [])
        train_losses.append(float(np.mean(values)) if values else np.nan)
        eval_losses.append(eval_points.get(ep, np.nan))
    return epochs, train_losses, eval_losses


def _save_loss_plots(log_history, out_png="outputs/loss_curve_epoch.png", out_json="outputs/loss_curve_epoch.json"):
    epochs, train_losses, eval_losses = _extract_epoch_losses(log_history)
    payload = {
        "epochs": epochs,
        "train_loss": [None if np.isnan(v) else round(float(v), 6) for v in train_losses],
        "eval_loss": [None if np.isnan(v) else round(float(v), 6) for v in eval_losses],
    }
    save_json(payload, out_json)

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print(f"matplotlib is not installed. Saved numeric loss history only: {out_json}")
        return

    if not epochs:
        print("No loss logs found to plot.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="train_loss")
    plt.plot(epochs, eval_losses, marker="s", label="eval_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Eval Loss by Epoch")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _extract_eval_series(log_history, metric_name: str):
    values_by_epoch = {}
    for row in log_history:
        if not isinstance(row, dict):
            continue
        epoch = row.get("epoch")
        if epoch is None:
            continue
        key = f"eval_{metric_name}"
        if key not in row:
            continue
        epoch_idx = max(1, int(np.ceil(float(epoch))))
        values_by_epoch[epoch_idx] = float(row[key])

    epochs = sorted(values_by_epoch.keys())
    values = [values_by_epoch[e] for e in epochs]
    return epochs, values


def _save_quality_plots(log_history, artifact_dir: str):
    ensure_dir(artifact_dir)

    metric_names = ["bleu", "chrf", "terminology_accuracy", "gen_len"]
    payload = {}

    for metric in metric_names:
        epochs, values = _extract_eval_series(log_history, metric)
        payload[metric] = {"epochs": epochs, "values": [round(v, 6) for v in values]}

    save_json(payload, str(Path(artifact_dir) / "quality_curves_epoch.json"))

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print(
            f"matplotlib is not installed. Saved numeric quality history only: "
            f"{Path(artifact_dir) / 'quality_curves_epoch.json'}"
        )
        return

    # Plot 1: BLEU + chrF
    bleu_epochs, bleu_values = payload["bleu"]["epochs"], payload["bleu"]["values"]
    chrf_epochs, chrf_values = payload["chrf"]["epochs"], payload["chrf"]["values"]
    if bleu_epochs or chrf_epochs:
        plt.figure(figsize=(8, 5))
        if bleu_epochs:
            plt.plot(bleu_epochs, bleu_values, marker="o", label="eval_bleu")
        if chrf_epochs:
            plt.plot(chrf_epochs, chrf_values, marker="s", label="eval_chrf")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Validation Quality by Epoch (BLEU / chrF)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(Path(artifact_dir) / "quality_bleu_chrf_epoch.png"), dpi=150)
        plt.close()

    # Plot 2: terminology accuracy + gen_len
    term_epochs, term_values = payload["terminology_accuracy"]["epochs"], payload["terminology_accuracy"]["values"]
    len_epochs, len_values = payload["gen_len"]["epochs"], payload["gen_len"]["values"]
    if term_epochs or len_epochs:
        plt.figure(figsize=(8, 5))
        if term_epochs:
            plt.plot(term_epochs, term_values, marker="o", label="eval_terminology_accuracy")
        if len_epochs:
            plt.plot(len_epochs, len_values, marker="^", label="eval_gen_len")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Validation Technical Metrics by Epoch")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(Path(artifact_dir) / "quality_tech_epoch.png"), dpi=150)
        plt.close()


def _save_summary_plot(log_history, artifact_dir: str):
    ensure_dir(artifact_dir)

    epochs_loss, train_losses, eval_losses = _extract_epoch_losses(log_history)
    bleu_epochs, bleu_values = _extract_eval_series(log_history, "bleu")
    chrf_epochs, chrf_values = _extract_eval_series(log_history, "chrf")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is not installed. Skip summary plot.")
        return

    if not (epochs_loss or bleu_epochs or chrf_epochs):
        print("No logs found for summary plot.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 5.8))

    if epochs_loss:
        ax1.plot(epochs_loss, train_losses, marker="o", color="#2E86DE", label="train_loss")
        ax1.plot(epochs_loss, eval_losses, marker="s", color="#1B4F72", label="eval_loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    if bleu_epochs:
        ax2.plot(bleu_epochs, bleu_values, marker="^", color="#27AE60", label="eval_bleu")
    if chrf_epochs:
        ax2.plot(chrf_epochs, chrf_values, marker="D", color="#CA6F1E", label="eval_chrf")
    ax2.set_ylabel("Quality Score")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    if lines_1 or lines_2:
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    plt.title("Training Summary: Loss and Translation Quality by Epoch")
    plt.tight_layout()
    plt.savefig(str(Path(artifact_dir) / "summary_training_dashboard.png"), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning script for EN->RU MT baseline")
    parser.add_argument("--config", default="configs/train_config.json", help="Path to config JSON")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["seed"]))

    artifacts_dir = str(config.get("artifacts_dir", "outputs"))
    final_model_dir = Path(str(config.get("final_model_dir", "outputs/checkpoints/final")))

    if bool(config.get("force_cuda", True)):
        _assert_cuda_or_fail()

    ensure_dir("outputs")
    ensure_dir(artifacts_dir)
    ensure_dir(config["output_dir"])
    ensure_dir("outputs/checkpoints")

    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    tokenized_splits, raw_splits = load_tokenized_datasets(config, tokenizer)

    use_fp16, use_bf16 = _build_precision_flags(config)

    training_args = Seq2SeqTrainingArguments(
        output_dir=config["output_dir"],
        learning_rate=float(config["learning_rate"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["per_device_eval_batch_size"]),
        dataloader_num_workers=int(config.get("dataloader_num_workers", 0)),
        num_train_epochs=float(config["num_train_epochs"]),
        weight_decay=float(config["weight_decay"]),
        warmup_ratio=float(config["warmup_ratio"]),
        logging_steps=int(config["logging_steps"]),
        eval_strategy=config["eval_strategy"],
        save_strategy=config["save_strategy"],
        predict_with_generate=bool(config["predict_with_generate"]),
        fp16=use_fp16,
        bf16=use_bf16,
        use_cpu=False,
        seed=int(config["seed"]),
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        save_total_limit=2,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_splits["train"],
        eval_dataset=tokenized_splits["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
        **_trainer_processing_kwargs(tokenizer),
    )

    resume_from_checkpoint = config.get("resume_from_checkpoint")
    if resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=str(resume_from_checkpoint))
    else:
        trainer.train()
    main_log_history = list(trainer.state.log_history)

    stabilization_epochs = float(config.get("stabilization_epochs", 0.0))
    stabilization_lr = float(config.get("stabilization_learning_rate", float(config["learning_rate"]) * 0.5))
    full_log_history = main_log_history
    if stabilization_epochs > 0:
        print(
            f"Starting stabilization phase: epochs={stabilization_epochs}, "
            f"learning_rate={stabilization_lr:g}"
        )
        trainer = _run_stabilization_phase(
            trainer=trainer,
            tokenizer=tokenizer,
            tokenized_train=tokenized_splits["train"],
            tokenized_val=tokenized_splits["validation"],
            learning_rate=stabilization_lr,
            num_epochs=stabilization_epochs,
            seed=int(config["seed"]),
        )
        full_log_history = _merge_log_histories(
            main_history=main_log_history,
            stabilization_history=trainer.state.log_history,
            main_epochs=float(config["num_train_epochs"]),
        )

    _save_loss_plots(
        full_log_history,
        out_png=str(Path(artifacts_dir) / "loss_curve_epoch.png"),
        out_json=str(Path(artifacts_dir) / "loss_curve_epoch.json"),
    )
    _save_quality_plots(full_log_history, artifact_dir=artifacts_dir)
    _save_summary_plot(full_log_history, artifact_dir=artifacts_dir)

    ensure_dir(str(final_model_dir))
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    test_prediction = trainer.predict(tokenized_splits["test"])
    final_metrics = _extract_core_metrics(test_prediction.metrics)
    final_metrics["num_test_samples"] = len(raw_splits["test"])
    save_json(final_metrics, str(Path(artifacts_dir) / "final_metrics.json"))

    decoded_predictions = _decode_predictions(test_prediction.predictions, tokenizer)
    final_samples = []
    for idx in range(min(20, len(raw_splits["test"]))):
        final_samples.append(
            {
                "src": raw_splits["test"][idx]["source"],
                "reference": raw_splits["test"][idx]["target"],
                "prediction": decoded_predictions[idx].strip(),
            }
        )
    save_json(final_samples, str(Path(artifacts_dir) / "final_samples.json"))

    baseline_metrics_path = Path(artifacts_dir) / "baseline_metrics.json"
    if baseline_metrics_path.exists():
        baseline_metrics = load_config(str(baseline_metrics_path))
        baseline_metrics = {
            "bleu": float(baseline_metrics.get("bleu", 0.0)),
            "chrf": float(baseline_metrics.get("chrf", 0.0)),
        }
    else:
        baseline_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        baseline_pred = _predict_on_test(
            model=baseline_model,
            tokenizer=tokenizer,
            tokenized_test=tokenized_splits["test"],
            per_device_eval_batch_size=int(config["per_device_eval_batch_size"]),
            output_dir=str(Path(artifacts_dir) / "baseline_eval_tmp"),
        )
        baseline_core = _extract_core_metrics(baseline_pred.metrics)
        baseline_metrics = {
            "bleu": baseline_core["bleu"],
            "chrf": baseline_core["chrf"],
        }
        save_json(
            {
                "bleu": baseline_core["bleu"],
                "chrf": baseline_core["chrf"],
                "gen_len": baseline_core["gen_len"],
                "num_test_samples": len(raw_splits["test"]),
            },
            str(Path(artifacts_dir) / "baseline_metrics.json"),
        )

    comparison = {
        "baseline": {
            "bleu": baseline_metrics["bleu"],
            "chrf": baseline_metrics["chrf"],
        },
        "fine_tuned": {
            "bleu": final_metrics["bleu"],
            "chrf": final_metrics["chrf"],
        },
        "delta": {
            "bleu": round(final_metrics["bleu"] - baseline_metrics["bleu"], 4),
            "chrf": round(final_metrics["chrf"] - baseline_metrics["chrf"], 4),
        },
    }
    save_json(comparison, str(Path(artifacts_dir) / "comparison.json"))

    split_sizes = {
        "train": len(raw_splits["train"]),
        "validation": len(raw_splits["validation"]),
        "test": len(raw_splits["test"]),
    }
    report_text = _build_report(
        config=config,
        split_sizes=split_sizes,
        baseline_metrics=baseline_metrics,
        final_metrics=final_metrics,
    )
    Path(artifacts_dir, "report.txt").write_text(report_text, encoding="utf-8")

    print("Training completed.")
    print(f"Final BLEU: {final_metrics['bleu']:.4f}")
    print(f"Final chrF: {final_metrics['chrf']:.4f}")
    print(f"Artifacts saved to {artifacts_dir}/")


if __name__ == "__main__":
    main()
