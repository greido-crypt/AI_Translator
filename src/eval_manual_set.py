from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import sacrebleu

from language_detector import LanguageDetector
from model_router import ModelSelection
from report_builder import ReportBuilder
from text_analyzer import TextAnalyzer
from translator_service import TranslatorService
from utils import ensure_dir, load_config, save_json


TERM_PAIRS_EN_RU = {
    "frontend": "фронтенд",
    "backend": "бэкенд",
    "logging module": "модуль логирования",
    "status code": "код состояния",
    "cache": "кэш",
    "authentication failures": "ошибки аутентификации",
    "timeout errors": "ошибки тайм-аута",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual stress evaluation for EN->RU technical MT.")
    parser.add_argument("--config_path", default="configs/train_config.json")
    parser.add_argument("--cases_path", default="configs/manual_test_cases.json")
    parser.add_argument("--baseline_model", default="Helsinki-NLP/opus-mt-en-ru")
    parser.add_argument("--fine_tuned_model", default="./outputs/checkpoints/final")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--chunk_size", type=int, default=1200)
    return parser.parse_args()


def _run_model(
    model_name: str,
    cases: List[Dict[str, str]],
    service: TranslatorService,
    detector: LanguageDetector,
    analyzer: TextAnalyzer,
    report_builder: ReportBuilder,
    chunk_size: int,
) -> Dict:
    predictions: List[str] = []
    references = [item["reference"] for item in cases]
    sample_reports = []
    per_sample = []

    total_inference_time = 0.0
    total_chunks = 0
    total_protected = 0
    total_restored = 0
    terminology_scores = []
    code_scores = []
    formatting_scores = []

    for item in cases:
        src = item["src"]
        ref = item["reference"]

        detection = detector.detect(src)
        analysis = analyzer.analyze(src)
        selection = ModelSelection(
            source_lang="en",
            target_lang="ru",
            model_name=model_name,
            reason="Manual stress-test execution",
            inference_mode="balanced",
        )
        runtime = service.translate(src, selection=selection, chunk_size_chars=chunk_size)
        report = report_builder.build(
            source_text=src,
            translated_text=runtime.translated_text,
            detection=detection,
            selection=selection,
            analysis=analysis,
            runtime=runtime,
        )

        prediction = runtime.translated_text.strip()
        predictions.append(prediction)
        sample_reports.append(report)

        total_inference_time += runtime.total_inference_time
        total_chunks += runtime.chunk_count
        total_protected += len(runtime.protected_tokens)
        total_restored += runtime.protected_token_hits
        terminology_scores.append(report["terminology_preservation_score"])
        code_scores.append(report["code_preservation_score"])
        formatting_scores.append(report["formatting_preservation_score"])

        missed_terms = []
        src_lower = src.lower()
        pred_lower = prediction.lower()
        for en_term, ru_term in TERM_PAIRS_EN_RU.items():
            if en_term in src_lower and ru_term not in pred_lower:
                missed_terms.append({"en": en_term, "expected_ru": ru_term})

        per_sample.append(
            {
                "src": src,
                "reference": ref,
                "prediction": prediction,
                "report": report,
                "missed_terms": missed_terms,
            }
        )

    bleu = sacrebleu.corpus_bleu(predictions, [references]).score
    chrf = sacrebleu.corpus_chrf(predictions, [references]).score

    return {
        "model_name": model_name,
        "metrics": {
            "bleu": round(float(bleu), 4),
            "chrf": round(float(chrf), 4),
            "avg_terminology_preservation_score": round(float(statistics.mean(terminology_scores)), 4),
            "avg_code_preservation_score": round(float(statistics.mean(code_scores)), 4),
            "avg_formatting_preservation_score": round(float(statistics.mean(formatting_scores)), 4),
            "avg_chunk_count": round(total_chunks / max(1, len(cases)), 4),
            "avg_inference_time_sec": round(total_inference_time / max(1, len(cases)), 4),
            "technical_token_restoration_rate": round(total_restored / max(1, total_protected), 4),
            "num_cases": len(cases),
        },
        "samples": per_sample,
        "sample_reports": sample_reports,
    }


def _build_summary_text(result_baseline: Dict, result_fine: Dict, cases_count: int) -> str:
    b = result_baseline["metrics"]
    f = result_fine["metrics"]
    lines = [
        "Manual Stress-Test Report",
        f"Cases: {cases_count}",
        "",
        "Baseline:",
        f"  model: {result_baseline['model_name']}",
        f"  BLEU: {b['bleu']}",
        f"  chrF: {b['chrf']}",
        f"  terminology_score: {b['avg_terminology_preservation_score']}",
        f"  code_score: {b['avg_code_preservation_score']}",
        f"  formatting_score: {b['avg_formatting_preservation_score']}",
        "",
        "Fine-tuned:",
        f"  model: {result_fine['model_name']}",
        f"  BLEU: {f['bleu']}",
        f"  chrF: {f['chrf']}",
        f"  terminology_score: {f['avg_terminology_preservation_score']}",
        f"  code_score: {f['avg_code_preservation_score']}",
        f"  formatting_score: {f['avg_formatting_preservation_score']}",
        "",
        "Delta (fine-tuned - baseline):",
        f"  BLEU: {round(f['bleu'] - b['bleu'], 4)}",
        f"  chrF: {round(f['chrf'] - b['chrf'], 4)}",
        f"  terminology_score: {round(f['avg_terminology_preservation_score'] - b['avg_terminology_preservation_score'], 4)}",
        f"  code_score: {round(f['avg_code_preservation_score'] - b['avg_code_preservation_score'], 4)}",
        f"  formatting_score: {round(f['avg_formatting_preservation_score'] - b['avg_formatting_preservation_score'], 4)}",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = load_config(args.config_path)
    with Path(args.cases_path).open("r", encoding="utf-8-sig") as f:
        cases = json.load(f)

    if not isinstance(cases, list) or len(cases) < 5:
        raise ValueError("Manual test set is missing or too small. Need at least 5 cases.")
    for item in cases:
        if "src" not in item or "reference" not in item:
            raise ValueError("Each manual case must contain 'src' and 'reference'.")

    ensure_dir(args.output_dir)
    detector = LanguageDetector()
    analyzer = TextAnalyzer()
    report_builder = ReportBuilder()
    service = TranslatorService()

    baseline_result = _run_model(
        model_name=args.baseline_model,
        cases=cases,
        service=service,
        detector=detector,
        analyzer=analyzer,
        report_builder=report_builder,
        chunk_size=args.chunk_size,
    )

    fine_tuned_model = args.fine_tuned_model
    if not Path(fine_tuned_model).exists():
        fine_tuned_model = config.get("model_name", "Helsinki-NLP/opus-mt-en-ru")

    fine_result = _run_model(
        model_name=fine_tuned_model,
        cases=cases,
        service=service,
        detector=detector,
        analyzer=analyzer,
        report_builder=report_builder,
        chunk_size=args.chunk_size,
    )

    comparison = {
        "baseline": baseline_result["metrics"],
        "fine_tuned": fine_result["metrics"],
        "delta": {
            "bleu": round(fine_result["metrics"]["bleu"] - baseline_result["metrics"]["bleu"], 4),
            "chrf": round(fine_result["metrics"]["chrf"] - baseline_result["metrics"]["chrf"], 4),
            "avg_terminology_preservation_score": round(
                fine_result["metrics"]["avg_terminology_preservation_score"]
                - baseline_result["metrics"]["avg_terminology_preservation_score"],
                4,
            ),
            "avg_code_preservation_score": round(
                fine_result["metrics"]["avg_code_preservation_score"]
                - baseline_result["metrics"]["avg_code_preservation_score"],
                4,
            ),
            "avg_formatting_preservation_score": round(
                fine_result["metrics"]["avg_formatting_preservation_score"]
                - baseline_result["metrics"]["avg_formatting_preservation_score"],
                4,
            ),
        },
    }

    save_json(baseline_result["metrics"], Path(args.output_dir) / "manual_baseline_metrics.json")
    save_json(fine_result["metrics"], Path(args.output_dir) / "manual_fine_tuned_metrics.json")
    save_json(comparison, Path(args.output_dir) / "manual_comparison.json")
    save_json(baseline_result["samples"], Path(args.output_dir) / "manual_baseline_samples.json")
    save_json(fine_result["samples"], Path(args.output_dir) / "manual_fine_tuned_samples.json")

    summary_text = _build_summary_text(baseline_result, fine_result, len(cases))
    (Path(args.output_dir) / "manual_report.txt").write_text(summary_text, encoding="utf-8")

    print("Manual stress-test finished.")
    print(summary_text)


if __name__ == "__main__":
    main()
