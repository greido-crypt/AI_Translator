from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List

from language_detector import LanguageDetectionResult
from model_router import ModelSelection
from text_analyzer import TextAnalysis
from translator_service import TranslationRuntimeResult


class ReportBuilder:
    def build(
        self,
        source_text: str,
        translated_text: str,
        detection: LanguageDetectionResult,
        selection: ModelSelection,
        analysis: TextAnalysis,
        runtime: TranslationRuntimeResult,
    ) -> Dict:
        terminology_score = self._terminology_score(source_text, translated_text, runtime.protected_tokens)
        code_score = self._code_preservation_score(source_text, translated_text)
        formatting_score = self._formatting_score(source_text, translated_text)

        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "detected_language": detection.language,
            "target_language": selection.target_lang,
            "selected_model": selection.model_name,
            "reason_for_model_selection": selection.reason,
            "inference_mode": selection.inference_mode,
            "preprocessing_summary": runtime.preprocessing_summary,
            "chunk_count": runtime.chunk_count,
            "total_inference_time": round(runtime.total_inference_time, 3),
            "technical_token_preservation_summary": {
                "protected": len(runtime.protected_tokens),
                "restored": runtime.protected_token_hits,
            },
            "terminology_preservation_score": round(terminology_score, 3),
            "code_preservation_score": round(code_score, 3),
            "formatting_preservation_score": round(formatting_score, 3),
            "warnings": runtime.warnings,
            "device_info": runtime.device_info,
            "model_info": runtime.model_info,
            "analysis": asdict(analysis),
            "detection_candidates": detection.candidates,
        }

    def to_pretty_text(self, report: Dict) -> str:
        lines = [
            f"detected language: {report['detected_language']}",
            f"target language: {report['target_language']}",
            f"selected model: {report['selected_model']}",
            f"reason for model selection: {report['reason_for_model_selection']}",
            f"inference mode: {report['inference_mode']}",
            f"preprocessing summary: {json.dumps(report['preprocessing_summary'], ensure_ascii=False)}",
            f"chunk count: {report['chunk_count']}",
            f"total inference time: {report['total_inference_time']} sec",
            (
                "technical token preservation summary: "
                f"{json.dumps(report['technical_token_preservation_summary'], ensure_ascii=False)}"
            ),
            f"terminology preservation score: {report['terminology_preservation_score']}",
            f"code preservation score: {report['code_preservation_score']}",
            f"formatting preservation score: {report['formatting_preservation_score']}",
            f"warnings: {report['warnings']}",
            f"device info: {report['device_info']}",
            f"model info: {json.dumps(report['model_info'], ensure_ascii=False)}",
        ]
        return "\n".join(lines)

    def _terminology_score(self, source: str, translated: str, protected_tokens: List[str]) -> float:
        if not protected_tokens:
            return 1.0
        kept = sum(1 for token in protected_tokens if token in translated)
        return kept / max(1, len(protected_tokens))

    def _code_preservation_score(self, source: str, translated: str) -> float:
        markers = ["```", "`", "--", "/", "\\", "{", "}"]
        src_count = sum(source.count(m) for m in markers)
        if src_count == 0:
            return 1.0
        dst_count = sum(translated.count(m) for m in markers)
        return min(src_count, dst_count) / max(src_count, 1)

    def _formatting_score(self, source: str, translated: str) -> float:
        markers = ["\n", "#", "*", "- ", "1. ", "> "]
        ratios = []
        for marker in markers:
            s = source.count(marker)
            t = translated.count(marker)
            if s == 0:
                continue
            ratios.append(min(s, t) / s)
        return sum(ratios) / len(ratios) if ratios else 1.0
