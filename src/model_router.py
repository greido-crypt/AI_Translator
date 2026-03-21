from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from language_detector import LanguageDetectionResult
from text_analyzer import TextAnalysis


@dataclass
class ModelSelection:
    source_lang: str
    target_lang: str
    model_name: str
    reason: str
    inference_mode: str


class ModelRouter:
    def __init__(
        self,
        preferred_en_ru_checkpoint: str = "./outputs/checkpoints/final",
        fallback_en_ru_checkpoint: str = "./outputs/checkpoints_synth8000/final",
    ):
        self.preferred_en_ru_checkpoint = preferred_en_ru_checkpoint
        self.fallback_en_ru_checkpoint = fallback_en_ru_checkpoint

    def route(
        self,
        detection: LanguageDetectionResult,
        analysis: TextAnalysis,
        ui_target_lang: str,
        quality_mode: str,
    ) -> ModelSelection:
        source_lang = detection.language if detection.language in {"en", "ru"} else "en"
        target_lang = self._resolve_target_lang(source_lang, ui_target_lang)
        inference_mode = self._resolve_inference_mode(quality_mode, analysis)

        if source_lang == "en" and target_lang == "ru":
            model_name, reason = self._choose_en_ru_model(analysis, quality_mode)
        elif source_lang == "ru" and target_lang == "en":
            model_name = "Helsinki-NLP/opus-mt-ru-en"
            reason = "Detected Russian source text; selected dedicated ru->en Marian model."
        elif source_lang == target_lang:
            model_name = "identity"
            reason = "Source and target languages match; translation bypass mode selected."
        else:
            model_name = "Helsinki-NLP/opus-mt-en-ru"
            reason = "Fallback to baseline technical translator due to unsupported direction."

        return ModelSelection(
            source_lang=source_lang,
            target_lang=target_lang,
            model_name=model_name,
            reason=reason,
            inference_mode=inference_mode,
        )

    def _resolve_target_lang(self, source_lang: str, ui_target_lang: str) -> str:
        if ui_target_lang != "auto":
            return ui_target_lang
        return "ru" if source_lang == "en" else "en"

    def _choose_en_ru_model(self, analysis: TextAnalysis, quality_mode: str) -> tuple[str, str]:
        preferred = Path(self.preferred_en_ru_checkpoint)
        fallback = Path(self.fallback_en_ru_checkpoint)
        extra_candidates = [
            Path("./outputs/checkpoints_highacc/final"),
        ]

        candidates = [preferred, fallback] + extra_candidates
        valid = []
        for path in candidates:
            if path.exists() and (path / "config.json").exists():
                valid.append(path)

        if valid:
            # Pick the most recently updated checkpoint to avoid stale GUI routing.
            best = max(valid, key=lambda p: p.stat().st_mtime)
            return (
                str(best),
                f"Local EN->RU checkpoint selected: {best}",
            )

        return (
            "Helsinki-NLP/opus-mt-en-ru",
            "Selected baseline EN->RU Marian model for stable technical translation quality.",
        )

    def _resolve_inference_mode(self, requested_mode: str, analysis: TextAnalysis) -> str:
        if requested_mode != "balanced":
            return requested_mode
        if analysis.char_count >= 1200 and (
            analysis.has_code_blocks or analysis.has_json_yaml_xml or analysis.has_cli_commands
        ):
            return "high_quality"
        return requested_mode

    def available_ui_languages(self) -> Dict[str, str]:
        return {"en": "English", "ru": "Русский"}
