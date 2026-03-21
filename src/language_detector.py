from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

try:
    from langdetect import DetectorFactory, LangDetectException, detect_langs

    DetectorFactory.seed = 42
except Exception:  # pragma: no cover
    detect_langs = None
    LangDetectException = Exception


@dataclass
class LanguageDetectionResult:
    language: str
    confidence: float
    candidates: List[Dict[str, float]]
    method: str


class LanguageDetector:
    CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
    LATIN_RE = re.compile(r"[A-Za-z]")

    def detect(self, text: str) -> LanguageDetectionResult:
        text = (text or "").strip()
        if not text:
            return LanguageDetectionResult("unknown", 0.0, [], "empty")

        heuristic = self._heuristic_detect(text)
        if heuristic:
            return heuristic

        if detect_langs is None:
            return LanguageDetectionResult("en", 0.5, [{"lang": "en", "prob": 0.5}], "fallback")

        try:
            langs = detect_langs(text)
            candidates = [{"lang": item.lang, "prob": float(item.prob)} for item in langs]
            best = candidates[0]
            return LanguageDetectionResult(best["lang"], best["prob"], candidates, "langdetect")
        except LangDetectException:
            return LanguageDetectionResult("en", 0.5, [{"lang": "en", "prob": 0.5}], "fallback")

    def _heuristic_detect(self, text: str) -> LanguageDetectionResult | None:
        cyr = len(self.CYRILLIC_RE.findall(text))
        lat = len(self.LATIN_RE.findall(text))
        total = cyr + lat
        if total < 10:
            return None

        if cyr / total > 0.7:
            return LanguageDetectionResult("ru", cyr / total, [{"lang": "ru", "prob": cyr / total}], "heuristic")
        if lat / total > 0.7:
            return LanguageDetectionResult("en", lat / total, [{"lang": "en", "prob": lat / total}], "heuristic")
        return None
