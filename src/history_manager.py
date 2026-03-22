from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

from utils import load_config, save_json


class HistoryManager:
    def __init__(self, history_path: str = "outputs/translation_history.json", max_items: int = 100) -> None:
        self.history_path = Path(history_path)
        self.max_items = max_items

    def load(self) -> List[Dict]:
        if not self.history_path.exists():
            return []
        try:
            return load_config(str(self.history_path))
        except Exception:
            return []

    def add(self, source_text: str, translated_text: str, report: Dict) -> List[Dict]:
        history = self.load()
        item = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "source_preview": source_text[:200],
            "translation_preview": translated_text[:200],
            "detected_language": report.get("detected_language"),
            "target_language": report.get("target_language"),
            "model": report.get("selected_model"),
            "inference_mode": report.get("inference_mode"),
            "metrics": {
                "terminology": report.get("terminology_preservation_score"),
                "code": report.get("code_preservation_score"),
                "formatting": report.get("formatting_preservation_score"),
            },
        }
        history.insert(0, item)
        history = history[: self.max_items]
        save_json(history, str(self.history_path))
        return history

    def clear(self) -> None:
        if self.history_path.exists():
            self.history_path.unlink()
