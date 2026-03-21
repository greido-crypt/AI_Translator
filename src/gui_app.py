from __future__ import annotations

import json
import threading
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from history_manager import HistoryManager
from language_detector import LanguageDetector
from model_router import ModelRouter
from report_builder import ReportBuilder
from text_analyzer import TextAnalyzer
from translator_service import TranslatorService
from utils import ensure_dir, save_json


class TechnicalTranslatorApp(ctk.CTk):
    UI_TEXTS = {
        "en": {
            "title": "Technical MT Studio",
            "translate": "Translate",
            "copy": "Copy Result",
            "save": "Save Translation",
            "export": "Export Report",
            "source": "Source Panel",
            "translation": "Translation Panel",
            "report": "Report Panel",
            "history": "History Panel",
            "settings": "Settings Panel",
            "controls": "Control Panel",
            "target_lang": "Target Language",
            "mode": "Mode",
            "theme": "Theme",
            "ui_lang": "UI Language",
            "chunk_size": "Chunk Size",
            "placeholder": "Paste technical text here...",
            "ready": "Ready",
            "translating": "Translating...",
        },
        "ru": {
            "title": "Technical MT Studio",
            "translate": "Перевести",
            "copy": "Копировать результат",
            "save": "Сохранить перевод",
            "export": "Экспорт отчёта",
            "source": "Панель исходного текста",
            "translation": "Панель перевода",
            "report": "Панель отчёта",
            "history": "Панель истории",
            "settings": "Панель настроек",
            "controls": "Панель управления",
            "target_lang": "Целевой язык",
            "mode": "Режим",
            "theme": "Тема",
            "ui_lang": "Язык интерфейса",
            "chunk_size": "Размер чанка",
            "placeholder": "Вставьте технический текст...",
            "ready": "Готово",
            "translating": "Перевод...",
        },
    }

    def __init__(self):
        super().__init__()
        self.title("Technical MT Studio")
        self.geometry("1450x920")
        self.minsize(1200, 800)

        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("dark-blue")

        self.language_detector = LanguageDetector()
        self.text_analyzer = TextAnalyzer()
        self.model_router = ModelRouter()
        self.translator = TranslatorService()
        self.report_builder = ReportBuilder()
        self.history_manager = HistoryManager()

        self.ui_lang = ctk.StringVar(value="en")
        self.target_lang_var = ctk.StringVar(value="auto")
        self.mode_var = ctk.StringVar(value="balanced")
        self.theme_var = ctk.StringVar(value="Dark")
        self.chunk_size_var = ctk.IntVar(value=1400)

        self.last_report = {}
        self.last_translation = ""

        self._build_layout()
        self._apply_localization()
        self._refresh_history()

    def _build_layout(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.header = ctk.CTkFrame(self, corner_radius=16)
        self.header.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 8))
        self.header.grid_columnconfigure(0, weight=1)

        self.header_title = ctk.CTkLabel(
            self.header,
            text="Technical MT Studio",
            font=ctk.CTkFont(size=26, weight="bold"),
        )
        self.header_title.grid(row=0, column=0, padx=18, pady=14, sticky="w")

        self.status_label = ctk.CTkLabel(self.header, text="Ready", font=ctk.CTkFont(size=14))
        self.status_label.grid(row=0, column=1, padx=18, pady=14, sticky="e")

        self.main = ctk.CTkFrame(self, fg_color="transparent")
        self.main.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))
        self.main.grid_columnconfigure(0, weight=3)
        self.main.grid_columnconfigure(1, weight=2)
        self.main.grid_rowconfigure(0, weight=3)
        self.main.grid_rowconfigure(1, weight=2)

        self.left_top = ctk.CTkFrame(self.main, corner_radius=16)
        self.left_top.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
        self.left_top.grid_columnconfigure(0, weight=1)
        self.left_top.grid_rowconfigure(1, weight=1)

        self.source_label = ctk.CTkLabel(self.left_top, text="Source Panel", font=ctk.CTkFont(size=16, weight="bold"))
        self.source_label.grid(row=0, column=0, sticky="w", padx=14, pady=(10, 6))

        self.source_text = ctk.CTkTextbox(self.left_top, corner_radius=12, font=ctk.CTkFont(family="Consolas", size=13))
        self.source_text.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))

        self.right_top = ctk.CTkFrame(self.main, corner_radius=16)
        self.right_top.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=(0, 8))
        self.right_top.grid_columnconfigure(0, weight=1)
        self.right_top.grid_rowconfigure(1, weight=1)

        self.translation_label = ctk.CTkLabel(
            self.right_top, text="Translation Panel", font=ctk.CTkFont(size=16, weight="bold")
        )
        self.translation_label.grid(row=0, column=0, sticky="w", padx=14, pady=(10, 6))

        self.translation_text = ctk.CTkTextbox(self.right_top, corner_radius=12, font=ctk.CTkFont(family="Consolas", size=13))
        self.translation_text.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))

        self.left_bottom = ctk.CTkFrame(self.main, corner_radius=16)
        self.left_bottom.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(8, 0))
        self.left_bottom.grid_columnconfigure(0, weight=1)
        self.left_bottom.grid_rowconfigure(1, weight=1)

        self.report_label = ctk.CTkLabel(self.left_bottom, text="Report Panel", font=ctk.CTkFont(size=16, weight="bold"))
        self.report_label.grid(row=0, column=0, sticky="w", padx=14, pady=(10, 6))

        self.report_text = ctk.CTkTextbox(self.left_bottom, corner_radius=12, font=ctk.CTkFont(family="Consolas", size=12))
        self.report_text.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))

        self.right_bottom = ctk.CTkFrame(self.main, corner_radius=16)
        self.right_bottom.grid(row=1, column=1, sticky="nsew", padx=(8, 0), pady=(8, 0))
        self.right_bottom.grid_columnconfigure(0, weight=1)
        self.right_bottom.grid_rowconfigure(1, weight=1)
        self.right_bottom.grid_rowconfigure(3, weight=1)

        self.controls_label = ctk.CTkLabel(self.right_bottom, text="Control Panel", font=ctk.CTkFont(size=16, weight="bold"))
        self.controls_label.grid(row=0, column=0, sticky="w", padx=14, pady=(10, 6))

        self.controls_frame = ctk.CTkFrame(self.right_bottom, corner_radius=12)
        self.controls_frame.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 10))
        self.controls_frame.grid_columnconfigure((0, 1), weight=1)

        self.target_lang_label = ctk.CTkLabel(self.controls_frame, text="Target Language")
        self.target_lang_label.grid(row=0, column=0, padx=8, pady=(8, 4), sticky="w")
        self.target_lang_menu = ctk.CTkOptionMenu(self.controls_frame, values=["auto", "ru", "en"], variable=self.target_lang_var)
        self.target_lang_menu.grid(row=1, column=0, padx=8, pady=(0, 8), sticky="ew")

        self.mode_label = ctk.CTkLabel(self.controls_frame, text="Mode")
        self.mode_label.grid(row=0, column=1, padx=8, pady=(8, 4), sticky="w")
        self.mode_menu = ctk.CTkOptionMenu(
            self.controls_frame,
            values=["fast", "balanced", "high_quality"],
            variable=self.mode_var,
        )
        self.mode_menu.grid(row=1, column=1, padx=8, pady=(0, 8), sticky="ew")

        self.translate_button = ctk.CTkButton(self.controls_frame, text="Translate", command=self._start_translate)
        self.translate_button.grid(row=2, column=0, padx=8, pady=8, sticky="ew")

        self.copy_button = ctk.CTkButton(self.controls_frame, text="Copy Result", command=self._copy_result)
        self.copy_button.grid(row=2, column=1, padx=8, pady=8, sticky="ew")

        self.save_button = ctk.CTkButton(self.controls_frame, text="Save Translation", command=self._save_translation)
        self.save_button.grid(row=3, column=0, padx=8, pady=(0, 8), sticky="ew")

        self.export_button = ctk.CTkButton(self.controls_frame, text="Export Report", command=self._export_report)
        self.export_button.grid(row=3, column=1, padx=8, pady=(0, 8), sticky="ew")

        self.history_label = ctk.CTkLabel(self.right_bottom, text="History Panel", font=ctk.CTkFont(size=16, weight="bold"))
        self.history_label.grid(row=2, column=0, sticky="w", padx=14, pady=(0, 6))

        self.history_text = ctk.CTkTextbox(self.right_bottom, corner_radius=12, font=ctk.CTkFont(size=12), height=150)
        self.history_text.grid(row=3, column=0, sticky="nsew", padx=14, pady=(0, 10))

        self.settings_label = ctk.CTkLabel(self.right_bottom, text="Settings Panel", font=ctk.CTkFont(size=16, weight="bold"))
        self.settings_label.grid(row=4, column=0, sticky="w", padx=14, pady=(0, 6))

        self.settings_frame = ctk.CTkFrame(self.right_bottom, corner_radius=12)
        self.settings_frame.grid(row=5, column=0, sticky="ew", padx=14, pady=(0, 14))
        self.settings_frame.grid_columnconfigure((0, 1), weight=1)

        self.theme_label = ctk.CTkLabel(self.settings_frame, text="Theme")
        self.theme_label.grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))
        self.theme_menu = ctk.CTkOptionMenu(
            self.settings_frame,
            values=["Dark", "Light"],
            variable=self.theme_var,
            command=self._on_theme_changed,
        )
        self.theme_menu.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))

        self.ui_lang_label = ctk.CTkLabel(self.settings_frame, text="UI Language")
        self.ui_lang_label.grid(row=0, column=1, sticky="w", padx=8, pady=(8, 4))
        self.ui_lang_menu = ctk.CTkOptionMenu(
            self.settings_frame,
            values=["en", "ru"],
            variable=self.ui_lang,
            command=lambda _: self._apply_localization(),
        )
        self.ui_lang_menu.grid(row=1, column=1, sticky="ew", padx=8, pady=(0, 8))

        self.chunk_label = ctk.CTkLabel(self.settings_frame, text="Chunk Size")
        self.chunk_label.grid(row=2, column=0, sticky="w", padx=8, pady=(0, 4))
        self.chunk_slider = ctk.CTkSlider(
            self.settings_frame,
            from_=700,
            to=3000,
            number_of_steps=23,
            variable=self.chunk_size_var,
        )
        self.chunk_slider.grid(row=3, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8))

        self.source_text.insert("1.0", self._t("placeholder"))

    def _t(self, key: str) -> str:
        return self.UI_TEXTS[self.ui_lang.get()].get(key, key)

    def _apply_localization(self):
        self.header_title.configure(text=self._t("title"))
        self.source_label.configure(text=self._t("source"))
        self.translation_label.configure(text=self._t("translation"))
        self.report_label.configure(text=self._t("report"))
        self.history_label.configure(text=self._t("history"))
        self.settings_label.configure(text=self._t("settings"))
        self.controls_label.configure(text=self._t("controls"))
        self.target_lang_label.configure(text=self._t("target_lang"))
        self.mode_label.configure(text=self._t("mode"))
        self.theme_label.configure(text=self._t("theme"))
        self.ui_lang_label.configure(text=self._t("ui_lang"))
        self.chunk_label.configure(text=self._t("chunk_size"))
        self.translate_button.configure(text=self._t("translate"))
        self.copy_button.configure(text=self._t("copy"))
        self.save_button.configure(text=self._t("save"))
        self.export_button.configure(text=self._t("export"))
        self.status_label.configure(text=self._t("ready"))

    def _on_theme_changed(self, value: str):
        ctk.set_appearance_mode(value)

    def _set_busy(self, busy: bool):
        self.translate_button.configure(state="disabled" if busy else "normal")
        self.status_label.configure(text=self._t("translating") if busy else self._t("ready"))

    def _start_translate(self):
        source = self.source_text.get("1.0", "end").strip()
        if not source:
            messagebox.showwarning("Warning", "Source text is empty.")
            return

        self._set_busy(True)
        threading.Thread(target=self._translate_worker, args=(source,), daemon=True).start()

    def _translate_worker(self, source: str):
        try:
            detection = self.language_detector.detect(source)
            analysis = self.text_analyzer.analyze(source)
            selection = self.model_router.route(
                detection=detection,
                analysis=analysis,
                ui_target_lang=self.target_lang_var.get(),
                quality_mode=self.mode_var.get(),
            )
            runtime = self.translator.translate(
                text=source,
                selection=selection,
                chunk_size_chars=int(self.chunk_size_var.get()),
            )
            report = self.report_builder.build(
                source_text=source,
                translated_text=runtime.translated_text,
                detection=detection,
                selection=selection,
                analysis=analysis,
                runtime=runtime,
            )
            report_text = self.report_builder.to_pretty_text(report)

            self.last_report = report
            self.last_translation = runtime.translated_text
            self.history_manager.add(source, runtime.translated_text, report)

            self.after(0, lambda: self._apply_results(runtime.translated_text, report_text))
            self.after(0, self._refresh_history)
        except Exception as exc:
            self.after(0, lambda: messagebox.showerror("Translation Error", str(exc)))
        finally:
            self.after(0, lambda: self._set_busy(False))

    def _apply_results(self, translation: str, report_text: str):
        self.translation_text.delete("1.0", "end")
        self.translation_text.insert("1.0", translation)

        self.report_text.delete("1.0", "end")
        self.report_text.insert("1.0", report_text)

    def _copy_result(self):
        text = self.translation_text.get("1.0", "end").strip()
        if not text:
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.status_label.configure(text="Copied")

    def _save_translation(self):
        text = self.translation_text.get("1.0", "end").strip()
        if not text:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("Markdown", "*.md"), ("All files", "*.*")],
        )
        if not path:
            return
        Path(path).write_text(text, encoding="utf-8")

    def _export_report(self):
        if not self.last_report:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        if path.endswith(".txt"):
            Path(path).write_text(self.report_builder.to_pretty_text(self.last_report), encoding="utf-8")
        else:
            save_json(self.last_report, path)

    def _refresh_history(self):
        items = self.history_manager.load()
        self.history_text.delete("1.0", "end")
        lines = []
        for item in items[:20]:
            lines.append(
                f"[{item['timestamp']}] {item['detected_language']}->{item['target_language']} | "
                f"{item['model']} | {item['inference_mode']}\n"
                f"src: {item['source_preview']}\n"
                f"out: {item['translation_preview']}\n"
                "-" * 60
            )
        self.history_text.insert("1.0", "\n".join(lines))


def main():
    ensure_dir("outputs")
    app = TechnicalTranslatorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
