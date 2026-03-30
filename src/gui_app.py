from __future__ import annotations

import json
import os
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
            "clear_history": "Clear History",
            "settings": "Settings Panel",
            "controls": "Control Panel",
            "target_lang": "Target Language",
            "mode": "Mode",
            "theme": "Theme",
            "ui_lang": "UI Language",
            "chunk_size": "Chunk Size",
            "chunk_hint": "700-1200: code-heavy text | 1400: default | 1600-2000: long prose",
            "model_select": "Model Selection",
            "model_path": "Local Model Path",
            "browse": "Browse...",
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
            "clear_history": "Удалить историю",
            "settings": "Панель настроек",
            "controls": "Панель управления",
            "target_lang": "Целевой язык",
            "mode": "Режим",
            "theme": "Тема",
            "ui_lang": "Язык интерфейса",
            "chunk_size": "Размер чанка",
            "chunk_hint": "700-1200: много кода | 1400: по умолчанию | 1600-2000: длинная проза",
            "model_select": "Выбор модели",
            "model_path": "Путь к локальной модели",
            "browse": "Обзор...",
            "placeholder": "Вставьте технический текст...",
            "ready": "Готово",
            "translating": "Перевод...",
        },
    }

    def __init__(self):
        super().__init__()
        self.project_root = Path(__file__).resolve().parent.parent
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
        self.model_choice_var = ctk.StringVar(value="auto")
        self.manual_model_path_var = ctk.StringVar(value="./outputs/checkpoints/final")

        self.last_report = {}
        self.last_translation = ""
        self._status_anim_job = None
        self._status_anim_step = 0
        self._typing_anim_job = None
        self._typing_target_text = ""
        self._typing_index = 0
        self._report_flash_job = None
        self._report_flash_step = 0
        self._translation_flash_job = None
        self._translation_flash_step = 0

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

        self.progress_bar = ctk.CTkProgressBar(self.header, mode="indeterminate", height=6)
        self.progress_bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=18, pady=(0, 12))
        self.progress_bar.grid_remove()

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
        self.right_bottom.grid_rowconfigure(1, weight=0)
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
        self.translate_button.grid(row=2, column=0, columnspan=2, padx=8, pady=(8, 6), sticky="ew")

        self.copy_button = ctk.CTkButton(self.controls_frame, text="Copy Result", command=self._copy_result)
        self.copy_button.grid(row=3, column=0, columnspan=2, padx=8, pady=(0, 6), sticky="ew")

        self.save_button = ctk.CTkButton(self.controls_frame, text="Save Translation", command=self._save_translation)
        self.save_button.grid(row=4, column=0, columnspan=2, padx=8, pady=(0, 6), sticky="ew")

        self.export_button = ctk.CTkButton(self.controls_frame, text="Export Report", command=self._export_report)
        self.export_button.grid(row=5, column=0, columnspan=2, padx=8, pady=(0, 8), sticky="ew")

        self.history_label = ctk.CTkLabel(self.right_bottom, text="History Panel", font=ctk.CTkFont(size=16, weight="bold"))
        self.history_label.grid(row=2, column=0, sticky="w", padx=14, pady=(0, 6))

        self.clear_history_button = ctk.CTkButton(
            self.right_bottom,
            text="Clear History",
            command=self._clear_history,
            width=130,
        )
        self.clear_history_button.grid(row=2, column=0, sticky="e", padx=14, pady=(0, 6))

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
        self.chunk_value_label = ctk.CTkLabel(self.settings_frame, text="1400")
        self.chunk_value_label.grid(row=2, column=1, sticky="e", padx=8, pady=(0, 4))
        self.chunk_slider = ctk.CTkSlider(
            self.settings_frame,
            from_=700,
            to=3000,
            number_of_steps=23,
            variable=self.chunk_size_var,
            command=self._on_chunk_size_changed,
        )
        self.chunk_slider.grid(row=3, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8))

        self.chunk_hint_label = ctk.CTkLabel(
            self.settings_frame,
            text="700-1200: code-heavy text | 1400: default | 1600-2000: long prose",
            font=ctk.CTkFont(size=11),
            text_color=("gray35", "gray70"),
            wraplength=320,
            justify="left",
        )
        self.chunk_hint_label.grid(row=4, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 8))

        self.model_select_label = ctk.CTkLabel(self.settings_frame, text="Model Selection")
        self.model_select_label.grid(row=5, column=0, sticky="w", padx=8, pady=(0, 4))
        self.model_select_menu = ctk.CTkOptionMenu(
            self.settings_frame,
            values=["auto", "baseline", "local_path"],
            variable=self.model_choice_var,
            command=self._on_model_choice_changed,
        )
        self.model_select_menu.grid(row=5, column=1, sticky="ew", padx=8, pady=(0, 4))

        self.model_path_label = ctk.CTkLabel(self.settings_frame, text="Local Model Path")
        self.model_path_label.grid(row=6, column=0, sticky="w", padx=8, pady=(0, 4))
        self.model_path_entry = ctk.CTkEntry(
            self.settings_frame,
            textvariable=self.manual_model_path_var,
            placeholder_text="./outputs/checkpoints/final",
        )
        self.model_path_entry.grid(row=6, column=1, sticky="ew", padx=8, pady=(0, 4))

        self.model_browse_button = ctk.CTkButton(
            self.settings_frame,
            text="Browse...",
            command=self._browse_model_path,
            width=110,
        )
        self.model_browse_button.grid(row=7, column=1, sticky="e", padx=8, pady=(0, 8))

        self._on_model_choice_changed(self.model_choice_var.get())

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
        self.chunk_hint_label.configure(text=self._t("chunk_hint"))
        self.model_select_label.configure(text=self._t("model_select"))
        self.model_path_label.configure(text=self._t("model_path"))
        self.model_browse_button.configure(text=self._t("browse"))
        self.translate_button.configure(text=self._t("translate"))
        self.copy_button.configure(text=self._t("copy"))
        self.save_button.configure(text=self._t("save"))
        self.export_button.configure(text=self._t("export"))
        self.clear_history_button.configure(text=self._t("clear_history"))
        self.status_label.configure(text=self._t("ready"))
        self.chunk_value_label.configure(text=str(int(self.chunk_size_var.get())))

    def _on_theme_changed(self, value: str):
        ctk.set_appearance_mode(value)

    def _on_chunk_size_changed(self, value: float):
        self.chunk_value_label.configure(text=str(int(value)))

    def _on_model_choice_changed(self, value: str):
        if value == "local_path":
            self.model_path_entry.configure(state="normal")
            self.model_browse_button.configure(state="normal")
        else:
            self.model_path_entry.configure(state="disabled")
            self.model_browse_button.configure(state="disabled")

    def _browse_model_path(self):
        path = filedialog.askdirectory(initialdir=str(self.project_root))
        if not path:
            return
        selected = Path(path)
        try:
            rel = selected.resolve().relative_to(self.project_root.resolve())
            normalized = rel.as_posix()
        except Exception:
            normalized = str(selected)
        self.manual_model_path_var.set(normalized)

    def _set_busy(self, busy: bool):
        self.translate_button.configure(state="disabled" if busy else "normal")
        if busy:
            self.progress_bar.grid()
            self.progress_bar.start()
            self._start_status_animation()
        else:
            self.progress_bar.stop()
            self.progress_bar.grid_remove()
            self._stop_status_animation()
            self.status_label.configure(text=self._t("ready"))

    def _start_status_animation(self):
        self._status_anim_step = 0
        self._animate_status()

    def _stop_status_animation(self):
        if self._status_anim_job is not None:
            self.after_cancel(self._status_anim_job)
            self._status_anim_job = None

    def _animate_status(self):
        dots = "." * (self._status_anim_step % 4)
        self.status_label.configure(text=f"{self._t('translating')}{dots}")
        self._status_anim_step += 1
        self._status_anim_job = self.after(280, self._animate_status)

    def _start_translate(self):
        source = self.source_text.get("1.0", "end").strip()
        if not source:
            messagebox.showwarning("Warning", "Source text is empty.")
            return
        if self.model_choice_var.get() == "local_path":
            model_path = self.manual_model_path_var.get().strip()
            if not model_path:
                messagebox.showwarning("Warning", "Local model path is empty.")
                return
            missing = self._missing_model_files(model_path)
            if missing:
                joined = ", ".join(missing)
                messagebox.showerror(
                    "Model Path Error",
                    f"Selected model directory is incomplete.\nMissing files: {joined}",
                )
                return

        self._set_busy(True)
        threading.Thread(target=self._translate_worker, args=(source,), daemon=True).start()

    def _missing_model_files(self, model_path: str):
        required = [
            "config.json",
            "tokenizer_config.json",
            "vocab.json",
            "source.spm",
            "target.spm",
        ]
        p = Path(model_path)
        if not p.is_absolute():
            p = (self.project_root / p).resolve()
        if not p.exists() or not p.is_dir():
            return required
        missing = [name for name in required if not (p / name).exists()]
        if not ((p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()):
            missing.append("model.safetensors|pytorch_model.bin")
        return missing

    def _translate_worker(self, source: str):
        try:
            detection = self.language_detector.detect(source)
            analysis = self.text_analyzer.analyze(source)
            selection = self.model_router.route(
                detection=detection,
                analysis=analysis,
                ui_target_lang=self.target_lang_var.get(),
                quality_mode=self.mode_var.get(),
                manual_model_mode=self.model_choice_var.get(),
                manual_model_path=self.manual_model_path_var.get(),
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
            err_text = str(exc)
            self.after(0, lambda msg=err_text: messagebox.showerror("Translation Error", msg))
        finally:
            self.after(0, lambda: self._set_busy(False))

    def _apply_results(self, translation: str, report_text: str):
        self._animate_translation_text(translation)
        self._flash_translation_panel()

        self.report_text.delete("1.0", "end")
        self.report_text.insert("1.0", report_text)
        self._flash_report_panel()

    def _animate_translation_text(self, translation: str):
        if self._typing_anim_job is not None:
            self.after_cancel(self._typing_anim_job)
            self._typing_anim_job = None
        self._typing_target_text = translation
        self._typing_index = 0
        self.translation_text.delete("1.0", "end")
        self._typing_step()

    def _typing_step(self):
        step = 28
        next_idx = min(len(self._typing_target_text), self._typing_index + step)
        if next_idx > self._typing_index:
            chunk = self._typing_target_text[self._typing_index : next_idx]
            self.translation_text.insert("end", chunk)
            self.translation_text.see("end")
            self._typing_index = next_idx
        if self._typing_index < len(self._typing_target_text):
            self._typing_anim_job = self.after(14, self._typing_step)
        else:
            self._typing_anim_job = None

    def _flash_report_panel(self):
        if self._report_flash_job is not None:
            self.after_cancel(self._report_flash_job)
            self._report_flash_job = None
        self._report_flash_step = 0
        self._animate_report_flash()

    def _animate_report_flash(self):
        # Short highlight after translation completes.
        colors = ["#2f6f5e", "#2a6154", "#26574c", "#235046", "#214a41", "transparent"]
        idx = min(self._report_flash_step, len(colors) - 1)
        self.left_bottom.configure(fg_color=colors[idx])
        self._report_flash_step += 1
        if self._report_flash_step < len(colors):
            self._report_flash_job = self.after(80, self._animate_report_flash)
        else:
            self._report_flash_job = None

    def _flash_translation_panel(self):
        if self._translation_flash_job is not None:
            self.after_cancel(self._translation_flash_job)
            self._translation_flash_job = None
        self._translation_flash_step = 0
        self._animate_translation_flash()

    def _animate_translation_flash(self):
        # Short highlight for updated translation output.
        colors = ["#2f5f9e", "#2a588f", "#25507f", "#224a74", "#1f4368", "transparent"]
        idx = min(self._translation_flash_step, len(colors) - 1)
        self.right_top.configure(fg_color=colors[idx])
        self._translation_flash_step += 1
        if self._translation_flash_step < len(colors):
            self._translation_flash_job = self.after(80, self._animate_translation_flash)
        else:
            self._translation_flash_job = None

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

    def _clear_history(self):
        title = "Confirm" if self.ui_lang.get() == "en" else "Подтверждение"
        prompt = "Delete translation history?" if self.ui_lang.get() == "en" else "Удалить историю переводов?"
        if not messagebox.askyesno(title, prompt):
            return
        self.history_manager.clear()
        self._refresh_history()


def main():
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)
    ensure_dir("outputs")
    app = TechnicalTranslatorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
