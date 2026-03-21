# EN->RU Technical MT Baseline + GUI

Minimal reproducible coursework project for technical machine translation with CLI scripts and a modern desktop GUI.

## Core MT Setup

- Base model: `Helsinki-NLP/opus-mt-en-ru`
- Optional fine-tuned checkpoint: `./outputs/checkpoints/final`
- Default EN-RU dataset for training/eval config: `opus100` (`en-ru`)
- Training pipeline supports technical subset filtering via keywords (see `configs/train_config.json`)

## Metrics

- BLEU
- chrF
- terminology accuracy (technical term preservation on references)
- Average generated length (`gen_len`)

## Install

```bash
pip install -r requirements.txt
```

## CLI Baseline Workflow

Baseline evaluation:

```bash
python src/evaluate_baseline.py
```

Fine-tuning:

```bash
python src/train.py
```

Inference:

```bash
python src/inference.py --text "The parser validates the input before execution."
python src/inference.py --text "The compiler throws an exception if the config file is missing." --model_path ./outputs/checkpoints/final
```

## GUI App

Run GUI:

```bash
python src/gui_app.py
```

Implemented GUI features:

- dark/light theme
- large source and translation panels
- technical-text-aware preprocessing (markdown, code blocks, inline code, CLI lines, paths, JSON/XML)
- language auto-detection
- automatic model routing
- chunking for long texts
- full inference report
- translation history
- copy/save/export actions
- interface language switch (EN/RU)
- modes: `fast` / `balanced` / `high_quality`

Backend modules:

- `src/language_detector.py`
- `src/text_analyzer.py`
- `src/model_router.py`
- `src/translator_service.py`
- `src/report_builder.py`
- `src/history_manager.py`

## Config

- Main training config: `configs/train_config.json`
- For larger fine-tuning: increase sample sizes and keep `use_technical_filter=true`
