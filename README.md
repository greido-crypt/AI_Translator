# EN->RU Technical MT Baseline + GUI

Minimal reproducible coursework project for technical machine translation (English -> Russian) with CLI scripts and a desktop GUI.

## What Is Included

- baseline model: `Helsinki-NLP/opus-mt-en-ru`
- fine-tuning pipeline (`transformers` + `datasets` + `torch`)
- baseline/final evaluation scripts
- GUI for technical text translation with report panel
- technical token protection for code/CLI/paths/JSON/XML

## Dataset and Config

- default dataset in config: `opus100` (`en-ru`)
- main config: `configs/train_config.json`
- extended configs:
  - `configs/train_config_synth8000.json`
  - `configs/train_high_accuracy.json`

## Metrics

- BLEU
- chrF
- terminology preservation score
- code preservation score
- formatting preservation score

## Installation

```bash
pip install -r requirements.txt
```

## Download Fine-Tuned Model (Yandex.Disk)

Model weights are stored outside GitHub due size limits.

1. Download model files from Yandex.Disk: `<PASTE_YOUR_YANDEX_LINK_HERE>`
2. Place them into `./outputs/checkpoints/final`
3. Required files:
   - `config.json`
   - `generation_config.json`
   - `tokenizer_config.json`
   - `vocab.json`
   - `source.spm`
   - `target.spm`
   - `model.safetensors`

Quick check:

```bash
python src/inference.py --text "The parser validates the input before execution." --model_path ./outputs/checkpoints/final
```

## CLI Usage

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

## GUI Usage

Run:

```bash
python src/gui_app.py
```

GUI includes:

- dark/light themes
- translation modes: `fast`, `balanced`, `high_quality`
- language auto-detection
- model auto-routing
- long-text chunking
- history, copy/save/export actions
- detailed inference report

## Project Structure

```text
configs/   # training and synthetic-data configs
src/       # pipeline, GUI, and service modules
outputs/   # runtime outputs (ignored in git, except .gitkeep)
```

## Notes for GitHub

- model files are not stored in repository
- checkpoints and runtime artifacts are ignored by `.gitignore`
- share model via Yandex.Disk link in this README
