from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import Dataset, DatasetDict, concatenate_datasets, get_dataset_config_names, load_dataset

DEFAULT_TECH_KEYWORDS = [
    "api",
    "http",
    "https",
    "request",
    "response",
    "server",
    "client",
    "frontend",
    "backend",
    "database",
    "sql",
    "nosql",
    "cache",
    "caching",
    "logging",
    "authentication",
    "authorization",
    "timeout",
    "exception",
    "token",
    "endpoint",
    "parser",
    "compiler",
    "runtime",
    "framework",
    "microservice",
    "json",
    "yaml",
    "xml",
    "docker",
    "kubernetes",
]

DEFAULT_HARD_AUG_RULES = [
    (r"\bstatus code\b", "HTTP status code"),
    (r"\bbackend\b", "backend service"),
    (r"\bfrontend\b", "frontend client"),
    (r"\bcache\b", "cache layer"),
    (r"\blogging\b", "logging subsystem"),
]


def _maybe_fix_mojibake_ru(text: str) -> str:
    candidate = text.strip()
    if not candidate:
        return candidate

    suspicious = candidate.count("Р") + candidate.count("С") + candidate.count("Ð") + candidate.count("Ñ")
    if suspicious < max(6, len(candidate) // 12):
        return candidate

    for src_enc, dst_enc in (("cp1251", "utf-8"), ("latin1", "utf-8")):
        try:
            repaired = candidate.encode(src_enc, errors="strict").decode(dst_enc, errors="strict")
            cyr_count = len(re.findall(r"[\u0400-\u04FF]", repaired))
            if cyr_count > len(repaired) * 0.35:
                return repaired
        except Exception:
            continue
    return candidate


def _load_local_json_dataset(path: str) -> Dataset:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Local dataset file not found: {input_path}")

    with input_path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)

    if not isinstance(payload, list) or not payload:
        raise ValueError("Local dataset JSON must be a non-empty list.")

    rows = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        src = str(item.get("src", item.get("source", ""))).strip()
        tgt = str(item.get("reference", item.get("target", ""))).strip()
        tgt = _maybe_fix_mojibake_ru(tgt)
        if src and tgt:
            rows.append({"source": src, "target": tgt})

    if not rows:
        raise ValueError("Local dataset has no valid src/reference pairs after parsing.")
    return Dataset.from_list(rows)


def _candidate_configs(lang_pair: str) -> List[str]:
    base = lang_pair.replace("_", "-").lower()
    src, tgt = base.split("-")
    candidates = [f"{src}-{tgt}", f"{src}_{tgt}", f"{tgt}-{src}", f"{tgt}_{src}"]
    unique = []
    for item in candidates:
        if item not in unique:
            unique.append(item)
    return unique


def _resolve_configs(dataset_name: str, lang_pair: str) -> List[str]:
    guessed = _candidate_configs(lang_pair)
    try:
        config_names = get_dataset_config_names(dataset_name, trust_remote_code=True)
    except Exception:
        return guessed

    matches = [cfg for cfg in config_names if cfg in guessed]
    if matches:
        return matches

    src, tgt = lang_pair.replace("_", "-").lower().split("-")
    for cfg in config_names:
        normalized = cfg.replace("_", "-").lower()
        if normalized in (f"{src}-{tgt}", f"{tgt}-{src}"):
            matches.append(cfg)

    return matches if matches else guessed


def _load_dataset_for_lang_pair(dataset_name: str, lang_pair: str) -> DatasetDict:
    dataset_ids = [dataset_name]
    short_name = dataset_name.split("/")[-1]
    if short_name not in dataset_ids:
        dataset_ids.append(short_name)

    errors = []
    load_kwargs_options = [{"trust_remote_code": True}, {}]
    for dataset_id in dataset_ids:
        for config_name in _resolve_configs(dataset_id, lang_pair):
            for load_kwargs in load_kwargs_options:
                try:
                    return load_dataset(dataset_id, config_name, **load_kwargs)
                except Exception as exc:
                    mode = "trust_remote_code=True" if load_kwargs else "default"
                    errors.append(f"{dataset_id}/{config_name} [{mode}]: {exc}")

    joined_errors = "\n".join(errors[:8])
    raise RuntimeError(
        "Failed to load dataset for requested language pair. "
        "Check dataset/config compatibility or use a dataset with EN-RU pair. "
        f"Tried combinations:\n{joined_errors}"
    )


def _merge_all_splits(dataset_dict: DatasetDict) -> Dataset:
    split_names = list(dataset_dict.keys())
    if not split_names:
        raise ValueError("Loaded dataset has no splits.")
    datasets = [dataset_dict[name] for name in split_names]
    return datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)


def _extract_source_target(example: Dict, source_lang: str, target_lang: str) -> Dict[str, str]:
    if "translation" in example and isinstance(example["translation"], dict):
        translation = example["translation"]
        if source_lang not in translation or target_lang not in translation:
            raise KeyError(
                f"Missing translation keys. Expected '{source_lang}' and '{target_lang}', "
                f"got: {list(translation.keys())}"
            )
        source = translation[source_lang]
        target = translation[target_lang]
    elif source_lang in example and target_lang in example:
        source = example[source_lang]
        target = example[target_lang]
    else:
        raise KeyError(
            "Could not extract text pair. Expected either 'translation' dict "
            f"or direct fields '{source_lang}' and '{target_lang}'."
        )

    return {"source": str(source).strip(), "target": str(target).strip()}


def _drop_full_duplicates(dataset: Dataset) -> Dataset:
    seen = set()
    keep_indices = []
    for idx, row in enumerate(dataset):
        key = (row["source"], row["target"])
        if key in seen:
            continue
        seen.add(key)
        keep_indices.append(idx)
    return dataset.select(keep_indices)


def _apply_technical_filter(dataset: Dataset, config: Dict) -> Dataset:
    use_filter = bool(config.get("use_technical_filter", False))
    if not use_filter:
        return dataset

    keywords = [kw.lower() for kw in config.get("technical_keywords", DEFAULT_TECH_KEYWORDS)]
    min_hits = int(config.get("technical_min_keyword_hits", 1))

    if not keywords:
        return dataset

    pattern = re.compile(r"\b(?:" + "|".join(re.escape(kw) for kw in keywords) + r")\b", re.IGNORECASE)

    def is_technical(ex: Dict[str, str]) -> bool:
        src = ex["source"]
        tgt = ex["target"]
        hits = len(pattern.findall(src)) + len(pattern.findall(tgt))
        return hits >= min_hits

    filtered = dataset.filter(is_technical, desc="Filter technical pairs")
    if len(filtered) == 0:
        raise ValueError(
            "Technical filter removed all samples. Relax technical keywords/min hits or disable filter."
        )
    return filtered


def _augment_source_text(text: str, rng: random.Random) -> str:
    updated = text
    if rng.random() < 0.30:
        pattern, replacement = rng.choice(DEFAULT_HARD_AUG_RULES)
        updated = re.sub(pattern, replacement, updated, flags=re.IGNORECASE)
    if rng.random() < 0.45:
        updated = re.sub(r"\s+", " ", updated).strip()
    if rng.random() < 0.35:
        updated = updated.replace(" ,", ",").replace(" .", ".")
    if rng.random() < 0.30:
        updated = re.sub(r"\bHTTP\b", "Http", updated)
    if rng.random() < 0.25:
        updated = updated.replace("API", "Api")
    if rng.random() < 0.22:
        updated = re.sub(r"\bendpoint\b", "API endpoint", updated, flags=re.IGNORECASE)
    if rng.random() < 0.18:
        updated = updated + " "
    return updated


def _augment_train_split(train_ds: Dataset, config: Dict) -> Dataset:
    if not bool(config.get("enable_noise_augmentation", False)):
        return train_ds

    ratio = float(config.get("noise_augmentation_ratio", 0.1))
    ratio = max(0.0, min(ratio, 0.5))
    if ratio <= 0:
        return train_ds

    augment_count = int(len(train_ds) * ratio)
    if augment_count <= 0:
        return train_ds

    seed = int(config.get("seed", 42))
    sampled = train_ds.shuffle(seed=seed + 11).select(range(augment_count))

    def _mutate(example: Dict, idx: int) -> Dict[str, str]:
        rng = random.Random(seed + idx)
        return {"source": _augment_source_text(example["source"], rng), "target": example["target"]}

    augmented = sampled.map(_mutate, with_indices=True, desc="Create noisy source variants")
    merged = concatenate_datasets([train_ds, augmented]).shuffle(seed=seed + 13)
    return merged


def load_and_prepare_raw_datasets(config: Dict) -> DatasetDict:
    dataset_name = config["dataset_name"]
    lang_pair = config["lang_pair"]
    source_lang = config["source_lang"]
    target_lang = config["target_lang"]

    max_train = int(config["max_train_samples"])
    max_eval = int(config["max_eval_samples"])
    max_test = int(config["max_test_samples"])

    local_dataset_path = config.get("local_dataset_path")
    use_local_only = bool(config.get("use_local_dataset_only", False))

    datasets_to_merge: List[Dataset] = []
    if local_dataset_path:
        local_ds = _load_local_json_dataset(str(local_dataset_path))
        datasets_to_merge.append(local_ds)

    if not use_local_only:
        loaded = _load_dataset_for_lang_pair(dataset_name, lang_pair)
        merged = _merge_all_splits(loaded)
        try:
            remote_pairs = merged.map(
                lambda ex: _extract_source_target(ex, source_lang, target_lang),
                remove_columns=merged.column_names,
                desc="Extract source/target",
            )
        except KeyError as exc:
            raise ValueError(
                f"Failed to parse dataset examples for languages '{source_lang}' and '{target_lang}': {exc}"
            ) from exc
        datasets_to_merge.append(remote_pairs)

    if not datasets_to_merge:
        raise ValueError("No dataset source available. Provide HF dataset or local_dataset_path.")

    pair_dataset = datasets_to_merge[0] if len(datasets_to_merge) == 1 else concatenate_datasets(datasets_to_merge)

    pair_dataset = pair_dataset.filter(
        lambda ex: bool(ex["source"]) and bool(ex["target"]),
        desc="Filter empty pairs",
    )

    pair_dataset = _drop_full_duplicates(pair_dataset)

    max_source_chars = int(config.get("max_source_chars", config["max_source_length"] * 8))
    max_target_chars = int(config.get("max_target_chars", config["max_target_length"] * 8))

    pair_dataset = pair_dataset.filter(
        lambda ex: len(ex["source"]) <= max_source_chars and len(ex["target"]) <= max_target_chars,
        desc="Filter long pairs",
    )

    pair_dataset = _apply_technical_filter(pair_dataset, config)

    required_total = max_train + max_eval + max_test
    available = len(pair_dataset)
    if available < required_total:
        raise ValueError(
            f"Too few examples after filtering: {available}. Required at least {required_total} "
            f"({max_train} train + {max_eval} val + {max_test} test)."
        )

    shuffled = pair_dataset.shuffle(seed=int(config.get("seed", 42)))

    train_end = max_train
    val_end = train_end + max_eval
    test_end = val_end + max_test

    splits = DatasetDict(
        {
            "train": shuffled.select(range(0, train_end)),
            "validation": shuffled.select(range(train_end, val_end)),
            "test": shuffled.select(range(val_end, test_end)),
        }
    )

    splits["train"] = _augment_train_split(splits["train"], config)
    return splits


def preprocess_function(examples: Dict, tokenizer, max_source_length: int, max_target_length: int) -> Dict:
    model_inputs = tokenizer(
        examples["source"],
        max_length=max_source_length,
        truncation=True,
    )
    labels = tokenizer(
        text_target=examples["target"],
        max_length=max_target_length,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def load_tokenized_datasets(config: Dict, tokenizer) -> Tuple[DatasetDict, DatasetDict]:
    raw_splits = load_and_prepare_raw_datasets(config)

    tokenized = raw_splits.map(
        lambda batch: preprocess_function(
            batch,
            tokenizer=tokenizer,
            max_source_length=int(config["max_source_length"]),
            max_target_length=int(config["max_target_length"]),
        ),
        batched=True,
        remove_columns=raw_splits["train"].column_names,
        desc="Tokenize",
    )

    return tokenized, raw_splits
