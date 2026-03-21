from __future__ import annotations

import argparse
import random
from pathlib import Path

from utils import save_json


SUBJECTS = [
    ("The frontend service", "Фронтенд-сервис"),
    ("The backend API", "Бэкенд API"),
    ("The logging module", "Модуль логирования"),
    ("The authentication service", "Сервис аутентификации"),
    ("The cache layer", "Кэш-слой"),
    ("The parser", "Парсер"),
    ("The scheduler", "Планировщик"),
    ("The migration script", "Скрипт миграции"),
    ("The worker process", "Воркер-процесс"),
    ("The monitoring agent", "Агент мониторинга"),
]

VERBS = [
    ("sends", "отправляет"),
    ("validates", "проверяет"),
    ("stores", "сохраняет"),
    ("returns", "возвращает"),
    ("caches", "кэширует"),
    ("retries", "повторяет"),
    ("logs", "логирует"),
    ("normalizes", "нормализует"),
    ("indexes", "индексирует"),
    ("serializes", "сериализует"),
]

OBJECTS = [
    ("an HTTP request", "HTTP-запрос"),
    ("the input payload", "входной payload"),
    ("the response body", "тело ответа"),
    ("database records", "записи базы данных"),
    ("user metadata", "метаданные пользователя"),
    ("frequently requested data", "часто запрашиваемые данные"),
    ("error events", "события ошибок"),
    ("JSON documents", "JSON-документы"),
    ("YAML configuration", "YAML-конфигурацию"),
    ("CLI output", "вывод CLI"),
]

CONDITIONS = [
    (
        "If parameters are invalid, it returns status code 400.",
        "Если параметры некорректны, он возвращает код состояния 400.",
    ),
    (
        "If authentication fails, it returns status code 401.",
        "Если аутентификация завершается ошибкой, он возвращает код состояния 401.",
    ),
    (
        "If access is denied, it returns status code 403.",
        "Если доступ запрещён, он возвращает код состояния 403.",
    ),
    (
        "If a timeout occurs, it retries the request three times.",
        "Если возникает тайм-аут, он повторяет запрос три раза.",
    ),
    (
        "If the resource is missing, it returns status code 404.",
        "Если ресурс отсутствует, он возвращает код состояния 404.",
    ),
]

EXTRAS = [
    (
        "Use `python src/train.py` to start fine-tuning.",
        "Используйте `python src/train.py`, чтобы запустить дообучение.",
    ),
    (
        "Use `pip install -r requirements.txt` before execution.",
        "Используйте `pip install -r requirements.txt` перед выполнением.",
    ),
    (
        "Config path: `configs/train_config.json`.",
        "Путь к конфигу: `configs/train_config.json`.",
    ),
    (
        "Endpoint: `/v1/users/{id}`.",
        "Эндпоинт: `/v1/users/{id}`.",
    ),
    (
        "Payload example: {\"enabled\": true, \"timeout\": 30}.",
        "Пример payload: {\"enabled\": true, \"timeout\": 30}.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic EN->RU technical test set.")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="configs/synthetic_1000_cases.json")
    return parser.parse_args()


def build_case(rng: random.Random) -> dict:
    subject_en, subject_ru = rng.choice(SUBJECTS)
    verb_en, verb_ru = rng.choice(VERBS)
    object_en, object_ru = rng.choice(OBJECTS)
    cond_en, cond_ru = rng.choice(CONDITIONS)
    extra_en, extra_ru = rng.choice(EXTRAS)

    src = f"{subject_en} {verb_en} {object_en}. {cond_en} {extra_en}"
    ref = f"{subject_ru} {verb_ru} {object_ru}. {cond_ru} {extra_ru}"
    return {"src": src, "reference": ref}


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    items = [build_case(rng) for _ in range(args.count)]
    save_json(items, args.output)

    print(f"Generated {len(items)} synthetic cases.")
    print(f"Saved to: {Path(args.output)}")


if __name__ == "__main__":
    main()
