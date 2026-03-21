from __future__ import annotations

import argparse
import random
from pathlib import Path

from utils import save_json


SUBJECTS = [
    ("The frontend service", "Фронтенд-сервис"),
    ("The backend API", "Бэкенд API"),
    ("The auth module", "Модуль аутентификации"),
    ("The logging subsystem", "Подсистема логирования"),
    ("The cache layer", "Кэш-слой"),
    ("The parser", "Парсер"),
    ("The scheduler", "Планировщик"),
    ("The worker process", "Воркер-процесс"),
]

ACTIONS = [
    ("validates input data", "проверяет входные данные"),
    ("stores records in the database", "сохраняет записи в базе данных"),
    ("returns status code 400 for invalid parameters", "возвращает код состояния 400 при неверных параметрах"),
    ("caches frequently requested data", "кэширует часто запрашиваемые данные"),
    ("logs authentication failures and timeout errors", "логирует ошибки аутентификации и ошибки тайм-аута"),
    ("retries failed requests with exponential backoff", "повторяет неуспешные запросы с экспоненциальным backoff"),
]

TAILS = [
    (
        "This approach improves reliability, scalability, and maintainability.",
        "Этот подход повышает надёжность, масштабируемость и удобство сопровождения.",
    ),
    (
        "Use strict validation before executing business logic.",
        "Используйте строгую валидацию перед выполнением бизнес-логики.",
    ),
]

INLINE_CODE = [
    ("Call `model.generate()` after tokenization.", "Вызовите `model.generate()` после токенизации."),
    ("Run `python src/train.py` to start fine-tuning.", "Запустите `python src/train.py`, чтобы начать дообучение."),
    (
        "Install dependencies with `pip install -r requirements.txt`.",
        "Установите зависимости командой `pip install -r requirements.txt`.",
    ),
]

CLI_LINES = [
    (
        "CLI command: `$ curl -X POST https://api.example.com/v1/login`.",
        "Команда CLI: `$ curl -X POST https://api.example.com/v1/login`.",
    ),
    (
        "CLI command: `$ python src/inference.py --text \"Hello\"`.",
        "Команда CLI: `$ python src/inference.py --text \"Hello\"`.",
    ),
]

PATHS = [
    (
        "Config path: `configs/train_config.json`.",
        "Путь к конфигу: `configs/train_config.json`.",
    ),
    (
        "Checkpoint path: `outputs/checkpoints/final`.",
        "Путь к чекпоинту: `outputs/checkpoints/final`.",
    ),
    (
        "Windows path: `C:\\projects\\mt\\configs\\train_config.json`.",
        "Путь Windows: `C:\\projects\\mt\\configs\\train_config.json`.",
    ),
]

JSON_XML_YAML = [
    (
        'JSON payload: {"enabled": true, "timeout": 30, "retries": 3}.',
        'JSON-payload: {"enabled": true, "timeout": 30, "retries": 3}.',
    ),
    (
        "YAML snippet: `timeout: 30\\nretries: 3\\nenabled: true`.",
        "YAML-фрагмент: `timeout: 30\\nretries: 3\\nenabled: true`.",
    ),
    (
        "XML payload: `<request><id>42</id><status>ok</status></request>`.",
        "XML-payload: `<request><id>42</id><status>ok</status></request>`.",
    ),
]

CODE_BLOCKS = [
    (
        "```python\nresponse = client.get('/health')\nassert response.status_code == 200\n```",
        "```python\nresponse = client.get('/health')\nassert response.status_code == 200\n```",
    ),
    (
        "```json\n{\"name\": \"service\", \"enabled\": true, \"timeout\": 30}\n```",
        "```json\n{\"name\": \"service\", \"enabled\": true, \"timeout\": 30}\n```",
    ),
]

MARKDOWN = [
    (
        "### Deployment Notes\n- Validate configs\n- Restart the service\n- Check logs",
        "### Заметки по деплою\n- Проверьте конфиги\n- Перезапустите сервис\n- Проверьте логи",
    ),
    (
        "1. Build the image\n2. Push to registry\n3. Deploy to production",
        "1. Соберите образ\n2. Отправьте в registry\n3. Разверните в production",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate rich synthetic EN->RU test set.")
    parser.add_argument("--count", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="configs/synthetic_8000_rich_cases.json")
    return parser.parse_args()


def build_base_sentence(rng: random.Random) -> tuple[str, str]:
    s_en, s_ru = rng.choice(SUBJECTS)
    a_en, a_ru = rng.choice(ACTIONS)
    t_en, t_ru = rng.choice(TAILS)
    return f"{s_en} {a_en}. {t_en}", f"{s_ru} {a_ru}. {t_ru}"


def build_case(rng: random.Random) -> dict:
    src, ref = build_base_sentence(rng)

    enrichers = [
        INLINE_CODE,
        CLI_LINES,
        PATHS,
        JSON_XML_YAML,
        CODE_BLOCKS,
        MARKDOWN,
    ]
    picks = rng.sample(enrichers, k=rng.randint(2, 4))
    for group in picks:
        e_en, e_ru = rng.choice(group)
        src = f"{src}\n\n{e_en}"
        ref = f"{ref}\n\n{e_ru}"

    if rng.random() < 0.35:
        src = f"Endpoint `/v1/users/{{id}}` returns metadata.\n\n{src}"
        ref = f"Эндпоинт `/v1/users/{{id}}` возвращает метаданные.\n\n{ref}"

    return {"src": src, "reference": ref}


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    items = [build_case(rng) for _ in range(args.count)]
    save_json(items, args.output)
    print(f"Generated {len(items)} rich synthetic cases.")
    print(f"Saved to: {Path(args.output)}")


if __name__ == "__main__":
    main()
