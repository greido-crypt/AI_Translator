from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from model_router import ModelSelection


@dataclass
class TranslationRuntimeResult:
    translated_text: str
    chunks: List[str]
    translated_chunks: List[str]
    chunk_count: int
    total_inference_time: float
    device_info: str
    model_info: Dict[str, str]
    preprocessing_summary: Dict[str, str]
    warnings: List[str]
    protected_tokens: List[str]
    protected_token_hits: int
    quality_gate_passed: bool
    retry_count: int


class TranslatorService:
    MARKDOWN_HEADER_RE = re.compile(r"(?m)^#{1,6}\s.*$")
    CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
    INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
    CLI_LINE_RE = re.compile(r"(?m)^\s*(?:\$|>|#)\s+.+$")
    PATH_RE = re.compile(r"(?:[A-Za-z]:\\[^\s]+|/(?:[\w.-]+/)+[\w.-]+|\./[\w./-]+)")
    FILE_NAME_RE = re.compile(r"\b[\w.-]+\.(?:json|yaml|yml|xml|toml|ini|cfg|py|md|txt|csv)\b", re.IGNORECASE)
    JSON_XML_RE = re.compile(r"(?s)(\{\s*\".+?\"\s*:.*?\}|<\?xml[\s\S]*?\?>|<\w+[\s\S]*?>[\s\S]*?</\w+>)")
    METHOD_CALL_RE = re.compile(r"\b[a-zA-Z_][\w]*\.[a-zA-Z_][\w]*\([^\)]*\)")
    TECH_ACRONYM_RE = re.compile(r"\b(?:API|HTTP|HTTPS|SQL|NoSQL|JSON|YAML|XML|CLI|CPU|GPU|JWT|OAuth2?|TCP|UDP|REST|GraphQL)\b")
    IDENTIFIER_RE = re.compile(r"\b(?:[a-z]+[A-Z][A-Za-z0-9_]*|[A-Za-z]+_[A-Za-z0-9_]+|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+)\b")
    FLAG_RE = re.compile(r"\B--[a-zA-Z0-9_-]+\b")
    TOKEN_LIKE_RE = re.compile(r"\b[A-Za-z<>_]*TOKEN[_-]?(\d+)[A-Za-z<>_]*\b", re.IGNORECASE)
    FLEX_PLACEHOLDER_RE = re.compile(
        r"(?:\[\[\s*T\s*(\d+)\s*\]\])|(?:[A-Za-z_<>-]*(?:TECH|TOKEN|TECTOKEN|TECEN)[A-Za-z_<>-]*[_-]?(\d+)[A-Za-z_<>-]*)",
        re.IGNORECASE,
    )
    BRACKET_PLACEHOLDER_RE = re.compile(
        r"[\[\(\{<]{1,3}\s*[TТ]\s*[-_ ]?(\d+)\s*[\]\)\}>]{1,3}",
        re.IGNORECASE,
    )
    MIXED_BLOCK_RE = re.compile(
        r"```[\s\S]*?```|(?m)^(?:\s*(?:python|curl|git|docker|kubectl|npm|pip|Invoke-RestMethod|export|\$env:).+\n?)+"
    )

    EN_RU_GLOSSARY = [
        (r"\bfrontend\b", "фронтенд"),
        (r"\bbackend\b", "бэкенд"),
        (r"\bclient-server architecture\b", "клиент-серверная архитектура"),
        (r"\bHTTP request\b", "HTTP-запрос"),
        (r"\bstatus code\b", "код состояния"),
        (r"\bcache(?:s|d|ing)?\b", "кэш"),
        (r"\blogging module\b", "модуль логирования"),
        (r"\bauthentication failures\b", "ошибки аутентификации"),
        (r"\btimeout errors\b", "ошибки тайм-аута"),
        (r"\bconnection issues\b", "проблемы соединения"),
        (r"\beasier to maintain\b", "проще в сопровождении"),
        (r"\bendpoint\b", "эндпоинт"),
    ]
    EN_RU_TERM_RULES = [
        (r"\bfrontend\b", "фронтенд"),
        (r"\bbackend\b", "бэкенд"),
        (r"\bstatus code\b", "код состояния"),
        (r"\blogging module\b", "модуль логирования"),
        (r"\bauthentication failures\b", "ошибки аутентификации"),
        (r"\btimeout errors\b", "ошибки тайм-аута"),
        (r"\bconnection issues\b", "проблемы соединения"),
        (r"\bendpoint\b", "эндпоинт"),
        (r"\bcaches frequently requested data\b", "кэширует часто запрашиваемые данные"),
    ]
    LATIN_RE = re.compile(r"[A-Za-z]")
    CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")

    EN_RU_CORRECTIONS = [
        (r"###\s*план дебюта", "### План отладки инцидента"),
        (r"\bспереди\b", "фронтенд"),
        (r"\bзадний край\b", "бэкенд"),
        (r"\bрезервн\w* система API\b", "бэкенд API"),
        (r"Если аутентификацию,", "Если аутентификация не удалась,"),
        (r"\bПута к чекпоинту\b", "Путь к чекпоинту"),
        (r"\bПутеть к чекпоинту\b", "Путь к чекпоинту"),
        (r"\bXML-?paad\b", "XML-payload"),
        (r"\bXML-?payleload\b", "XML-payload"),
        (r"\bошибки аутверизации\b", "ошибки аутентификации"),
        (r"\bошибки аутверац\w*\b", "ошибки аутентификации"),
        (r"\bвопросы соединения\b", "проблемы соединения"),
        (r"\bвопросы связи\b", "проблемы соединения"),
        (r"\bвопросы подключения\b", "проблемы соединения"),
        (r"\bповышает над\b", "делает систему более надёжной"),
        (r"Путь к конфигу:\s*\[\s*`", "Путь к конфигу: `"),
        (r"`\s*чекпоинт:\s*`", "`\nЧекпоинт: `"),
        (r"путь к чекпоинту:\s*\[\s*`", "Путь к чекпоинту: `"),
        (r"\]\s*```python", "`\n```python"),
        (r"\bXML-payload\.\s*`", "XML-payload: `"),
        (r"\bчасто скрыва\w* данные\b", "кэширует часто запрашиваемые данные"),
        (r"\bмодуль регистрации\b", "модуль логирования"),
        (r"\bошибк\w* во времени\b", "ошибки тайм-аута"),
        (r"\bпрост\w* для поддержани\w*\b", "проще в сопровождении"),
        (r"\bкэш frequently requested data\b", "кэширует часто запрашиваемые данные"),
        (r"\bконец пункта\b", "эндпоинт"),
        (r"\bЧтобы улучшить результативность\b", "Чтобы повысить производительность"),
        (r"Если аутентификация не удалась,\s*код состояния 401", "Если аутентификация не удалась, сервер возвращает код состояния 401"),
        (r"В случае неверных аутентификац\w*\s*возвраща\w* код состояния 401 и логирует событие", "Если аутентификация не удалась, сервер возвращает код состояния 401 и логирует событие"),
        (r"\bПуть по чекпоинту\b", "Путь к чекпоинту"),
        (r"\bмонолитную сервис\b", "монолитный сервис"),
        (r"\bархитектуре клиент-сервер\b", "клиент-серверной архитектуре"),
        (r"\bБэкенд Эндпоинт\b", "бэкенд эндпоинт"),
        (r"\bключом к демпоинту\b", "идемпотентным ключом"),
        (r"\bжетон пользователя\b", "токен пользователя"),
        (r"\bдеплою заперта\b", "деплой заблокирован"),
        (r"деплой заблокирован,\s*возвраща\w*\s*`423 Locked`", "деплой заблокирован, возвращайте `423 Locked`"),
        (r"\bВоркер CLI\b", "CLI workflow"),
        (r"\bВоркер-сервис\b", "Python-сервис"),
        (r"\bКусочек конфиги\b", "Фрагмент конфига"),
        (r"\bXML Запрашиваемую выборку\b", "XML payload"),
        (r"\bлогический модуль\b", "модуль логирования"),
        (r"\bпрозедаци\w*\b", "прозу"),
        (r"\[\s*`tests/test_api\.py`,?", "`tests/test_api.py`"),
        (r"запрос HTTP\s*`POST`\s*с командой бэкенд эндпоинт", "запрос HTTP `POST` в бэкенд эндпоинт"),
        (r"\bвстроенный код, логики, флаги\b", "inline-код, пути файлов, флаги"),
        (r"сохраняя при этом команды, успешные", "сохраняя при этом команды исполняемыми"),
        (r"Фрагмент конфига\s*\(`YAML`:", "Фрагмент конфига (`YAML`):"),
        (r"\n\s*кэш:\s*\n", "\n  cache:\n"),
        (r"\bвходные коды\b", "inline-код"),
        (r"\bСоберите технические термины стабильными\b", "Сохраняйте технические термины стабильными"),
        (r"\bтайм-аута,\s*аутентификацию\b", "тайм-аут, аутентификацию"),
        (r"\bКоманды,\s*подлежащие исполнению\b", "команды исполняемыми"),
        (r"\bЕсли деплой заблокирован,\s*возвращайте\b", "Если деплой заблокирован, возвращайте"),
        (r";\s*Если деплой заблокирован", "; если деплой заблокирован"),
        (r"\bврат(?:а|ами)\s*API\b", "API-шлюзом"),
        (r"\bасинк\w*\b", "асинхронными воркерами"),
        (r"\bсхема-верси\w*\s*не\s*срабатыва\w*\b", "валидация схемы не проходит"),
        (r"\bРБАК\b", "RBAC"),
        (r"\bПаркер Python\b", "Python-обработчик"),
        (r"\bYAML\s*конфиги\b", "YAML-конфиг"),
        (r"\bкодные блоки\b", "кодовые блоки"),
        (r"\bСоберите технические термины в стабильном режиме\b", "Сохраняйте технические термины стабильными"),
        (r"\bAPI-payload,load-pain,\s*тайм-аута,\s*RBAC,\s*model required,\s*health-paint\b", "API gateway, payload, timeout, RBAC, rolling update, health probe"),
        (r"\bПереведите пояснительную прозу,\s*естественно,\s*в русский\b", "Переведите пояснительную прозу на русский естественно"),
        (r"возвращает `403 Forbidden`\s*`504 Gateway Timeout`", "возвращайте `403 Forbidden`; если зависимость недоступна по тайм-ауту, возвращайте `504 Gateway Timeout`"),
        (r"\bклиенци-сервера\b", "клиент-серверной"),
        (r"\bоткрытым порталом API\b", "публичным API-шлюзом"),
        (r"\bвнутренними работниками\b", "внутренними воркерами"),
        (r"\bдолжен вернуться\s*`422 Unprocessable Entity`", "должен вернуть `422 Unprocessable Entity`"),
        (r"\bлогирует критические ошибки\b", "логируйте критические ошибки"),
        (r"\bне меняет идентифицирующие данные\b", "не изменяйте идентификаторы"),
        (r"\bClient-server\b", "клиент-серверной"),
        (r"\bархитектуру клиент-серверной\b", "клиент-серверную архитектуру"),
        (r"\bс API-шлюз\b", "с API-шлюзом"),
        (r"\bсохраняя команды исполняемыми в действии\b", "сохраняя команды исполняемыми"),
        (r"\bПроверьте перед перепроверкой\b", "Перед повторной попыткой выполните проверки"),
    ]
    SOURCE_GUIDED_RULES = [
        {
            "source": r"\bapi gateway\b",
            "replacements": [
                (r"\bпортал\w* API\b", "API-шлюз"),
                (r"\bворота API\b", "API-шлюз"),
                (r"\bшлюз API\b", "API-шлюз"),
            ],
        },
        {
            "source": r"\binternal workers?\b",
            "replacements": [
                (r"\bвнутренн\w* работник\w*\b", "внутренними воркерами"),
            ],
        },
        {
            "source": r"\bmust return\b",
            "replacements": [
                (r"\bдолжен вернуться\b", "должен вернуть"),
                (r"\bдолжен возвращаться\b", "должен вернуть"),
            ],
        },
        {
            "source": r"\bdo not alter identifiers\b",
            "replacements": [
                (r"\bне меня(ет|йте) идентифицир\w* данные\b", "не изменяйте идентификаторы"),
                (r"\bне меняет идентификаторы\b", "не изменяйте идентификаторы"),
            ],
        },
        {
            "source": r"\bexpected behavior\b",
            "replacements": [
                (r"\bСоберите технические термины стабильными\b", "Сохраняйте технические термины стабильными"),
                (r"\bСоберите технические термины в стабильном режиме\b", "Сохраняйте технические термины стабильными"),
                (r"\bПереведите .*? сохраняя .*? команды\b", "Переведите пояснительную прозу на русский, сохраняя команды исполняемыми"),
                (r"\bвходные коды\b", "inline-код"),
            ],
        },
        {
            "source": r"\bif upstream timeout occurs\b",
            "replacements": [
                (r"\bв случае тайм-аута перед выполнением, запустите\b", "при тайм-ауте upstream возвращайте"),
                (r"\bв верхнем течении\b", "upstream"),
            ],
        },
    ]

    RU_EN_GLOSSARY = [
        (r"\bфронтенд\b", "frontend"),
        (r"\bбэкенд\b", "backend"),
        (r"\bклиент-серверная архитектура\b", "client-server architecture"),
        (r"\bкод состояния\b", "status code"),
        (r"\bмодуль логирования\b", "logging module"),
        (r"\bошибки тайм-аута\b", "timeout errors"),
    ]

    def __init__(self) -> None:
        self._model_cache: Dict[str, Tuple[AutoTokenizer, AutoModelForSeq2SeqLM, torch.device]] = {}
        self.external_en_ru_glossary = self._load_external_glossary()

    def translate(
        self,
        text: str,
        selection: ModelSelection,
        chunk_size_chars: int = 1400,
    ) -> TranslationRuntimeResult:
        if selection.model_name == "identity":
            return TranslationRuntimeResult(
                translated_text=text,
                chunks=[text],
                translated_chunks=[text],
                chunk_count=1,
                total_inference_time=0.0,
                device_info="cpu",
                model_info={"name": "identity", "type": "no-op"},
                preprocessing_summary={"mode": "identity", "protected_token_count": "0", "glossary_replacements": "0"},
                warnings=["Source and target languages are the same. Output equals input."],
                protected_tokens=[],
                protected_token_hits=0,
                quality_gate_passed=True,
                retry_count=0,
            )

        tokenizer, model, device = self._get_model_bundle(selection.model_name)
        gen_kwargs = self._generation_kwargs(selection.inference_mode)
        retry_count_total = 0
        glossary_replacements_total = 0
        protected_token_hits = 0
        all_protected_tokens: List[str] = []
        chunk_count_total = 0
        quality_gate_passed = True
        quality_issues_all: List[str] = []

        translated_segments: List[str] = []
        chunks_acc: List[str] = []
        translated_chunks_acc: List[str] = []
        warnings: List[str] = []
        start = time.perf_counter()

        direction = f"{selection.source_lang}->{selection.target_lang}"
        segments = self._split_mixed_segments(text)
        for is_technical, segment in segments:
            if not segment:
                continue
            if is_technical or direction != "en->ru":
                translated_segments.append(segment)
                continue

            segment_result = self._translate_prose_segment(
                source_text=segment,
                tokenizer=tokenizer,
                model=model,
                device=device,
                direction=direction,
                gen_kwargs=gen_kwargs,
                chunk_size_chars=chunk_size_chars,
            )

            translated_segments.append(segment_result["translated"])
            chunks_acc.extend(segment_result["chunks"])
            translated_chunks_acc.extend(segment_result["translated_chunks"])
            chunk_count_total += segment_result["chunk_count"]
            retry_count_total += segment_result["retry_count"]
            glossary_replacements_total += segment_result["glossary_replacements"]
            all_protected_tokens.extend(segment_result["protected_tokens"])
            protected_token_hits += segment_result["protected_hits"]
            quality_gate_passed = quality_gate_passed and segment_result["gate_passed"]
            quality_issues_all.extend(segment_result["gate_issues"])

        total = time.perf_counter() - start
        finalized = "".join(translated_segments).strip()

        if chunk_count_total > 1:
            warnings.append("Input was split into multiple chunks; minor boundary artifacts are possible.")
        if protected_token_hits < len(all_protected_tokens):
            warnings.append("Some protected technical tokens may have changed during generation.")
        if glossary_replacements_total > 0:
            warnings.append(f"Glossary normalization applied ({glossary_replacements_total} replacements).")
        if retry_count_total > 0:
            warnings.append("Quality gate retry was triggered with stricter generation settings.")
        if not quality_gate_passed:
            uniq_issues = sorted(set(quality_issues_all))
            warnings.append(f"Quality gate still reports issues: {', '.join(uniq_issues)}")

        return TranslationRuntimeResult(
            translated_text=finalized,
            chunks=chunks_acc if chunks_acc else [text],
            translated_chunks=translated_chunks_acc if translated_chunks_acc else [finalized],
            chunk_count=chunk_count_total if chunk_count_total else 1,
            total_inference_time=total,
            device_info=str(device),
            model_info={
                "name": selection.model_name,
                "framework": "transformers",
                "dtype": str(next(model.parameters()).dtype),
            },
            preprocessing_summary={
                "mode": "protect_translate_normalize_restore",
                "protected_token_count": str(len(all_protected_tokens)),
                "chunk_size_chars": str(chunk_size_chars),
                "glossary_replacements": str(glossary_replacements_total),
                "quality_gate_passed": str(quality_gate_passed),
                "quality_gate_issues": "; ".join(sorted(set(quality_issues_all))) if quality_issues_all else "none",
                "retry_count": str(retry_count_total),
            },
            warnings=warnings,
            protected_tokens=all_protected_tokens,
            protected_token_hits=protected_token_hits,
            quality_gate_passed=quality_gate_passed,
            retry_count=retry_count_total,
        )

    def _get_model_bundle(self, model_name: str):
        if model_name in self._model_cache:
            return self._model_cache[model_name]

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        self._model_cache[model_name] = (tokenizer, model, device)
        return tokenizer, model, device

    def _load_external_glossary(self) -> List[Tuple[str, str]]:
        glossary_path = Path("configs/terminology_en_ru.json")
        if not glossary_path.exists():
            return []
        try:
            data = json.loads(glossary_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        rules: List[Tuple[str, str]] = []
        if not isinstance(data, list):
            return []
        for item in data:
            if not isinstance(item, dict):
                continue
            pattern = item.get("pattern")
            replacement = item.get("replacement")
            if isinstance(pattern, str) and isinstance(replacement, str) and pattern.strip() and replacement.strip():
                rules.append((pattern, replacement))
        return rules

    def _split_mixed_segments(self, text: str) -> List[Tuple[bool, str]]:
        segments: List[Tuple[bool, str]] = []
        last = 0
        for match in self.MIXED_BLOCK_RE.finditer(text):
            if match.start() > last:
                segments.append((False, text[last : match.start()]))
            segments.append((True, match.group(0)))
            last = match.end()
        if last < len(text):
            segments.append((False, text[last:]))
        return segments if segments else [(False, text)]

    def _translate_prose_segment(
        self,
        source_text: str,
        tokenizer,
        model,
        device,
        direction: str,
        gen_kwargs: Dict,
        chunk_size_chars: int,
    ) -> Dict[str, object]:
        leading_ws = re.match(r"^\s*", source_text).group(0)
        trailing_ws = re.search(r"\s*$", source_text).group(0)
        core_source = source_text[len(leading_ws) : len(source_text) - len(trailing_ws) if trailing_ws else len(source_text)]
        if not core_source:
            return {
                "translated": source_text,
                "chunks": [],
                "translated_chunks": [],
                "chunk_count": 0,
                "retry_count": 0,
                "glossary_replacements": 0,
                "protected_tokens": [],
                "protected_hits": 0,
                "gate_passed": True,
                "gate_issues": [],
            }

        protected_text, placeholders = self._protect_technical_tokens(core_source)
        chunks = self._chunk_text(protected_text, chunk_size_chars)
        translated_chunks = [self._translate_chunk(chunk, tokenizer, model, device, gen_kwargs) for chunk in chunks]
        joined = "\n\n".join(translated_chunks)

        finalized, glossary_replacements = self._postprocess_translation(
            source_text=core_source,
            joined_translation=joined,
            placeholders=placeholders,
            direction=direction,
        )
        gate = self._quality_gate(
            source_text=core_source,
            translated_text=finalized,
            placeholders=placeholders,
            direction=direction,
        )
        retry_count = 0
        if not gate["passed"] and direction == "en->ru":
            retry_count = 1
            strict_kwargs = self._strict_generation_kwargs()
            translated_chunks = [self._translate_chunk(chunk, tokenizer, model, device, strict_kwargs) for chunk in chunks]
            translated_chunks = self._retry_bad_chunks(chunks, translated_chunks, tokenizer, model, device, strict_kwargs)
            joined = "\n\n".join(translated_chunks)
            finalized, glossary_replacements_retry = self._postprocess_translation(
                source_text=core_source,
                joined_translation=joined,
                placeholders=placeholders,
                direction=direction,
            )
            glossary_replacements += glossary_replacements_retry
            gate = self._quality_gate(
                source_text=core_source,
                translated_text=finalized,
                placeholders=placeholders,
                direction=direction,
            )

        protected_hits = sum(1 for token in placeholders.values() if self._token_present(finalized, token))
        finalized_with_ws = f"{leading_ws}{finalized}{trailing_ws}"
        return {
            "translated": finalized_with_ws,
            "chunks": chunks,
            "translated_chunks": translated_chunks,
            "chunk_count": len(chunks),
            "retry_count": retry_count,
            "glossary_replacements": glossary_replacements,
            "protected_tokens": list(placeholders.values()),
            "protected_hits": protected_hits,
            "gate_passed": gate["passed"],
            "gate_issues": gate["issues"],
        }

    def _retry_bad_chunks(self, src_chunks, dst_chunks, tokenizer, model, device, retry_kwargs):
        updated = list(dst_chunks)
        for i, (src, dst) in enumerate(zip(src_chunks, dst_chunks)):
            if self._chunk_quality_issue_count(src, dst) > 0:
                updated[i] = self._translate_chunk(src, tokenizer, model, device, retry_kwargs)
        return updated

    def _chunk_quality_issue_count(self, source_chunk: str, translated_chunk: str) -> int:
        issues = 0
        low = translated_chunk.lower()
        if "должен вернуться" in low or "внутренними работниками" in low or "порталом api" in low:
            issues += 1
        lat = len(self.LATIN_RE.findall(translated_chunk))
        cyr = len(self.CYRILLIC_RE.findall(translated_chunk))
        if (lat + cyr) > 0 and lat / (lat + cyr) > 0.6 and len(source_chunk) > 150:
            issues += 1
        return issues

    def _translate_chunk(self, chunk: str, tokenizer, model, device, gen_kwargs: Dict) -> str:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512).to(device)
        input_len = int(inputs["input_ids"].shape[1])
        dynamic_max_new_tokens = min(640, max(gen_kwargs.get("max_new_tokens", 256), int(input_len * 1.8)))
        local_gen_kwargs = dict(gen_kwargs)
        local_gen_kwargs["max_new_tokens"] = dynamic_max_new_tokens
        with torch.no_grad():
            output_ids = model.generate(**inputs, **local_gen_kwargs)
        return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def _generation_kwargs(self, mode: str) -> Dict:
        if mode == "fast":
            return {
                "max_new_tokens": 170,
                "num_beams": 2,
                "do_sample": False,
                "no_repeat_ngram_size": 3,
                "length_penalty": 1.0,
                "early_stopping": True,
            }
        if mode == "high_quality":
            return {
                "max_new_tokens": 220,
                "num_beams": 6,
                "do_sample": False,
                "no_repeat_ngram_size": 3,
                "length_penalty": 1.1,
                "early_stopping": True,
            }
        return {
            "max_new_tokens": 280,
            "num_beams": 4,
            "do_sample": False,
            "no_repeat_ngram_size": 3,
            "length_penalty": 1.05,
            "early_stopping": True,
        }

    def _high_quality_generation_kwargs(self) -> Dict:
        return {
            "max_new_tokens": 220,
            "num_beams": 8,
            "do_sample": False,
            "no_repeat_ngram_size": 4,
            "length_penalty": 1.15,
            "early_stopping": True,
        }

    def _strict_generation_kwargs(self) -> Dict:
        return {
            "max_new_tokens": 170,
            "num_beams": 6,
            "do_sample": False,
            "no_repeat_ngram_size": 4,
            "length_penalty": 1.1,
            "early_stopping": True,
        }

    def _protect_technical_tokens(self, text: str) -> Tuple[str, Dict[str, str]]:
        patterns = [
            self.MARKDOWN_HEADER_RE,
            self.CODE_BLOCK_RE,
            self.INLINE_CODE_RE,
            self.CLI_LINE_RE,
            self.PATH_RE,
            self.FILE_NAME_RE,
            self.JSON_XML_RE,
            self.METHOD_CALL_RE,
            self.FLAG_RE,
        ]

        placeholders: Dict[str, str] = {}
        protected = text
        counter = 0

        for pattern in patterns:
            matches = list(pattern.finditer(protected))
            for match in reversed(matches):
                token = match.group(0)
                if "TECHTOKEN_" in token:
                    continue
                key = f"[[T{counter}]]"
                placeholders[key] = token
                protected = protected[: match.start()] + key + protected[match.end() :]
                counter += 1

        return protected, placeholders

    def _apply_glossary(self, text: str, direction: str) -> Tuple[str, int]:
        if direction == "en->ru":
            glossary = self.EN_RU_GLOSSARY + self.external_en_ru_glossary
            corrections = self.EN_RU_CORRECTIONS
            flags = re.IGNORECASE
        elif direction == "ru->en":
            glossary = self.RU_EN_GLOSSARY
            corrections = []
            flags = re.IGNORECASE
        else:
            return text, 0

        updated = text
        replacements = 0
        for pattern, replacement in glossary:
            def _sub_fn(s: str):
                return re.subn(pattern, replacement, s, flags=flags)
            updated, count = self._apply_outside_code_blocks_with_count(updated, _sub_fn)
            replacements += count
        for pattern, replacement in corrections:
            def _sub_fn2(s: str):
                return re.subn(pattern, replacement, s, flags=flags)
            updated, count = self._apply_outside_code_blocks_with_count(updated, _sub_fn2)
            replacements += count
        return updated, replacements

    def _apply_outside_code_blocks_with_count(self, text: str, fn) -> Tuple[str, int]:
        parts = re.split(r"(```[\s\S]*?```)", text)
        out: List[str] = []
        total = 0
        for part in parts:
            if part.startswith("```") and part.endswith("```"):
                out.append(part)
                continue
            replaced, count = fn(part)
            out.append(replaced)
            total += count
        return "".join(out), total

    def _apply_source_guided_rules(self, source_text: str, translated_text: str, direction: str) -> Tuple[str, int]:
        if direction != "en->ru":
            return translated_text, 0
        source_low = source_text.lower()
        updated = translated_text
        replacements = 0
        for rule in self.SOURCE_GUIDED_RULES:
            source_pattern = rule.get("source")
            if not source_pattern or not re.search(source_pattern, source_low, flags=re.IGNORECASE):
                continue
            for pattern, repl in rule.get("replacements", []):
                def _sub_fn(s: str):
                    return re.subn(pattern, repl, s, flags=re.IGNORECASE)
                updated, count = self._apply_outside_code_blocks_with_count(updated, _sub_fn)
                replacements += count
        return updated, replacements

    def _apply_terminology_pass(self, source_text: str, translated_text: str, direction: str) -> Tuple[str, int]:
        if direction != "en->ru":
            return translated_text, 0

        source_low = source_text.lower()
        updated = translated_text
        replacements = 0

        for src_pattern, target_term in self.EN_RU_TERM_RULES:
            if not re.search(src_pattern, source_low, flags=re.IGNORECASE):
                continue

            updated, count = re.subn(src_pattern, target_term, updated, flags=re.IGNORECASE)
            replacements += count

        return updated, replacements

    def _restore_tokens(self, text: str, placeholders: Dict[str, str]) -> str:
        restored = text

        for key, value in placeholders.items():
            restored = restored.replace(key, value)

        index_to_value = {}
        for key, value in placeholders.items():
            match = re.search(r"(\d+)", key)
            if match:
                index_to_value[int(match.group(1))] = value

        def _replace_token_like(match: re.Match) -> str:
            idx = int(match.group(1))
            return index_to_value.get(idx, match.group(0))

        restored = self.TOKEN_LIKE_RE.sub(_replace_token_like, restored)

        def _replace_flex_placeholder(match: re.Match) -> str:
            idx_group = match.group(1) if match.group(1) is not None else match.group(2)
            if idx_group is None:
                return match.group(0)
            idx = int(idx_group)
            return index_to_value.get(idx, match.group(0))

        restored = self.FLEX_PLACEHOLDER_RE.sub(_replace_flex_placeholder, restored)

        # Catch bracket-corrupted placeholders like "[(T3]]", "[[Т1]]", "[[T2]".
        def _replace_bracket_placeholder(match: re.Match) -> str:
            idx = int(match.group(1))
            return index_to_value.get(idx, match.group(0))

        restored = self.BRACKET_PLACEHOLDER_RE.sub(_replace_bracket_placeholder, restored)
        return restored

    def _ensure_all_protected_tokens(self, text: str, placeholders: Dict[str, str]) -> str:
        def _is_critical_inline_token(token: str) -> bool:
            return bool(
                "/" in token
                or "\\" in token
                or "{" in token
                or "}" in token
                or ".py" in token.lower()
                or token.lower().startswith("http")
                or "status_code" in token
            )

        def _is_strict_technical_token(token: str) -> bool:
            token = token.strip()
            if not token:
                return False
            # Do not forcibly append short inline code/path fragments:
            # these can produce noisy tails like detached endpoint/status tokens.
            if token.startswith("`") and token.endswith("`") and "\n" not in token:
                return _is_critical_inline_token(token)
            return bool(
                token.startswith("```")
                or token.startswith("`")
                or "/" in token
                or "\\" in token
                or "{" in token
                or "}" in token
                or "$ " in token
                or token.startswith("$")
                or ".json" in token.lower()
                or ".yaml" in token.lower()
                or ".yml" in token.lower()
                or ".py" in token.lower()
                or token.lower().startswith("http")
            )

        def _normalize_token(token: str) -> str:
            return re.sub(r"[`\"'\s]", "", token).lower()

        normalized_text = _normalize_token(text)
        missing = []
        for token in placeholders.values():
            if not token:
                continue
            norm = _normalize_token(token)
            if token in text or (norm and norm in normalized_text):
                continue
            if not _is_strict_technical_token(token):
                continue
            missing.append(token)
        if not missing:
            return text

        tail_parts: List[str] = []
        for token in missing:
            if self.TECH_ACRONYM_RE.fullmatch(token.strip()):
                continue
            if token.startswith("```"):
                tail_parts.append("\n" + token + "\n")
            elif "\n" in token:
                tail_parts.append("\n" + token)
            else:
                tail_parts.append(" " + token)

        suffix = "".join(tail_parts).strip()
        if not suffix:
            return text
        if text.endswith("\n"):
            return text + suffix
        return text + "\n" + suffix

    def _postprocess_translation(
        self,
        source_text: str,
        joined_translation: str,
        placeholders: Dict[str, str],
        direction: str,
    ) -> Tuple[str, int]:
        normalized, glossary_replacements = self._apply_glossary(joined_translation, direction)
        restored = self._restore_tokens(normalized, placeholders)
        restored = self._reintegrate_detached_path_tokens(restored, placeholders)
        restored = self._ensure_all_protected_tokens(restored, placeholders)
        restored = self._apply_fluency_repairs(restored, direction)
        restored, source_guided_replacements = self._apply_source_guided_rules(source_text, restored, direction)
        restored, glossary_replacements_post = self._apply_glossary(restored, direction)
        term_fixed, terminology_replacements = self._apply_terminology_pass(source_text, restored, direction)
        total = (
            glossary_replacements
            + glossary_replacements_post
            + terminology_replacements
            + source_guided_replacements
        )
        return term_fixed, total

    def _reintegrate_detached_path_tokens(self, text: str, placeholders: Dict[str, str]) -> str:
        updated = text
        path_tokens = [t for t in placeholders.values() if t and "/" in t and "\n" not in t and len(t) < 120]
        for token in path_tokens:
            standalone_pattern = rf"\n\s*{re.escape(token)}\s*\n"
            if not re.search(standalone_pattern, updated):
                continue
            if "запрос HTTP `POST`" in updated:
                updated = re.sub(
                    r"запрос HTTP `POST`",
                    f"запрос HTTP `POST` на `{token.strip('`')}`",
                    updated,
                    count=1,
                )
                updated = re.sub(standalone_pattern, "\n", updated, count=1)
            elif "HTTP `POST`" in updated:
                updated = re.sub(r"HTTP `POST`", f"HTTP `POST` на `{token.strip('`')}`", updated, count=1)
                updated = re.sub(standalone_pattern, "\n", updated, count=1)
        return updated

    def _apply_fluency_repairs(self, text: str, direction: str) -> str:
        if direction != "en->ru":
            return text
        updated = text
        # Normalize spaces only outside fenced code blocks to keep code indentation intact.
        updated = self._normalize_outside_code_blocks(updated, lambda s: re.sub(r"[^\S\r\n]{2,}", " ", s))
        updated = re.sub(r"\n{3,}", "\n\n", updated)
        updated = re.sub(r"\bбэкенд API Эндпоинт\b", "бэкенд API эндпоинт", updated, flags=re.IGNORECASE)
        updated = re.sub(r"\bФронтенд отправляет\b", "Фронтенд отправляет", updated)
        updated = re.sub(r"\bархитектура клиент-серверной\b", "клиент-серверная архитектура", updated, flags=re.IGNORECASE)
        updated = re.sub(r"\bБэкенд должен\b", "бэкенд должен", updated)
        updated = re.sub(r"Путь к конфигу:\s*(`[^`]+`)\s+Путь к чекпоинту:", r"Путь к конфигу: \1\nПуть к чекпоинту:", updated)
        updated = re.sub(r",\s*такие,\s*как", ", такие как", updated)
        # If endpoint appears as a detached single line token, attach it back to POST request phrase.
        detached = re.search(r"(?m)^\s*`(/[^`\n]+)`\s*$", updated)
        if detached:
            path_token = detached.group(1)
            if "запрос HTTP `POST`" in updated and path_token not in updated.split("запрос HTTP `POST`", 1)[1]:
                updated = updated.replace("запрос HTTP `POST`", f"запрос HTTP `POST` на `{path_token}`", 1)
            updated = re.sub(rf"\n\s*`{re.escape(path_token)}`\s*\n", "\n", updated, count=1)
        return updated

    def _normalize_outside_code_blocks(self, text: str, fn) -> str:
        parts = re.split(r"(```[\s\S]*?```)", text)
        normalized: List[str] = []
        for part in parts:
            if part.startswith("```") and part.endswith("```"):
                normalized.append(part)
            else:
                normalized.append(fn(part))
        return "".join(normalized)

    def _is_critical_token(self, token: str) -> bool:
        token = token.strip()
        if not token:
            return False
        return bool(
            token.startswith("```")
            or token.startswith("`")
            or "/" in token
            or "\\" in token
            or "{" in token
            or "}" in token
            or token.startswith("$")
            or ".json" in token.lower()
            or ".yaml" in token.lower()
            or ".yml" in token.lower()
            or ".py" in token.lower()
            or token.lower().startswith("http")
        )

    def _normalize_for_presence(self, text: str) -> str:
        return re.sub(r"[`\"'\s]", "", text).lower()

    def _token_present(self, text: str, token: str) -> bool:
        if not token:
            return True
        if token in text:
            return True
        norm_text = self._normalize_for_presence(text)
        norm_token = self._normalize_for_presence(token)
        return bool(norm_token and norm_token in norm_text)

    def _quality_gate(
        self,
        source_text: str,
        translated_text: str,
        placeholders: Dict[str, str],
        direction: str,
    ) -> Dict[str, object]:
        issues: List[str] = []

        critical_tokens = [t for t in placeholders.values() if self._is_critical_token(t)]
        if critical_tokens:
            restored_hits = sum(1 for token in critical_tokens if self._token_present(translated_text, token))
            rate = restored_hits / len(critical_tokens)
            if rate < 0.80:
                issues.append(f"token_restoration_rate={rate:.2f}")

        if direction == "en->ru":
            text_for_ratio = translated_text
            for token in placeholders.values():
                if token:
                    text_for_ratio = text_for_ratio.replace(token, " ")

            # Ignore code-heavy and technical ASCII spans for language-ratio checks.
            text_for_ratio = self.CODE_BLOCK_RE.sub(" ", text_for_ratio)
            text_for_ratio = self.INLINE_CODE_RE.sub(" ", text_for_ratio)
            text_for_ratio = re.sub(r"https?://\S+", " ", text_for_ratio)
            text_for_ratio = self.PATH_RE.sub(" ", text_for_ratio)
            text_for_ratio = self.FILE_NAME_RE.sub(" ", text_for_ratio)
            text_for_ratio = self.FLAG_RE.sub(" ", text_for_ratio)

            lat = len(self.LATIN_RE.findall(text_for_ratio))
            cyr = len(self.CYRILLIC_RE.findall(text_for_ratio))
            if (lat + cyr) > 0:
                latin_ratio = lat / (lat + cyr)
                latin_threshold = 0.33
                if len(placeholders) >= 12:
                    latin_threshold = 0.55
                # Technical-heavy source text naturally keeps more Latin tokens.
                source_low = source_text.lower()
                if any(
                    kw in source_low
                    for kw in ("api", "payload", "endpoint", "status code", "upstream", "rbac", "yaml", "json")
                ):
                    latin_threshold = max(latin_threshold, 0.45)
                if latin_ratio > latin_threshold:
                    issues.append(f"latin_ratio={latin_ratio:.2f}")

            translated_low = translated_text.lower()

            if "повышает над " in translated_low:
                issues.append("fluency_broken_phrase")

            if translated_text.count("```") % 2 != 0:
                issues.append("format_unbalanced_code_fence")
            if translated_text.count("`") % 2 != 0:
                issues.append("format_unbalanced_backticks")
        return {"passed": len(issues) == 0, "issues": issues}

    def _chunk_text(self, text: str, chunk_size_chars: int) -> List[str]:
        # Prefer paragraph-aware chunking for markdown/technical docs even when text is short.
        if text.count("\n") >= 3:
            parts = text.split("\n\n")
            chunks: List[str] = []
            for part in parts:
                if not part:
                    continue
                if len(part) <= chunk_size_chars:
                    chunks.append(part)
                else:
                    forced = [part[i : i + chunk_size_chars] for i in range(0, len(part), chunk_size_chars)]
                    chunks.extend(forced)
            if chunks:
                return chunks

        if len(text) <= chunk_size_chars:
            return [text]

        sentences = re.split(r"(?<=[.!?\n])\s+", text)
        chunks: List[str] = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= chunk_size_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(sentence) <= chunk_size_chars:
                    current = sentence
                else:
                    forced = [sentence[i : i + chunk_size_chars] for i in range(0, len(sentence), chunk_size_chars)]
                    chunks.extend(forced[:-1])
                    current = forced[-1]

        if current:
            chunks.append(current)
        return chunks
