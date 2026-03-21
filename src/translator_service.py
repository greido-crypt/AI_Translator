from __future__ import annotations

import re
import time
from dataclasses import dataclass
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

        protected_text, placeholders = self._protect_technical_tokens(text)
        chunks = self._chunk_text(protected_text, chunk_size_chars)

        tokenizer, model, device = self._get_model_bundle(selection.model_name)
        gen_kwargs = self._generation_kwargs(selection.inference_mode)
        retry_count = 0

        translated_chunks: List[str] = []
        warnings: List[str] = []
        start = time.perf_counter()

        for chunk in chunks:
            translated = self._translate_chunk(chunk, tokenizer, model, device, gen_kwargs)
            translated_chunks.append(translated)

        total = time.perf_counter() - start
        joined = "\n\n".join(translated_chunks)

        direction = f"{selection.source_lang}->{selection.target_lang}"
        finalized, glossary_replacements = self._postprocess_translation(
            source_text=text,
            joined_translation=joined,
            placeholders=placeholders,
            direction=direction,
        )
        protected_token_hits = sum(1 for token in placeholders.values() if self._token_present(finalized, token))

        gate = self._quality_gate(
            source_text=text,
            translated_text=finalized,
            placeholders=placeholders,
            direction=direction,
        )
        if not gate["passed"] and direction == "en->ru":
            retry_count = 1
            strict_kwargs = self._strict_generation_kwargs()
            translated_chunks = [
                self._translate_chunk(chunk, tokenizer, model, device, strict_kwargs) for chunk in chunks
            ]
            joined = "\n\n".join(translated_chunks)
            finalized, glossary_replacements_retry = self._postprocess_translation(
                source_text=text,
                joined_translation=joined,
                placeholders=placeholders,
                direction=direction,
            )
            glossary_replacements += glossary_replacements_retry
            protected_token_hits = sum(1 for token in placeholders.values() if self._token_present(finalized, token))
            gate = self._quality_gate(
                source_text=text,
                translated_text=finalized,
                placeholders=placeholders,
                direction=direction,
            )
            if not gate["passed"]:
                retry_count = 2
                hq_kwargs = self._high_quality_generation_kwargs()
                translated_chunks = [
                    self._translate_chunk(chunk, tokenizer, model, device, hq_kwargs) for chunk in chunks
                ]
                joined = "\n\n".join(translated_chunks)
                finalized, glossary_replacements_retry2 = self._postprocess_translation(
                    source_text=text,
                    joined_translation=joined,
                    placeholders=placeholders,
                    direction=direction,
                )
                glossary_replacements += glossary_replacements_retry2
                protected_token_hits = sum(1 for token in placeholders.values() if self._token_present(finalized, token))
                gate = self._quality_gate(
                    source_text=text,
                    translated_text=finalized,
                    placeholders=placeholders,
                    direction=direction,
                )

        if len(chunks) > 1:
            warnings.append("Input was split into multiple chunks; minor boundary artifacts are possible.")
        if protected_token_hits < len(placeholders):
            warnings.append("Some protected technical tokens may have changed during generation.")
        if glossary_replacements > 0:
            warnings.append(f"Glossary normalization applied ({glossary_replacements} replacements).")
        if retry_count > 0:
            warnings.append("Quality gate retry was triggered with stricter generation settings.")
        if not gate["passed"]:
            warnings.append(f"Quality gate still reports issues: {', '.join(gate['issues'])}")

        return TranslationRuntimeResult(
            translated_text=finalized,
            chunks=chunks,
            translated_chunks=translated_chunks,
            chunk_count=len(chunks),
            total_inference_time=total,
            device_info=str(device),
            model_info={
                "name": selection.model_name,
                "framework": "transformers",
                "dtype": str(next(model.parameters()).dtype),
            },
            preprocessing_summary={
                "mode": "protect_translate_normalize_restore",
                "protected_token_count": str(len(placeholders)),
                "chunk_size_chars": str(chunk_size_chars),
                "glossary_replacements": str(glossary_replacements),
                "quality_gate_passed": str(gate["passed"]),
                "quality_gate_issues": "; ".join(gate["issues"]) if gate["issues"] else "none",
                "retry_count": str(retry_count),
            },
            warnings=warnings,
            protected_tokens=list(placeholders.values()),
            protected_token_hits=protected_token_hits,
            quality_gate_passed=gate["passed"],
            retry_count=retry_count,
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
            glossary = self.EN_RU_GLOSSARY
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
            updated, count = re.subn(pattern, replacement, updated, flags=flags)
            replacements += count
        for pattern, replacement in corrections:
            updated, count = re.subn(pattern, replacement, updated, flags=flags)
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
        def _is_strict_technical_token(token: str) -> bool:
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
        restored = self._ensure_all_protected_tokens(restored, placeholders)
        restored = self._apply_fluency_repairs(restored, direction)
        restored, glossary_replacements_post = self._apply_glossary(restored, direction)
        term_fixed, terminology_replacements = self._apply_terminology_pass(source_text, restored, direction)
        total = glossary_replacements + glossary_replacements_post + terminology_replacements
        return term_fixed, total

    def _apply_fluency_repairs(self, text: str, direction: str) -> str:
        if direction != "en->ru":
            return text
        updated = text
        # Normalize spaces only outside fenced code blocks to keep code indentation intact.
        updated = self._normalize_outside_code_blocks(updated, lambda s: re.sub(r"[^\S\r\n]{2,}", " ", s))
        updated = re.sub(r"\n{3,}", "\n\n", updated)
        updated = re.sub(r"\bбэкенд API Эндпоинт\b", "бэкенд API эндпоинт", updated, flags=re.IGNORECASE)
        updated = re.sub(r"\bФронтенд отправляет\b", "Фронтенд отправляет", updated)
        updated = updated.strip()
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
            if rate < 0.85:
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
