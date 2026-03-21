from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class TextAnalysis:
    char_count: int
    token_count_estimate: int
    technical_density: float
    has_markdown: bool
    has_code_blocks: bool
    has_inline_code: bool
    has_cli_commands: bool
    has_paths: bool
    has_api_methods: bool
    has_json_yaml_xml: bool
    preserved_candidates: List[str]


class TextAnalyzer:
    CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
    INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
    CLI_RE = re.compile(r"(?m)^\s*(?:\$|>|#)\s+.+$")
    PATH_RE = re.compile(r"(?:[A-Za-z]:\\[^\s]+|/(?:[\w.-]+/)+[\w.-]+|\./[\w./-]+)")
    API_RE = re.compile(r"\b[a-zA-Z_][\w]*\.[a-zA-Z_][\w]*\([^\)]*\)")
    JSON_YAML_XML_RE = re.compile(r"(?s)(\{\s*\".+\"\s*:.*\}|<\?xml|<\w+>|^\s*\w+\s*:\s*.+$)", re.MULTILINE)
    MARKDOWN_RE = re.compile(r"(?m)^(#{1,6}\s|[-*+]\s|\d+\.\s|>\s)|\*\*|__|\[[^\]]+\]\([^\)]+\)")
    TECH_TOKEN_RE = re.compile(r"\b(?:[A-Za-z_][\w.-]{2,}|--?[\w-]+|\w+/\w+|\w+\.\w+|\w+::\w+)\b")

    def analyze(self, text: str) -> TextAnalysis:
        text = text or ""
        char_count = len(text)
        token_count_estimate = max(1, int(char_count / 4))

        code_blocks = self.CODE_BLOCK_RE.findall(text)
        inline_code = self.INLINE_CODE_RE.findall(text)
        cli = self.CLI_RE.findall(text)
        paths = self.PATH_RE.findall(text)
        api = self.API_RE.findall(text)
        struct = self.JSON_YAML_XML_RE.findall(text)

        tech_tokens = self.TECH_TOKEN_RE.findall(text)
        technical_density = min(1.0, len(tech_tokens) / max(1, token_count_estimate))

        preserved_candidates = list(dict.fromkeys(code_blocks + inline_code + cli + paths + api))

        return TextAnalysis(
            char_count=char_count,
            token_count_estimate=token_count_estimate,
            technical_density=technical_density,
            has_markdown=bool(self.MARKDOWN_RE.search(text)),
            has_code_blocks=bool(code_blocks),
            has_inline_code=bool(inline_code),
            has_cli_commands=bool(cli),
            has_paths=bool(paths),
            has_api_methods=bool(api),
            has_json_yaml_xml=bool(struct),
            preserved_candidates=preserved_candidates,
        )

    def preprocessing_summary(self, analysis: TextAnalysis) -> Dict[str, str]:
        return {
            "char_count": str(analysis.char_count),
            "token_count_estimate": str(analysis.token_count_estimate),
            "technical_density": f"{analysis.technical_density:.3f}",
            "markdown": str(analysis.has_markdown),
            "code_blocks": str(analysis.has_code_blocks),
            "inline_code": str(analysis.has_inline_code),
            "cli_commands": str(analysis.has_cli_commands),
            "paths": str(analysis.has_paths),
            "api_methods": str(analysis.has_api_methods),
            "json_yaml_xml": str(analysis.has_json_yaml_xml),
        }
