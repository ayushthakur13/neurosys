from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class ParsedLog:
    raw_line: str
    event_template: str
    event_id: str


class SimpleDrainParser:
    """A light-weight Drain-like parser for template extraction.

    This parser groups lines by token count and uses wildcard tokens for high-variance positions.
    """

    number_re = re.compile(r"^-?\d+(\.\d+)?$")
    hex_re = re.compile(r"^(0x)?[0-9a-fA-F]+$")
    ip_re = re.compile(r"^\d+\.\d+\.\d+\.\d+$")

    def __init__(self, wildcard: str = "<*>"):
        self.wildcard = wildcard
        self.templates_by_len: dict[int, list[list[str]]] = defaultdict(list)
        self.template_ids: dict[str, str] = {}

    def _normalize_token(self, token: str) -> str:
        if self.number_re.match(token) or self.hex_re.match(token) or self.ip_re.match(token):
            return self.wildcard
        if re.search(r"blk_-?\d+", token):
            return self.wildcard
        if len(token) > 16 and any(ch.isdigit() for ch in token):
            return self.wildcard
        return token

    def _similarity(self, a: list[str], b: list[str]) -> float:
        match = sum(1 for x, y in zip(a, b) if x == y)
        return match / max(len(a), 1)

    def _to_template(self, tokens: list[str], base: list[str]) -> list[str]:
        return [x if x == y else self.wildcard for x, y in zip(tokens, base)]

    def parse_line(self, line: str) -> ParsedLog:
        tokens = [self._normalize_token(t) for t in line.strip().split()]
        candidates = self.templates_by_len[len(tokens)]

        if not candidates:
            template = tokens
            self.templates_by_len[len(tokens)].append(template)
        else:
            sims = [self._similarity(tokens, t) for t in candidates]
            best_idx = int(max(range(len(sims)), key=lambda i: sims[i]))
            if sims[best_idx] >= 0.5:
                candidates[best_idx] = self._to_template(tokens, candidates[best_idx])
                template = candidates[best_idx]
            else:
                template = tokens
                candidates.append(template)

        template_str = " ".join(template)
        if template_str not in self.template_ids:
            self.template_ids[template_str] = f"E{len(self.template_ids) + 1}"

        return ParsedLog(raw_line=line.rstrip("\n"), event_template=template_str, event_id=self.template_ids[template_str])
