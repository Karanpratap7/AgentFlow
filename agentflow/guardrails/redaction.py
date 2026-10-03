"""Redaction of sensitive spans before anything reaches a model provider.

Redaction happens *before* the prompt leaves the process, so secrets never enter
the provider's logs or the cache. Each detector returns
``[(pattern, replacement, kind)]``; the redactor applies them left-to-right and
reports what it masked.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from re import Pattern

Detectors = list[tuple[str, Pattern[str], Callable[[re.Match[str]], str]]]


@dataclass(slots=True)
class Redaction:
    kind: str
    count: int
    placeholder: str

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "count": self.count, "placeholder": self.placeholder}


def _mask_keep_prefix(match: re.Match[str], keep: int = 4) -> str:
    value = match.group(0)
    return f"{value[:keep]}{'*' * max(4, len(value) - keep)}"


DETECTORS: Detectors = [
    (
        "aws_access_key",
        re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
        lambda m: _mask_keep_prefix(m),
    ),
    (
        "anthropic_api_key",
        re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{16,}\b"),
        lambda m: _mask_keep_prefix(m, 8),
    ),
    (
        "openai_api_key",
        re.compile(r"\bsk-(?!ant-)[A-Za-z0-9]{20,}\b"),
        lambda m: _mask_keep_prefix(m, 7),
    ),
    (
        "github_token",
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b"),
        lambda m: _mask_keep_prefix(m, 7),
    ),
    (
        "private_key",
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----"),
        lambda m: "[REDACTED:private_key]",
    ),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b"),
        lambda m: _mask_keep_prefix(m, 3),
    ),
    ("email", re.compile(r"\b[\w.+\-]+@[\w\-]+\.[\w.\-]{2,}\b"), lambda m: _mask_keep_prefix(m, 2)),
    (
        "credit_card",
        re.compile(r"\b(?:\d[ \-]?){13,19}\b"),
        lambda m: "[REDACTED:credit_card]",
    ),
    ("ssn_us", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), lambda m: "[REDACTED:ssn_us]"),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"), lambda m: "[REDACTED:iban]"),
    (
        "phone_in",
        re.compile(r"(?<![\d+])(?:\+?91[\s\-]?)?[6-9]\d{4}[\s\-]?\d{5}\b"),
        lambda m: "[REDACTED:phone_in]",
    ),
]


@dataclass(slots=True)
class RedactionResult:
    text: str
    redactions: list[Redaction] = field(default_factory=list)

    @property
    def redacted(self) -> bool:
        return bool(self.redactions)

    @property
    def total(self) -> int:
        return sum(r.count for r in self.redactions)

    def kinds(self) -> list[str]:
        return sorted({r.kind for r in self.redactions})

    def to_dict(self) -> dict[str, object]:
        return {
            "redacted": self.redacted,
            "total": self.total,
            "kinds": self.kinds(),
            "findings": [r.to_dict() for r in self.redactions],
        }


class Redactor:
    """Applies every detector and tallies what it changed."""

    def __init__(self, detectors: Detectors | None = None) -> None:
        self.detectors = detectors or DETECTORS

    def redact(self, text: str) -> RedactionResult:
        if not text:
            return RedactionResult(text=text)

        findings: dict[str, int] = {}
        placeholder: dict[str, str] = {}
        out = text
        for kind, pattern, replace in self.detectors:
            replaced, count = pattern.subn(replace, out)
            if count:
                out = replaced
                findings[kind] = findings.get(kind, 0) + count
                placeholder[kind] = f"[REDACTED:{kind}]"

        return RedactionResult(
            text=out,
            redactions=[
                Redaction(kind=kind, count=count, placeholder=placeholder[kind])
                for kind, count in sorted(findings.items())
            ],
        )

    def redact_iterable(self, values: Iterable[str]) -> list[RedactionResult]:
        return [self.redact(v) for v in values]


__all__ = ["DETECTORS", "Redaction", "RedactionResult", "Redactor"]