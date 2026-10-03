"""Guardrail detectors.

Two stages, deliberately separated:

* ``input``  rules see untrusted user text and decide whether the request may run
  at all. They run before any token is spent on retrieval or routing.
* ``output`` rules see model text and decide whether it may leave the process.

Every rule carries an ``action`` (block / redact / flag / rewrite) and a
``severity``, so a pipeline can be configured per deployment without editing
detector code.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from re import Pattern


class Action(StrEnum):
    BLOCK = "block"
    REDACT = "redact"
    FLAG = "flag"
    REWRITE = "rewrite"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(slots=True)
class Finding:
    """One rule hit against one piece of text."""

    rule: str
    stage: str
    action: Action
    severity: Severity
    message: str
    evidence: str = ""
    span: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "rule": self.rule,
            "stage": self.stage,
            "action": str(self.action),
            "severity": str(self.severity),
            "message": self.message,
            "evidence": self.evidence,
        }


@dataclass(slots=True)
class Detector:
    """Declarative guardrail rule."""

    name: str
    stage: str
    action: Action
    severity: Severity
    pattern: Pattern[str] | None
    message: str
    checker: Callable[[str], bool] | None = None
    group: int = 0

    def run(self, text: str) -> Finding | None:
        if self.pattern is not None:
            match = self.pattern.search(text)
            if not match:
                return None
            return Finding(
                rule=self.name,
                stage=self.stage,
                action=self.action,
                severity=self.severity,
                message=self.message,
                evidence=match.group(0)[:120],
                span=match.span(),
            )
        if self.checker is not None and self.checker(text):
            return Finding(
                rule=self.name,
                stage=self.stage,
                action=self.action,
                severity=self.severity,
                message=self.message,
            )
        return None


def _word_set(words: Iterable[str]) -> Pattern[str]:
    terms = sorted({w.strip().lower() for w in words if w.strip()}, key=len, reverse=True)
    if not terms:
        return re.compile(r"(?!x)x")
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE)


# -- input-stage rules -----------------------------------------------------
INPUT_RULES: list[Detector] = [
    Detector(
        name="prompt_injection",
        stage="input",
        action=Action.BLOCK,
        severity=Severity.CRITICAL,
        pattern=re.compile(
            # "ignore your instructions" ... but also the bare overrides that carry
            # no noun at all: "disregard the above", "forget everything".
            r"(?:ignore|disregard|forget|override)\s+(?:all\s+|any\s+|the\s+)?"
            r"(?:previous\s+|prior\s+|above\s+|earlier\s+|preceding\s+|foregoing\s+)?"
            r"(?:instruction|rule|prompt|direction|constraint|guideline)s?"
            r"|(?:ignore|disregard|forget)\s+(?:everything|anything|all)\b"
            r"|disregard\s+(?:the\s+)?(?:above|prior|previous|preceding)\b",
            re.IGNORECASE,
        ),
        message="prompt-injection attempt: instruction override detected",
    ),
    Detector(
        name="system_prompt_exfiltration",
        stage="input",
        action=Action.BLOCK,
        severity=Severity.HIGH,
        pattern=re.compile(
            r"(?:reveal|show|print|repeat|output)\s+(?:me\s+)?(?:your\s+|the\s+)?"
            r"(?:system\s+prompt|initial\s+instructions|hidden\s+instructions)",
            re.IGNORECASE,
        ),
        message="attempt to exfiltrate system instructions",
    ),
    Detector(
        name="role_hijack",
        stage="input",
        action=Action.BLOCK,
        severity=Severity.HIGH,
        pattern=re.compile(
            # "you are now DAN", "you are now in developer mode", "act as DAN" —
            # the optional preposition and missing "if you" were both easy misses.
            r"(?:you\s+are\s+now|from\s+now\s+on[,\s]+you)\s+"
            r"(?:in\s+|as\s+)?(?:a\s+|an\s+)?"
            r"(?:unrestricted|unfiltered|jailbroken|developer\s+mode|dan\b|evil\s+mode)"
            r"|act\s+as\s+(?:if\s+you\s+are\s+)?(?:a\s+|an\s+)?"
            r"(?:unrestricted|unfiltered|jailbroken|developer\s+mode|dan\b)",
            re.IGNORECASE,
        ),
        message="role-hijack / jailbreak framing",
    ),
    Detector(
        name="encoded_payload",
        stage="input",
        action=Action.FLAG,
        severity=Severity.MEDIUM,
        pattern=re.compile(r"(?:base64|rot13|hex)\s*(?:decode|encoded|payload)", re.IGNORECASE),
        message="possible encoded instruction payload",
    ),
    Detector(
        name="exfiltration_destination",
        stage="input",
        action=Action.BLOCK,
        severity=Severity.CRITICAL,
        pattern=re.compile(
            r"https?://[^\s]*(?:\bwebhook\.site\b|requestbin|pipedream\.net|ngrok\.io|"
            r"burpcollaborator|interact\.sh)",
            re.IGNORECASE,
        ),
        message="request to forward data to an exfiltration endpoint",
    ),
    Detector(
        name="destructive_sql",
        stage="input",
        action=Action.BLOCK,
        severity=Severity.HIGH,
        # An unbounded statement: no WHERE clause, or a WHERE that is always true.
        # `DELETE FROM users WHERE id = 42` is legitimate and stays allowed.
        pattern=re.compile(
            r"\b(?:drop\s+(?:table|database|schema|index)|truncate\s+table|"
            r"delete\s+from\s+[\w.\"'\[\]]+\s*(?:;|$)|"
            r"delete\s+from\s+[\w.\"'\[\]]+\s+where\s+(?:1\s*=\s*1|true)\b|"
            r"update\s+[\w.\"'\[\]]+\s+set\s+[^;]*?(?:;|$)|"
            r"update\s+[\w.\"'\[\]]+\s+set\s+\w+\s*=\s*\w+\s*;\s*drop)",
            re.IGNORECASE,
        ),
        message="destructive SQL statement requested",
    ),
    Detector(
        name="secret_handling",
        stage="input",
        action=Action.REDACT,
        severity=Severity.HIGH,
        pattern=re.compile(
            r"\b(?:give|show|print|leak|export|dump)\s+(?:me\s+)?(?:the\s+)?"
            r"(?:api\s*keys?|passwords?|private\s+keys?|secrets?|credentials?)\b",
            re.IGNORECASE,
        ),
        message="request for secret material",
    ),
]

# -- output-stage rules ----------------------------------------------------
OUTPUT_RULES: list[Detector] = [
    Detector(
        name="leaked_credential",
        stage="output",
        action=Action.BLOCK,
        severity=Severity.CRITICAL,
        pattern=re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
        message="model output contained an AWS access key id",
    ),
    Detector(
        name="leaked_api_key",
        stage="output",
        action=Action.BLOCK,
        severity=Severity.CRITICAL,
        pattern=re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{16,}\b"),
        message="model output contained an Anthropic API key",
    ),
    Detector(
        name="citation_fabrication",
        stage="output",
        action=Action.FLAG,
        severity=Severity.LOW,
        pattern=re.compile(r"\[(?:source|citation)\s*\d+\]"),
        message="output cites a source that may not be in the retrieved context",
    ),
    Detector(
        name="unbounded_refusal",
        stage="output",
        action=Action.REWRITE,
        severity=Severity.MEDIUM,
        pattern=re.compile(r"\b(?:i\s+can(?:'|no)t\s+(?:help|assist)|i\s+cannot\s+comply)\b", re.I),
        message="model refused; substitute a safe, grounded answer",
        group=1,
    ),
]


def blocked_topic_rules(topics: Iterable[str]) -> list[Detector]:
    """Build a topic blocklist rule set from configuration."""
    pattern = _word_set(topics)
    return [
        Detector(
            name="blocked_topic",
            stage="input",
            action=Action.BLOCK,
            severity=Severity.HIGH,
            pattern=pattern,
            message="request touches a configured blocked topic",
        )
    ]


@dataclass(slots=True)
class RuleSet:
    """Named collection of detectors, split by stage."""

    name: str
    detectors: list[Detector] = field(default_factory=list)

    def extend(self, detectors: Iterable[Detector]) -> RuleSet:
        self.detectors.extend(detectors)
        return self

    def for_stage(self, stage: str) -> list[Detector]:
        return [d for d in self.detectors if d.stage == stage]

    def __len__(self) -> int:
        return len(self.detectors)

    @property
    def names(self) -> list[str]:
        return [d.name for d in self.detectors]


DEFAULT_RULESET = RuleSet("default", INPUT_RULES + OUTPUT_RULES)

__all__ = [
    "DEFAULT_RULESET",
    "INPUT_RULES",
    "OUTPUT_RULES",
    "Action",
    "Detector",
    "Finding",
    "RuleSet",
    "Severity",
    "blocked_topic_rules",
]