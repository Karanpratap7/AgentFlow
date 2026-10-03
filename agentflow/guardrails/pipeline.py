"""The guardrail pipeline.

Wraps every model call:

    user input
        -> input rules (block / redact / flag)
        -> model + tools
        -> output rules (block / redact / rewrite / flag)
        -> redactor (final scrub)
        -> caller

Configurable behaviour lives here rather than in the detectors:
``fail_closed`` turns detector errors into blocks, ``audit`` collects every
finding, and ``on_rewrite`` supplies the safe substitute text used when a model
refuses or an output rule trips.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from agentflow.config import Settings
from agentflow.config import settings as default_settings
from agentflow.errors import GuardrailViolation
from agentflow.guardrails.detectors import (
    DEFAULT_RULESET,
    Action,
    Detector,
    Finding,
    RuleSet,
    blocked_topic_rules,
)
from agentflow.guardrails.redaction import RedactionResult, Redactor
from agentflow.llm.types import Message
from agentflow.telemetry import metrics

log = logging.getLogger("agentflow.guardrails")

SAFE_FALLBACK = (
    "I can't help with that request. If it was a mistake, rephrase it and I'll try again."
)


@dataclass(slots=True)
class GuardrailResult:
    """Outcome of running the pipeline over one request/response pair."""

    text: str
    findings: list[Finding] = field(default_factory=list)
    redaction: RedactionResult | None = None
    blocked: bool = False
    rewritten: bool = False
    flagged: list[str] = field(default_factory=list)

    @property
    def highest_severity(self) -> str | None:
        order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        if not self.findings:
            return None
        return max(self.findings, key=lambda f: order.get(str(f.severity), 0)).severity

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocked": self.blocked,
            "rewritten": self.rewritten,
            "flagged": list(self.flagged),
            "highest_severity": self.highest_severity,
            "findings": [f.to_dict() for f in self.findings],
            "redaction": self.redaction.to_dict() if self.redaction else None,
        }


class GuardrailPipeline:
    """Ordered rule executor with redaction, blocking and auditing."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        rules: RuleSet | None = None,
        redactor: Redactor | None = None,
        on_rewrite: Callable[[str], str] | None = None,
    ) -> None:
        self.settings = settings or default_settings
        self.enabled = self.settings.guardrails_enabled
        self.fail_closed = self.settings.guardrails_fail_closed
        # ``is None``, not ``or``: RuleSet defines __len__, so an explicitly empty
        # ruleset is falsy and ``rules or DEFAULT_RULESET`` would silently re-arm
        # every detector, quietly disabling the caller's configuration.
        self.rules = rules if rules is not None else DEFAULT_RULESET
        self.redactor = redactor if redactor is not None else Redactor()
        self.use_redaction = self.settings.pii_redaction
        self.on_rewrite = on_rewrite or (lambda _text: SAFE_FALLBACK)

        if self.settings.blocked_topics:
            self.rules.extend(blocked_topic_rules(self.settings.blocked_topics))

        self.audit: list[Finding] = []

    # -- public API -------------------------------------------------------
    def check_input(self, text: str, *, raise_on_block: bool = True) -> GuardrailResult:
        """Run the input stage. Raises when the request must not proceed."""
        if not self.enabled:
            return GuardrailResult(text=text)

        text, findings = self._apply(self.rules.for_stage("input"), text, stage="input")
        redaction: RedactionResult | None = None
        if self.use_redaction:
            redaction = self.redactor.redact(text)
            text = redaction.text

        blocked = any(f.action is Action.BLOCK for f in findings)
        result = GuardrailResult(
            text=text,
            findings=findings,
            redaction=redaction,
            blocked=blocked,
            flagged=[f.rule for f in findings if f.action is Action.FLAG],
        )
        self.audit.extend(findings)
        metrics.incr("guardrail_checks_total", stage="input")
        if blocked:
            metrics.incr("guardrail_blocks_total", stage="input")
        elif any(f.action is Action.FLAG for f in findings):
            metrics.incr("guardrail_flags_total", stage="input")

        if blocked and raise_on_block:
            first = next(f for f in findings if f.action is Action.BLOCK)
            raise GuardrailViolation(first.rule, first.message, stage="input")
        return result

    def check_output(self, text: str, *, context: str = "") -> GuardrailResult:
        """Run the output stage, including the final redaction scrub."""
        if not self.enabled:
            return GuardrailResult(text=text)

        text, findings = self._apply(self.rules.for_stage("output"), text, stage="output")
        redaction: RedactionResult | None = None
        if self.use_redaction:
            redaction = self.redactor.redact(text)
            text = redaction.text

        blocked = any(f.action is Action.BLOCK for f in findings)
        rewritten = False
        if any(f.action is Action.REWRITE for f in findings):
            replacement = self.on_rewrite(context or text)
            log.info("output rewritten by guardrail: %s", [f.rule for f in findings])
            text, rewritten = replacement, True

        result = GuardrailResult(
            text=text,
            findings=findings,
            redaction=redaction,
            blocked=blocked,
            rewritten=rewritten,
            flagged=[f.rule for f in findings if f.action is Action.FLAG],
        )
        self.audit.extend(findings)
        metrics.incr("guardrail_checks_total", stage="output")
        if blocked:
            metrics.incr("guardrail_blocks_total", stage="output")
        if rewritten:
            metrics.incr("guardrail_rewrites_total")

        if blocked:
            first = next(f for f in findings if f.action is Action.BLOCK)
            raise GuardrailViolation(first.rule, first.message, stage="output")
        return result

    def guard_messages(self, messages: list[Message]) -> list[Message]:
        """Run the input stage over every user/tool turn in a message list."""
        out: list[Message] = []
        for message in messages:
            if message.role in {"user", "system"} and isinstance(message.content, str):
                out.append(Message(message.role, self.check_input(message.content).text))
            else:
                out.append(message)
        return out

    # -- internals --------------------------------------------------------
    def _apply(
        self, detectors: Iterable[Detector], text: str, *, stage: str
    ) -> tuple[str, list[Finding]]:
        findings: list[Finding] = []
        out = text
        for detector in detectors:
            try:
                finding = detector.run(out)
            except Exception as exc:  # noqa: BLE001 - a broken rule must not be silent
                log.exception("guardrail %s crashed", detector.name)
                if self.fail_closed:
                    raise GuardrailViolation(
                        detector.name, f"guardrail engine error: {exc}", stage=stage
                    ) from exc
                continue
            if finding is None:
                continue
            findings.append(finding)
            if finding.action is Action.REWRITE:
                out = self.on_rewrite(out)
        return out, findings

    # -- reporting --------------------------------------------------------
    def audit_report(self) -> dict[str, Any]:
        tally: dict[str, int] = {}
        for finding in self.audit:
            tally[finding.rule] = tally.get(finding.rule, 0) + 1
        return {
            "enabled": self.enabled,
            "rules": self.rules.names,
            "rule_count": len(self.rules),
            "total_findings": len(self.audit),
            "by_rule": dict(sorted(tally.items(), key=lambda kv: -kv[1])),
        }


__all__ = ["DEFAULT_RULESET", "GuardrailPipeline", "GuardrailResult", "SAFE_FALLBACK"]