"""Automatic intent detection.

A three-signal cascade, cheapest first, so that classification cost tracks
difficulty:

1. ``rules``  - curated regex patterns. Free, microsecond-fast, and exact for
   the intents that matter most (unsafe requests, greetings, SQL).
2. ``lexical``- bag-of-words scoring against per-intent prototype term lists with
   IDF weighting. Free, and generalises to phrasings the rules miss.
3. ``llm``    - only asked when the top two signals are within
   ``margin`` of each other, i.e. only for genuinely ambiguous input. This is
   what keeps intent detection cheap: the expensive model is invoked for a small
   fraction of traffic.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from agentflow.llm.types import LLMClient, Message


class Intent(StrEnum):
    GREETING = "greeting"
    CHITCHAT = "chitchat"
    KNOWLEDGE = "knowledge"
    CODE = "code"
    SQL = "sql"
    SUMMARIZE = "summarize"
    ANALYSIS = "analysis"
    TRANSLATION = "translation"
    MATH = "math"
    SAFETY_RISK = "safety_risk"
    UNKNOWN = "unknown"


#: Cheap, high-precision rules. Checked first.
RULES: dict[Intent, re.Pattern[str]] = {
    Intent.SAFETY_RISK: re.compile(
        r"\b(?:ignore (?:all |your )?(?:previous |prior )?instructions|"
        r"jailbreak|system prompt|api[_ ]?key|password|private key|"
        r"drop table|rm -rf|eval\(|exec\()\b",
        re.IGNORECASE,
    ),
    Intent.GREETING: re.compile(
        r"^\s*(?:hi|hey|hello|yo|namaste|good\s+(?:morning|afternoon|evening))\b",
        re.IGNORECASE,
    ),
    Intent.SQL: re.compile(
        r"\b(?:select|insert|update|delete)\b.*\b(?:from|into|set|where|table)\b|"
        r"\b(?:sql|query|database|postgres|mysql|mongo)\b",
        re.IGNORECASE,
    ),
    Intent.CODE: re.compile(
        r"\b(?:refactor|debug|stack ?trace|traceback|function|method|class|"
        r"compile|unit test|pytest|jest|npm|dockerfile|regex|algorithm|"
        r"big[- ]o|complexity|deadlock|race condition|memory leak|"
        r"crash|exception|null ?pointer|segfault|stack ?overflow|"
        r"\d{3,5}\s*(?:error|exception)|type ?error|undefined is not|"
        r"endpoint|api|service|deploy|regression|flaky|hangs?|"
        r"returning\s+[45]\d\d\b|[45]\d\d\s*(?:error|status|response))\b",
        re.IGNORECASE,
    ),
    Intent.SUMMARIZE: re.compile(
        r"\b(?:summari[sz]e|tl;?dr|condense|shorten|bullet points?|key takeaways?)\b",
        re.IGNORECASE,
    ),
    Intent.TRANSLATION: re.compile(
        r"\b(?:translate|translation|locali[sz]e|in (?:hindi|spanish|french|german|japanese))\b",
        re.IGNORECASE,
    ),
    Intent.MATH: re.compile(r"[0-9]\s*[-+*/^]\s*[0-9]|\b(?:integral|derivative|probability)\b"),
    Intent.ANALYSIS: re.compile(
        r"\b(?:analy[sz]e|analysis|compare|trade-?offs?|why does|root cause|"
        r"evaluate|pros and cons|metrics?|benchmark|forecast)\b",
        re.IGNORECASE,
    ),
    Intent.KNOWLEDGE: re.compile(
        r"\b(?:what is|what are|what does|what are the|how does|how do|how did|"
        r"why is|why are|why does|why do|explain|define|when did|who is|"
        r"documentation|docs|policy|runbook|guide|according to|"
        r"what'?s the|where is|tell me about)\b",
        re.IGNORECASE,
    ),
    Intent.CHITCHAT: re.compile(
        r"\b(?:how are you|thanks|thank you|good morning|bye|goodbye)\b", re.IGNORECASE
    ),
}

#: Prototype terms per intent for the lexical stage.
PROTOTYPES: dict[Intent, set[str]] = {
    Intent.CODE: {
        "refactor", "debug", "function", "class", "test", "compile", "stacktrace",
        "repository", "module", "async", "await", "thread", "process", "memory",
        "leak", "deadlock", "race", "typescript", "python", "javascript", "lint",
    },
    Intent.KNOWLEDGE: {
        "what", "explain", "define", "documentation", "concept", "difference",
        "how", "works", "meaning", "overview", "introduction", "guide", "policy",
        "runbook", "architecture", "protocol", "standard",
    },
    Intent.ANALYSIS: {
        "analyze", "compare", "tradeoff", "evaluate", "metric", "latency",
        "throughput", "cost", "root", "cause", "hypothesis", "experiment",
        "benchmark", "profile", "bottleneck", "trend",
    },
    Intent.SUMMARIZE: {"summarize", "shorten", "condense", "digest", "tldr", "bullet", "recap"},
    Intent.SQL: {"sql", "query", "select", "table", "index", "schema", "join", "postgres", "mysql"},
    Intent.MATH: {"calculate", "solve", "equation", "probability", "integral", "sum", "product"},
    Intent.TRANSLATION: {"translate", "language", "spanish", "hindi", "french", "locale"},
    Intent.SAFETY_RISK: {"password", "secret", "key", "credential", "injection", "bypass", "exploit"},
}

_STOPWORDS = frozenset(
    """a an the is are was were be been being do does did doing have has had
    i me my we our you your he she it they them this that these those of in on at
    to for with and or but if then than so as by from about into over under
    please can could would should will shall may might must not no yes""".split()
)

_WORD = re.compile(r"[a-z0-9#+.\-]+")

#: Rules are evaluated in this order, so the highest-severity / most specific
#: intent wins over the more general ones it happens to overlap with.
RULE_PRIORITY: tuple[Intent, ...] = (
    Intent.SAFETY_RISK,
    Intent.GREETING,
    Intent.TRANSLATION,
    Intent.MATH,
    Intent.SUMMARIZE,
    Intent.SQL,
    Intent.CODE,
    Intent.ANALYSIS,
    Intent.CHITCHAT,
    Intent.KNOWLEDGE,
)


#: How unambiguous each rule is. A vague keyword (``service``, ``api``) should not
#: outweigh a specific phrase (``what is``, ``select ... from``), so intents whose
#: patterns are mostly generic nouns score below the phrase-shaped ones.
RULE_WEIGHTS: dict[Intent, float] = {
    Intent.SAFETY_RISK: 2.0,
    Intent.SQL: 1.6,
    Intent.MATH: 1.6,
    Intent.TRANSLATION: 1.4,
    Intent.SUMMARIZE: 1.4,
    Intent.GREETING: 1.4,
    Intent.KNOWLEDGE: 1.3,
    Intent.ANALYSIS: 1.0,
    Intent.CODE: 0.7,
    Intent.CHITCHAT: 1.4,
}


#: A rule may be right for the wrong reason. Mentioning "postgres" does not mean the
#: user wants SQL written, and "the ingest service" does not make a question about
#: the service's retry policy a coding task. When a rule fired on its vague half
#: *and* nothing specific, the score is scaled down.
#:
#: ``(vague, specific, factor)`` — the factor applies only when ``vague`` matches and
#: ``specific`` does not.
WEAK_SIGNALS: dict[Intent, tuple[re.Pattern[str], re.Pattern[str], float]] = {
    Intent.SQL: (
        re.compile(r"\b(?:postgres|mysql|mongo|redis|database)\b", re.IGNORECASE),
        re.compile(
            r"\b(?:select|insert|update|delete|from|where|into|table|join|"
            r"sql|query|schema|index)\b",
            re.IGNORECASE,
        ),
        0.45,
    ),
    Intent.CODE: (
        re.compile(r"\b(?:service|api|endpoint|deploy|regression|flaky)\b", re.IGNORECASE),
        re.compile(
            r"\b(?:refactor|debug|stack ?trace|traceback|function|class|compile|"
            r"unit test|pytest|deadlock|race condition|memory leak|crash|exception|"
            r"returning\s+[45]\d\d|[45]\d\d\s*(?:error|status|response))\b",
            re.IGNORECASE,
        ),
        0.7,
    ),
    Intent.KNOWLEDGE: (
        re.compile(r"^\s*(?:what|why|how|when|who|where)\b", re.IGNORECASE),
        re.compile(
            r"\b(?:documentation|docs|policy|policies|runbook|guide|"
            r"according to|meaning|definition|overview)\b",
            re.IGNORECASE,
        ),
        0.55,
    ),
    Intent.ANALYSIS: (
        re.compile(r"\b(?:metrics?|benchmark)\b", re.IGNORECASE),
        re.compile(
            r"\b(?:analy[sz]e|compare|trade-?offs?|root cause|evaluate|"
            r"pros and cons|forecast)\b",
            re.IGNORECASE,
        ),
        0.6,
    ),
}


def _demote(intent: Intent, text: str, base_weight: float) -> float:
    signal = WEAK_SIGNALS.get(intent)
    if signal is None:
        return base_weight
    vague, specific, factor = signal
    if vague.search(text) and not specific.search(text):
        return base_weight * factor
    return base_weight


def tokenize(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


# -- confidence blend ------------------------------------------------------
_RULE_TRUST = 0.97
_LEXICAL_TRUST = 0.72
_LLM_TRUST = 0.95


@dataclass(slots=True)
class IntentResult:
    """Classification output with the evidence that produced it."""

    intent: Intent
    confidence: float
    stage: str
    scores: dict[str, float] = field(default_factory=dict)
    runner_up: str | None = None
    margin: float = 0.0
    reason: str = ""

    @property
    def needs_retrieval(self) -> bool:
        return self.intent in {Intent.KNOWLEDGE, Intent.SUMMARIZE, Intent.ANALYSIS, Intent.UNKNOWN}

    @property
    def needs_tools(self) -> bool:
        return self.intent in {Intent.CODE, Intent.SQL, Intent.MATH, Intent.ANALYSIS}

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": str(self.intent),
            "confidence": round(self.confidence, 4),
            "stage": self.stage,
            "runner_up": self.runner_up,
            "margin": round(self.margin, 4),
            "reason": self.reason,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
        }


class IntentDetector:
    """Cascade classifier. Injectable LLM keeps the default path network-free."""

    def __init__(
        self,
        client: LLMClient | None = None,
        *,
        margin: float = 0.15,
        min_llm_confidence: float = 0.5,
    ) -> None:
        self.client = client
        self.margin = margin
        self.min_llm_confidence = min_llm_confidence
        self._idf = _build_idf(PROTOTYPES)
        self.stats: Counter[str] = Counter()

    # -- stages -----------------------------------------------------------
    def detect(self, text: str, *, allow_llm: bool = True) -> IntentResult:
        self.stats["calls"] += 1
        rule_hit = self._rules(text)
        if rule_hit is not None:
            self.stats["rule_hits"] += 1
            return rule_hit

        lexical = self._lexical(text)
        self.stats["lexical_falls_through"] += 1

        if not allow_llm or self.client is None:
            lexical.stage = "lexical"
            return lexical

        if lexical.margin >= self.margin:
            self.stats["llm_avoided"] += 1
            lexical.reason = f"margin {lexical.margin:.2f} >= {self.margin:.2f}, skipped model"
            return lexical

        self.stats["llm_invocations"] += 1
        return self._llm(text, lexical)

    def _rules(self, text: str) -> IntentResult | None:
        """Score every rule, not just the first one that fires.

        Scanning in priority order and returning early let a single weak keyword
    win: "What is the retry policy for the ingest service?" matched CODE's bare
    ``service`` before KNOWLEDGE's much stronger ``what is`` + ``policy``.
        Counting matches per intent, weighted by how unambiguous each rule is, lets
        two specific signals beat one vague one.
        """
        candidates: list[tuple[float, int, Intent, int]] = []
        for priority, intent in enumerate(RULE_PRIORITY):
            pattern = RULES.get(intent)
            if not pattern:
                continue
            matches = len(pattern.findall(text))
            if matches:
                weight = _demote(intent, text, RULE_WEIGHTS.get(intent, 1.0))
                candidates.append((matches * weight, -priority, intent, matches))

        if not candidates:
            return None

        score, _, intent, matches = max(candidates)
        runner_up = sorted(candidates, reverse=True)[1] if len(candidates) > 1 else None
        return IntentResult(
            intent=intent,
            confidence=_RULE_TRUST,
            stage="rules",
            scores={str(i): round(s, 3) for s, _, i, _ in sorted(candidates, reverse=True)},
            runner_up=str(runner_up[2]) if runner_up else None,
            margin=round(min(1.0, (score - (runner_up[0] if runner_up else 0.0)) / max(score, 1e-9)), 4),
            reason=(
                f"rule `{intent}` matched {matches} time(s) "
                f"at weight {RULE_WEIGHTS.get(intent, 1.0)}"
            ),
        )

    def _lexical(self, text: str) -> IntentResult:
        tokens = tokenize(text)
        if not tokens:
            return IntentResult(
                intent=Intent.UNKNOWN,
                confidence=0.0,
                stage="lexical",
                reason="no content tokens after stopword removal",
            )

        counts = Counter(tokens)
        scores: dict[str, float] = {}
        for intent, terms in PROTOTYPES.items():
            score = sum(self._idf.get(t, 1.0) * counts.get(t, 0) for t in terms if t in counts)
            scores[str(intent)] = score
            scores[str(intent)] += 1.5 * sum(counts.get(t, 0) for t in terms)

        if not any(scores.values()):
            return IntentResult(
                intent=Intent.CHITCHAT,
                confidence=_LEXICAL_TRUST * 0.6,
                stage="lexical",
                scores=scores,
                reason="no prototype term overlap; treated as conversational",
            )

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        top_label, top_score = ranked[0]
        runner_score = ranked[1][1] if len(ranked) > 1 else 0.0
        total = sum(scores.values()) or 1.0
        share = top_score / total
        margin = (top_score - runner_score) / (top_score or 1.0)

        return IntentResult(
            intent=Intent(top_label),
            confidence=round(min(0.95, _LEXICAL_TRUST * (0.4 + 0.6 * share)), 4),
            stage="lexical",
            scores=scores,
            runner_up=ranked[1][0] if len(ranked) > 1 else None,
            margin=round(margin, 4),
            reason=f"prototype overlap on {top_label}",
        )

    def _llm(self, text: str, prior: IntentResult) -> IntentResult:
        labels = ", ".join(str(i) for i in PROTOTYPES)
        prompt = (
            "Classify the user's request into exactly one intent.\n"
            f"Intents: {labels}\n"
            'Reply as JSON only: {"intent": "<label>", "confidence": <0-1>}\n\n'
            f"Request: {text[:800]}"
        )
        try:
            response = self.client.complete(  # type: ignore[union-attr]
                [Message("user", prompt)],
                system="You are a deterministic intent classifier. Output JSON only.",
                temperature=0.0,
                max_output_tokens=120,
                cache_system=True,
            )
            parsed = _parse_json(response.text)
            intent = Intent(str(parsed.get("intent", "unknown")))
            confidence = float(parsed.get("confidence", prior.confidence))
        except Exception:  # noqa: BLE001 - never let classification break the request
            prior.reason += " (llm stage failed, keeping lexical)"
            return prior

        blended = (1 - _LLM_TRUST) * prior.confidence + _LLM_TRUST * confidence
        if blended < self.min_llm_confidence:
            blended = prior.confidence
        scores = dict(prior.scores)
        scores[str(intent)] = max(scores.get(str(intent), 0.0), blended)
        return IntentResult(
            intent=intent,
            confidence=round(blended, 4),
            stage="llm",
            scores=scores,
            runner_up=prior.runner_up,
            margin=prior.margin,
            reason="model classification blended with lexical prior",
        )

    # -- introspection ----------------------------------------------------
    def effectiveness(self) -> dict[str, Any]:
        calls = self.stats["calls"] or 1
        return {
            "calls": self.stats["calls"],
            "rule_hit_rate": round(self.stats["rule_hits"] / calls, 4),
            "llm_invocations": self.stats["llm_invocations"],
            "llm_call_rate": round(self.stats["llm_invocations"] / calls, 4),
            "llm_cost_avoidance_rate": round(self.stats["llm_avoided"] / calls, 4),
            "rule_hits": self.stats["rule_hits"],
            "lexical_falls_through": self.stats["lexical_falls_through"],
        }


def _build_idf(prototypes: dict[Intent, set[str]]) -> dict[str, float]:
    """Rare prototype terms carry more signal than terms shared by every intent."""
    n = len(prototypes) or 1
    df: Counter[str] = Counter()
    for terms in prototypes.values():
        df.update(terms)
    return {term: math.log((n + 1) / (count + 0.5)) for term, count in df.items()}


def _parse_json(text: str) -> dict[str, Any]:
    import json

    cleaned = text.strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        parsed = json.loads(cleaned[start : end + 1])
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


__all__ = [
    "PROTOTYPES",
    "RULE_PRIORITY",
    "RULES",
    "Intent",
    "IntentDetector",
    "IntentResult",
    "tokenize",
]