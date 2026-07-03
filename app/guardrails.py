"""
Input and output guardrails for the RAG pipeline.

Input  (before retrieval): prompt-injection detection, PII detection + redaction.
Output (after generation): toxicity filter, faithfulness / hallucination check.

The cheap, deterministic checks (injection, PII, toxicity) are rule-based so they
add no latency or API cost. The faithfulness check needs to reason over the answer
vs. the retrieved context, so it uses the LLM.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate


# ─────────────────────────── Input guardrails ────────────────────────────────

# Common prompt-injection / jailbreak phrasings.
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|the\s+|your\s+)?(previous|prior|above)\s+(instructions|prompts?)",
    r"disregard\s+(all\s+|the\s+|your\s+)?(previous|prior|above)",
    r"forget\s+(everything|all|your\s+instructions)",
    r"you\s+are\s+now\s+",
    r"\bact\s+as\s+(a|an|if)\b",
    r"pretend\s+(to\s+be|you\s+are)",
    r"(reveal|show|print|repeat)\s+(me\s+)?(your\s+)?(system\s+)?(prompt|instructions)",
    r"\boverride\b.*\b(instructions|rules|restrictions)\b",
    r"\bjailbreak\b",
    r"\bdo\s+anything\s+now\b|\bDAN\b",
    r"new\s+instructions?\s*:",
]

# PII detectors → (label, compiled regex).
_PII_PATTERNS: Dict[str, re.Pattern] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\d{10}|\d{3}[\s-]\d{3}[\s-]\d{4})\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "aadhaar": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    "pan": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
}

# Minimal, illustrative toxicity list. For production, swap in a real classifier
# (e.g. detoxify / a moderation endpoint) — this is a deterministic placeholder.
_TOXIC_TERMS = [
    "kill yourself", "kys", "retard", "slur", "hate you",
    # NOTE: keep this list minimal; it is a stand-in for a real moderation model.
]


@dataclass
class InputGuardrailResult:
    allowed: bool
    query: str  # possibly PII-redacted
    reasons: List[str] = field(default_factory=list)
    pii_found: Dict[str, List[str]] = field(default_factory=dict)


def check_prompt_injection(query: str) -> List[str]:
    """Return a list of matched injection patterns (empty = clean)."""
    hits = []
    for pat in _INJECTION_PATTERNS:
        if re.search(pat, query, flags=re.IGNORECASE):
            hits.append(pat)
    return hits


def detect_pii(query: str) -> Dict[str, List[str]]:
    """Return {pii_label: [matches]} for any PII found in the query."""
    found: Dict[str, List[str]] = {}
    for label, pattern in _PII_PATTERNS.items():
        matches = [m.group(0) for m in pattern.finditer(query)]
        if matches:
            found[label] = matches
    return found


def redact_pii(query: str, pii: Dict[str, List[str]]) -> str:
    """Replace detected PII spans with [REDACTED_<LABEL>] placeholders."""
    redacted = query
    for label, matches in pii.items():
        for m in matches:
            redacted = redacted.replace(m, f"[REDACTED_{label.upper()}]")
    return redacted


def run_input_guardrails(query: str) -> InputGuardrailResult:
    """
    Prompt-injection is a hard block; PII is redacted and allowed through so the
    (de-identified) question can still be answered.
    """
    reasons: List[str] = []

    injection_hits = check_prompt_injection(query)
    if injection_hits:
        return InputGuardrailResult(
            allowed=False,
            query=query,
            reasons=["Possible prompt injection detected."],
        )

    pii = detect_pii(query)
    safe_query = redact_pii(query, pii) if pii else query
    if pii:
        reasons.append(f"Redacted PII: {', '.join(pii.keys())}")

    return InputGuardrailResult(
        allowed=True,
        query=safe_query,
        reasons=reasons,
        pii_found=pii,
    )


# ─────────────────────────── Output guardrails ───────────────────────────────

@dataclass
class OutputGuardrailResult:
    passed: bool
    answer: str  # possibly replaced if blocked
    toxic: bool = False
    warnings: List[str] = field(default_factory=list)


def check_toxicity(text: str) -> List[str]:
    """Return matched toxic terms (empty = clean)."""
    lowered = text.lower()
    return [t for t in _TOXIC_TERMS if t in lowered]


def check_faithfulness(llm, answer: str, context: str) -> Dict:
    """
    LLM grounding check: is every claim in `answer` supported by `context`?
    Combines the diagram's "Hallucination check" + "Faithfulness" boxes.

    Fails open (assumes grounded) if the check itself errors, so the guardrail
    never blocks a valid answer due to a transient LLM/JSON hiccup.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a strict fact-checker. Decide whether the ANSWER is fully "
                "supported by the CONTEXT. An answer is grounded only if every factual "
                "claim can be verified from the context. Respond ONLY with JSON: "
                '{{"grounded": true/false, "score": 0.0-1.0, "unsupported_claims": [..]}}',
            ),
            ("human", "CONTEXT:\n{context}\n\nANSWER:\n{answer}"),
        ]
    )
    chain = prompt | llm | JsonOutputParser()
    try:
        result = chain.invoke({"context": context, "answer": answer})
        return {
            "grounded": bool(result.get("grounded", True)),
            "score": float(result.get("score", 1.0)),
            "unsupported_claims": result.get("unsupported_claims", []) or [],
        }
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        return {"grounded": True, "score": 1.0, "unsupported_claims": []}


def run_output_guardrails(answer: str) -> OutputGuardrailResult:
    """
    Toxicity filter (rule-based, hard block). Groundedness/faithfulness is handled
    by the LLM evaluator (see evaluator.py) to avoid a redundant grounding call.
    """
    toxic_hits = check_toxicity(answer)
    if toxic_hits:
        return OutputGuardrailResult(
            passed=False,
            answer="The generated response was withheld because it failed the safety check.",
            toxic=True,
            warnings=["Toxic content detected in answer."],
        )
    return OutputGuardrailResult(passed=True, answer=answer, toxic=False)
