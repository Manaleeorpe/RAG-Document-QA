"""
LLM evaluator — the "LLM evaluator" box in the architecture diagram.

Scores a generated answer on three axes (each 0.0-1.0) in a single LLM call:
    relevance     — does the answer address the question?
    completeness  — does it cover everything the question asked?
    groundedness  — is every claim supported by the retrieved context?

Unlike the guardrails (which block/flag), the evaluator produces quality scores
for logging/observability and for triggering warnings or a future retry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate


@dataclass
class EvaluationResult:
    relevance: float
    completeness: float
    groundedness: float
    reason: str = ""
    failed: bool = False  # True when the eval itself errored (scores are fallbacks)

    @property
    def overall(self) -> float:
        return round((self.relevance + self.completeness + self.groundedness) / 3, 3)

    def as_dict(self) -> Dict:
        return {
            "relevance": self.relevance,
            "completeness": self.completeness,
            "groundedness": self.groundedness,
            "overall": self.overall,
            "reason": self.reason,
        }


def _clamp(value, default: float = 1.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def evaluate_answer(llm, question: str, answer: str, context: str) -> EvaluationResult:
    """
    Score an answer against the question and the retrieved context.

    Fails open (all scores 1.0, failed=True) if the LLM/JSON errors, so evaluation
    never breaks the response — it is observability, not a gate.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an evaluator for a RAG system. Score the ANSWER to the "
                "QUESTION given the CONTEXT, each from 0.0 to 1.0:\n"
                "- relevance: how directly the answer addresses the question.\n"
                "- completeness: how fully it covers what the question asked.\n"
                "- groundedness: how well every claim is supported by the context "
                "(no unsupported or invented facts).\n"
                "Respond ONLY with JSON: "
                '{{"relevance": 0.0, "completeness": 0.0, "groundedness": 0.0, "reason": "..."}}',
            ),
            ("human", "QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"),
        ]
    )
    chain = prompt | llm | JsonOutputParser()
    try:
        r = chain.invoke({"question": question, "context": context, "answer": answer})
        return EvaluationResult(
            relevance=_clamp(r.get("relevance")),
            completeness=_clamp(r.get("completeness")),
            groundedness=_clamp(r.get("groundedness")),
            reason=str(r.get("reason", "")),
        )
    except Exception:
        return EvaluationResult(1.0, 1.0, 1.0, reason="evaluation-failed-fallback", failed=True)
