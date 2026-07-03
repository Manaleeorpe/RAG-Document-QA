"""
Query analysis and routing.

    analyze_query  -> Topic guardrail (on_topic?) + Classify (factual|analytical|ambiguous)
    decompose_query -> sub-questions for analytical queries (Decompose + multi-search)
    generate_clarification -> a clarifying question for ambiguous queries (Clarify)

The topic guardrail and the classifier are done in a single LLM call because both
are "understand the query" tasks — this halves latency/cost while still exposing
`on_topic` (the guardrail) and `category` (the classifier) separately.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

VALID_CATEGORIES = {"factual", "analytical", "ambiguous"}


@dataclass
class QueryAnalysis:
    on_topic: bool
    category: str  # factual | analytical | ambiguous
    reason: str = ""


def analyze_query(llm, question: str, doc_name: str = None) -> QueryAnalysis:
    """
    Topic guardrail + classification in one call.

    `doc_name` is the document already selected for this request. It is given to
    the model so that references like "the file", "this resume", or "it" resolve
    to that document instead of being treated as ambiguous.

    Fails open (on_topic=True, category="factual") if the LLM/JSON errors, so a
    transient hiccup degrades to normal factual retrieval rather than blocking.
    """
    doc_context = (
        f"A specific document ('{doc_name}') is ALREADY selected for this request. "
        "Every question refers to THIS document by default. Words like 'the file', "
        "'this document', 'the resume', 'it', 'this', or 'here' mean that selected "
        "document. Because the document is always known, a question is almost never "
        "'ambiguous' just for lacking an explicit subject.\n"
        "Examples (with a document selected):\n"
        "  - 'what is in the file' -> factual\n"
        "  - 'what does it contain' -> factual\n"
        "  - 'summarize the resume' / 'give an overview' -> analytical\n"
        "  - 'compare the skills and experience' -> analytical\n"
        "  - 'what are the important ones?' (no prior context to resolve 'ones') -> ambiguous\n"
        if doc_name
        else ""
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You screen and classify questions for a document Q&A system that "
                "answers strictly from uploaded documents.\n"
                f"{doc_context}"
                "1. on_topic: true for any genuine attempt to ask something about the "
                "document, INCLUDING vague or brief questions. Set it false ONLY for "
                "chit-chat, harmful requests, or requests to do something other than "
                "answer from the document (e.g. write code, tell a joke).\n"
                "2. category:\n"
                "   - 'factual': a direct lookup of content in the document, including "
                "'what is in the file' style questions.\n"
                "   - 'analytical': needs reasoning, summarization, comparison, or "
                "several aspects (e.g. 'summarize it', 'compare X and Y').\n"
                "   - 'ambiguous': ONLY when the question cannot be answered even knowing "
                "the selected document, because it refers to something with no resolvable "
                "meaning (e.g. 'what about the other ones?' with no prior context).\n"
                "Respond ONLY with JSON: "
                '{{"on_topic": true/false, "category": "factual|analytical|ambiguous", "reason": "..."}}',
            ),
            ("human", "Question: {question}"),
        ]
    )
    chain = prompt | llm | JsonOutputParser()
    try:
        result = chain.invoke({"question": question})
        category = str(result.get("category", "factual")).lower().strip()
        if category not in VALID_CATEGORIES:
            category = "factual"
        return QueryAnalysis(
            on_topic=bool(result.get("on_topic", True)),
            category=category,
            reason=str(result.get("reason", "")),
        )
    except Exception:
        return QueryAnalysis(on_topic=True, category="factual", reason="analysis-failed-fallback")


def decompose_query(llm, question: str, max_sub_questions: int = 4) -> List[str]:
    """Break an analytical question into focused sub-questions for multi-search."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Break the user's analytical question into 2-{n} focused, "
                "self-contained sub-questions that together fully answer it. "
                'Respond ONLY with JSON: {{"sub_questions": ["...", "..."]}}',
            ),
            ("human", "Question: {question}"),
        ]
    )
    chain = prompt | llm | JsonOutputParser()
    try:
        result = chain.invoke({"question": question, "n": max_sub_questions})
        subs = [s.strip() for s in result.get("sub_questions", []) if str(s).strip()]
        return subs[:max_sub_questions] if subs else [question]
    except Exception:
        return [question]  # fall back to treating it as a single query


def generate_clarification(llm, question: str) -> str:
    """Produce one concise clarifying question for an ambiguous query."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "The user's question is ambiguous or underspecified. Ask ONE short, "
                "specific clarifying question that would let you answer it. "
                "Return only the clarifying question.",
            ),
            ("human", "Question: {question}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    try:
        return chain.invoke({"question": question}).strip()
    except Exception:
        return "Could you clarify or add more detail to your question?"
