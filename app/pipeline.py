import os
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from calLLM import callLLM
from chunker import chunk_documents
from document_processor import load_pdf
from embeddings import EmbeddingModel
from evaluator import evaluate_answer
from guardrails import run_input_guardrails, run_output_guardrails
from query_router import analyze_query, decompose_query, generate_clarification
from retrievers import build_hybrid_retriever, build_reranker
from vector_store import VectorStore

load_dotenv()


class DocumentPipeline:
    """RAG pipeline wired end-to-end from LangChain components."""

    def __init__(self):
        self.embedder = EmbeddingModel()
        self.vector_store = VectorStore(self.embedder.embeddings)

        # LLM (DeepSeek via the OpenAI-compatible endpoint) as a LangChain chat model.
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            temperature=0,
        )

        # Re-rank + filter compressor, built once and reused per request.
        self.reranker = build_reranker(top_n=5)

        # Guardrail / routing / evaluation toggles.
        self.enable_input_guardrails = True
        self.enable_routing = True
        self.enable_output_guardrails = True
        self.enable_evaluation = True
        self.groundedness_warn_threshold = 0.5

    @staticmethod
    def _format_docs(docs) -> str:
        """Render retrieved chunks into a single context string with source tags."""
        return "\n\n".join(
            f"[Source: {d.metadata.get('doc_name', 'unknown')}]\n{d.page_content}"
            for d in docs
        )

    @staticmethod
    def _merge_docs(doc_lists) -> list:
        """Flatten multi-search results, de-duplicating by (document, chunk_id)."""
        merged, seen = [], set()
        for docs in doc_lists:
            for d in docs:
                key = (d.metadata.get("document"), d.metadata.get("chunk_id"))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(d)
        return merged

    def _sources(self, docs, doc_name):
        return [
            {
                "index": i,
                "doc": d.metadata.get("doc_name", doc_name),
                "chunk_id": d.metadata.get("chunk_id"),
                "text": d.page_content[:100],
            }
            for i, d in enumerate(docs, start=1)
        ]

    def process_document(self, pdf_path: Path) -> Dict:
        pages = load_pdf(pdf_path)
        doc_name = pdf_path.stem

        chunks = chunk_documents(pages, doc_name)
        ids = [f"{doc_name}_{i}" for i in range(len(chunks))]
        self.vector_store.add_documents(chunks, ids=ids)

        return {
            "document": pdf_path.name,
            "chunks": len(chunks),
            "status": "processed",
        }

    def process_question(self, question: str, doc_name: str, session_id: str) -> Dict:
        # ── Input guardrails: prompt injection (block) + PII (redact) ──
        if self.enable_input_guardrails:
            guard = run_input_guardrails(question)
            if not guard.allowed:
                return {
                    "question": question,
                    "answer": "Your request was blocked by the input safety check.",
                    "blocked": True,
                    "reasons": guard.reasons,
                    "sources": [],
                }
            question = guard.query  # de-identified query flows downstream

        # ── Topic guardrail + classification (factual | analytical | ambiguous) ──
        category = "factual"
        if self.enable_routing:
            analysis = analyze_query(self.llm, question, doc_name=doc_name)
            category = analysis.category
            if not analysis.on_topic:
                return {
                    "question": question,
                    "answer": "That question is outside what I can answer from the documents.",
                    "blocked": True,
                    "category": category,
                    "reasons": [analysis.reason or "off-topic"],
                    "sources": [],
                }
            if category == "ambiguous":
                return {
                    "question": question,
                    "answer": generate_clarification(self.llm, question),
                    "needs_clarification": True,
                    "category": category,
                    "sources": [],
                }

        # ── Retrieval: hybrid search (Chroma + BM25, RRF) -> re-rank + filter ──
        retriever = build_hybrid_retriever(self.vector_store, doc_name, self.reranker)
        if retriever is None:
            return {
                "question": question,
                "answer": "I couldn't find that document, or it has no indexed content.",
                "category": category,
                "sources": [],
            }

        sub_questions = []
        if category == "analytical":
            # Decompose + multi-search: retrieve per sub-question, then merge/dedup.
            sub_questions = decompose_query(self.llm, question)
            docs = self._merge_docs([retriever.invoke(sq) for sq in sub_questions])
        else:  # factual
            docs = retriever.invoke(question)

        if not docs:
            return {
                "question": question,
                "answer": "I couldn't find anything relevant to that in the document.",
                "category": category,
                "sources": [],
            }

        # ── Generate answer (LLM + retrieved context + session history) ──
        context_str = self._format_docs(docs)
        answer = callLLM(context=context_str, question=question, session_id=session_id)

        # ── Output guardrail: toxicity (hard block) ──
        warnings = []
        toxic_blocked = False
        if self.enable_output_guardrails:
            out = run_output_guardrails(answer)
            answer = out.answer
            warnings = list(out.warnings)
            toxic_blocked = not out.passed

        # ── LLM evaluator: relevance / completeness / groundedness ──
        evaluation = None
        if self.enable_evaluation and not toxic_blocked:
            ev = evaluate_answer(self.llm, question, answer, context_str)
            evaluation = ev.as_dict()
            if ev.groundedness < self.groundedness_warn_threshold:
                warnings.append(
                    f"Low groundedness ({ev.groundedness:.2f}); answer may not be fully "
                    "supported by the sources."
                )

        response = {
            "question": question,
            "answer": answer,
            "category": category,
            "sources": self._sources(docs, doc_name),
        }
        if sub_questions:
            response["sub_questions"] = sub_questions
        if warnings:
            response["warnings"] = warnings
        if evaluation is not None:
            response["evaluation"] = evaluation
        return response


if __name__ == "__main__":
    pipeline = DocumentPipeline()
    result = pipeline.process_question(
        "What programming languages and skills are listed?",
        doc_name="Manalee_Orpe_BNPP_Resume",
        session_id="test-session",
    )
    print(result["answer"])
    for s in result["sources"]:
        print(s["index"], s["doc"], s["chunk_id"])
