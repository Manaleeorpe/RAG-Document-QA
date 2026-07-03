"""
test_retrievers.py — tests for app/retrievers.py :: build_hybrid_retriever

Uses a fake vector store + a no-op compressor so the hybrid wiring (BM25 +
EnsembleRetriever + ContextualCompressionRetriever) is exercised without loading
any embedding or cross-encoder models.
"""
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.runnables import RunnableLambda

from retrievers import build_hybrid_retriever
from langchain_classic.retrievers import ContextualCompressionRetriever


class _NoOpCompressor(BaseDocumentCompressor):
    """Returns documents unchanged — stands in for the CrossEncoder reranker."""

    def compress_documents(self, documents, query, callbacks=None):
        return list(documents)


class _FakeVectorStore:
    """Minimal stand-in exposing the two methods build_hybrid_retriever needs."""

    def __init__(self, docs):
        self._docs = docs

    def get_doc_chunks(self, doc_name):
        return self._docs

    def as_retriever(self, **kwargs):
        # A trivial dense retriever that just echoes the stored docs.
        return RunnableLambda(lambda _query: list(self._docs))


def _docs():
    return [
        Document(page_content="python and sql skills", metadata={"document": "d", "chunk_id": 0}),
        Document(page_content="machine learning experience", metadata={"document": "d", "chunk_id": 1}),
    ]


def test_returns_none_for_empty_document():
    vs = _FakeVectorStore(docs=[])
    assert build_hybrid_retriever(vs, "missing", _NoOpCompressor()) is None


def test_builds_contextual_compression_retriever():
    vs = _FakeVectorStore(docs=_docs())
    retriever = build_hybrid_retriever(vs, "d", _NoOpCompressor())
    assert isinstance(retriever, ContextualCompressionRetriever)


def test_hybrid_retriever_returns_documents():
    vs = _FakeVectorStore(docs=_docs())
    retriever = build_hybrid_retriever(vs, "d", _NoOpCompressor())
    results = retriever.invoke("python sql")
    assert len(results) >= 1
    assert all(isinstance(d, Document) for d in results)


def test_respects_k_parameter_on_bm25():
    vs = _FakeVectorStore(docs=_docs())
    # k larger than corpus must not error; fusion still returns the docs.
    retriever = build_hybrid_retriever(vs, "d", _NoOpCompressor(), k=50)
    assert retriever.invoke("machine learning")
