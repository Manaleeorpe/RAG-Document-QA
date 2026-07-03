"""
Hybrid retrieval stack, built entirely from LangChain components.

    Chroma (vector)  ┐
                     ├─ EnsembleRetriever (RRF fusion)
    BM25 (keyword)   ┘            │
                                  └─ ContextualCompressionRetriever
                                          └─ CrossEncoderReranker (re-rank + top_n)

This maps to the "Hybrid search" + "Re-rank + filter" boxes in the diagram and
the "ChromaDB / BM25" MCP-server layer.
"""
from typing import Optional

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.retrievers import BaseRetriever


def build_reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n: int = 5,
) -> CrossEncoderReranker:
    """CrossEncoder re-ranker used as the document compressor (re-rank + filter)."""
    return CrossEncoderReranker(
        model=HuggingFaceCrossEncoder(model_name=model_name),
        top_n=top_n,
    )


def build_hybrid_retriever(
    vector_store,
    doc_name: str,
    reranker: CrossEncoderReranker,
    k: int = 10,
    weights=(0.5, 0.5),
) -> Optional[BaseRetriever]:
    """
    Build the per-document hybrid retriever.

    Returns None if the document has no stored chunks (e.g. unknown doc_name),
    so the caller can short-circuit instead of constructing an empty BM25 index.
    """
    docs = vector_store.get_doc_chunks(doc_name)
    if not docs:
        return None

    # Dense semantic retrieval over the filtered document.
    vector_retriever = vector_store.as_retriever(
        search_kwargs={"k": k, "filter": {"document": doc_name}}
    )

    # Sparse keyword retrieval (BM25) built from the same document's chunks.
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k

    # Fuse both with Reciprocal Rank Fusion (EnsembleRetriever's default).
    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=list(weights),
    )

    # Re-rank the fused candidates and keep the reranker's top_n.
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble,
    )
