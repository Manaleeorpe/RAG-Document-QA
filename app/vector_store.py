from typing import Dict, List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class VectorStore:
    """
    Thin wrapper around LangChain's Chroma vector store.

    Persists to the same on-disk Chroma collection ("documents", cosine space)
    used previously, so existing data stays queryable. Exposes the underlying
    store via `as_retriever()` so it can plug straight into LangChain retrievers.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        persist_directory: str = "./chroma_db",
        collection_name: str = "documents",
    ):
        self.store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"},  # cosine similarity
        )

    def add_documents(self, documents: List[Document], ids: List[str]):
        """Embed and store LangChain Documents (chunks) under explicit ids."""
        self.store.add_documents(documents=documents, ids=ids)
        print(f"Added {len(documents)} chunks")

    def as_retriever(self, **kwargs):
        """Return a LangChain retriever over this store (used by the hybrid stack)."""
        return self.store.as_retriever(**kwargs)

    def get_doc_chunks(self, doc_name: str) -> List[Document]:
        """
        Fetch every stored chunk for one document as LangChain Documents.

        Needed to build the in-memory BM25 keyword index, which works over the
        full corpus rather than just nearest-neighbour vector hits.
        """
        got = self.store.get(where={"document": doc_name})
        texts = got.get("documents", []) or []
        metas = got.get("metadatas", []) or [{} for _ in texts]
        return [Document(page_content=t, metadata=m or {}) for t, m in zip(texts, metas)]

    def get_stats(self) -> Dict:
        return {
            "total_chunks": self.store._collection.count(),
            "collection_name": self.store._collection.name,
        }


if __name__ == "__main__":
    from embeddings import EmbeddingModel

    store = VectorStore(EmbeddingModel().embeddings)
    print(store.get_stats())
