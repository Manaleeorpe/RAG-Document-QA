import os
import time
from typing import Dict, List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


class VectorStore:
    """
    Wrapper around Pinecone via LangChain's PineconeVectorStore.

    Chunks are stored in a single Pinecone index and filtered per document
    using the `document` metadata field. An in-memory cache of raw chunks is
    kept per-process so the BM25 retriever can be rebuilt without a round-trip
    to Pinecone; on restart users re-upload to repopulate the cache.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        index_name: str = "documents",
    ):
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

        existing = [i.name for i in pc.list_indexes()]
        if index_name not in existing:
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # wait for index to be ready
            while not pc.describe_index(index_name).status["ready"]:
                time.sleep(1)

        self.store = PineconeVectorStore(
            index=pc.Index(index_name),
            embedding=embeddings,
        )
        self._index = pc.Index(index_name)
        self._doc_chunks: Dict[str, List[Document]] = {}  # in-process cache for BM25

    def add_documents(self, documents: List[Document], ids: List[str]):
        self.store.add_documents(documents=documents, ids=ids)
        if documents:
            doc_name = documents[0].metadata.get("document")
            if doc_name:
                self._doc_chunks[doc_name] = documents
        print(f"Added {len(documents)} chunks")

    def as_retriever(self, **kwargs):
        return self.store.as_retriever(**kwargs)

    def get_doc_chunks(self, doc_name: str) -> List[Document]:
        # return cached chunks if available (same process lifetime)
        if doc_name in self._doc_chunks:
            return self._doc_chunks[doc_name]
        # fallback after restart: query Pinecone with metadata filter
        results = self.store.similarity_search(
            query=doc_name,
            k=500,
            filter={"document": {"$eq": doc_name}},
        )
        if results:
            self._doc_chunks[doc_name] = results
        return results

    def get_stats(self) -> Dict:
        stats = self._index.describe_index_stats()
        return {
            "total_chunks": stats.total_vector_count,
            "collection_name": self._index.name if hasattr(self._index, "name") else "documents",
        }


if __name__ == "__main__":
    from embeddings import EmbeddingModel
    store = VectorStore(EmbeddingModel().embeddings)
    print(store.get_stats())
