from typing import List

import numpy as np
from langchain_openai import OpenAIEmbeddings


class EmbeddingModel:
    """
    Wrapper around LangChain's OpenAIEmbeddings (text-embedding-3-small, 1536 dims).

    The LangChain embeddings object is exposed as `.embeddings` and passed
    directly to the Chroma vector store. The numpy helpers (`embed_chunks`,
    `embed_query`) are kept for callers/tests that want raw vectors.
    """

    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.embeddings = OpenAIEmbeddings(model=model_name)
        self.model_name = model_name

    def embed_chunks(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = self.embeddings.embed_documents(texts)
        return np.array(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return np.array(self.embeddings.embed_query(query), dtype=np.float32)


if __name__ == "__main__":
    embedder = EmbeddingModel()
    chunks = ["This is chunk 1", "This is chunk 2"]
    embeddings = embedder.embed_chunks(chunks)
    print(f"Embeddings shape: {embeddings.shape}")
