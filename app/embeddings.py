from typing import List

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings


# model from huggingface.co/sentence-transformers/all-MiniLM-L6-v2
class EmbeddingModel:
    """
    Wrapper around LangChain's HuggingFaceEmbeddings (all-MiniLM-L6-v2, 384 dims).

    The LangChain embeddings object is exposed as `.embeddings` and passed
    directly to the Chroma vector store. The numpy helpers (`embed_chunks`,
    `embed_query`) are kept for callers/tests that want raw vectors.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading model {model_name}...")
        # The LangChain embeddings component used by the vector store / retrievers.
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.model_name = model_name
        print("Model loaded")

    def embed_chunks(self, texts: List[str]) -> np.ndarray:  # for the document
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = self.embeddings.embed_documents(texts)
        return np.array(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:  # for the user query
        # embed_query returns a flat list of 384 floats for a single sentence.
        return np.array(self.embeddings.embed_query(query), dtype=np.float32)


if __name__ == "__main__":
    embedder = EmbeddingModel()
    chunks = ["This is chunk 1", "This is chunk 2"]
    embeddings = embedder.embed_chunks(chunks)
    print(f"Embeddings shape: {embeddings.shape}")
