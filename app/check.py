# check.py
from embeddings import EmbeddingModel
from vector_store import VectorStore

store = VectorStore(EmbeddingModel().embeddings)
print("stats:", store.get_stats())
