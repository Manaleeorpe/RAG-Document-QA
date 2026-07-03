"""
test_embeddings.py — tests for app/embeddings.py :: EmbeddingModel

Categories covered
──────────────────
• Output dimension = 384  (all-MiniLM-L6-v2)
• Determinism — same input produces the same vector across calls
• Empty input handled gracefully
• Return type checks

The real SentenceTransformer model is loaded once per module (scope="module")
to keep the suite fast.  Mock-only tests are isolated to unit-level checks that
must not touch the network or GPU.
"""
import numpy as np
import pytest

from embeddings import EmbeddingModel


# ── Module-scoped fixture: real model, loaded once ────────────────────────────

@pytest.fixture(scope="module")
def embedder():
    """Load all-MiniLM-L6-v2 once for the entire module."""
    return EmbeddingModel()


# ── Output dimension = 384 ────────────────────────────────────────────────────

class TestOutputDimension:
    def test_embed_chunks_single_text_returns_1x384(self, embedder):
        result = embedder.embed_chunks(["Hello world"])
        assert result.shape == (1, 384), f"Expected (1, 384), got {result.shape}"

    def test_embed_chunks_batch_returns_Nx384(self, embedder):
        texts = ["First sentence.", "Second sentence.", "Third one."]
        result = embedder.embed_chunks(texts)
        assert result.shape == (3, 384), f"Expected (3, 384), got {result.shape}"

    def test_embed_query_returns_flat_384_vector(self, embedder):
        result = embedder.embed_query("What is the quarterly revenue?")
        assert result.shape == (384,), (
            f"embed_query must return a 1-D vector of 384 elements, got {result.shape}"
        )

    def test_embed_chunks_dtype_is_float(self, embedder):
        result = embedder.embed_chunks(["Revenue report Q3."])
        assert np.issubdtype(result.dtype, np.floating)

    def test_embed_query_dtype_is_float(self, embedder):
        result = embedder.embed_query("Quarterly earnings?")
        assert np.issubdtype(result.dtype, np.floating)


# ── Determinism ───────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_chunk_text_produces_identical_vector(self, embedder):
        text = "Revenue grew by 15% in Q3 2024."
        v1 = embedder.embed_chunks([text])
        v2 = embedder.embed_chunks([text])
        np.testing.assert_array_almost_equal(v1, v2, decimal=5)

    def test_same_query_produces_identical_vector(self, embedder):
        q = "What is the annual report about?"
        r1 = embedder.embed_query(q)
        r2 = embedder.embed_query(q)
        np.testing.assert_array_almost_equal(r1, r2, decimal=5)

    def test_different_texts_produce_different_vectors(self, embedder):
        v1 = embedder.embed_chunks(["Apple is a fruit."])
        v2 = embedder.embed_chunks(["The quarterly revenue increased significantly."])
        assert not np.allclose(v1, v2), "Semantically different texts must not collapse to the same vector"

    def test_batch_order_preserved(self, embedder):
        texts = ["Sentence A about apples.", "Sentence B about revenue."]
        batch = embedder.embed_chunks(texts)
        single_a = embedder.embed_chunks([texts[0]])
        single_b = embedder.embed_chunks([texts[1]])
        np.testing.assert_array_almost_equal(batch[0], single_a[0], decimal=5)
        np.testing.assert_array_almost_equal(batch[1], single_b[0], decimal=5)


# ── Empty input handling ──────────────────────────────────────────────────────

class TestEmptyInput:
    def test_empty_list_returns_array_with_zero_rows(self, embedder):
        result = embedder.embed_chunks([])
        assert result.shape[0] == 0, "Empty list input must produce an array with 0 rows"

    def test_empty_string_in_list_returns_valid_384_vector(self, embedder):
        result = embedder.embed_chunks([""])
        assert result.shape == (1, 384)

    def test_embed_query_empty_string_returns_valid_vector(self, embedder):
        result = embedder.embed_query("")
        assert result.shape == (384,)

    def test_empty_string_vector_is_finite(self, embedder):
        result = embedder.embed_query("")
        assert np.all(np.isfinite(result)), "Empty-string embedding must not contain NaN or Inf"


# ── Return type guards ────────────────────────────────────────────────────────

class TestReturnTypes:
    def test_embed_chunks_returns_ndarray(self, embedder):
        result = embedder.embed_chunks(["Test sentence."])
        assert isinstance(result, np.ndarray)

    def test_embed_query_returns_1d_ndarray(self, embedder):
        result = embedder.embed_query("Test query.")
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1, f"embed_query should be 1-D, got ndim={result.ndim}"

    def test_embed_chunks_values_are_normalised_range(self, embedder):
        """MiniLM outputs are not unit-normalised by default but values are bounded."""
        result = embedder.embed_chunks(["Range check sentence."])
        assert np.all(np.abs(result) < 10), "Embedding values should be in a reasonable numeric range"
