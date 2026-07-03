"""
conftest.py — shared fixtures for the entire test suite.

pytest auto-discovers and injects these fixtures by parameter name.
mocker is provided by pytest-mock (pip install pytest-mock).
"""
import sys
import os

import numpy as np
import pytest

# Make app/ importable from every test file without modifying PYTHONPATH externally.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))


# ── Email body fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def normal_email_text():
    return (
        "From: alice@example.com\n"
        "To: bob@example.com\n"
        "Subject: Q3 Financial Report\n\n"
        "Hi Bob,\n\n"
        "Please find the Q3 financial summary below. Revenue grew by 15% compared "
        "to last quarter, driven mainly by enterprise subscriptions.\n\n"
        "Best regards,\nAlice"
    )


@pytest.fixture
def long_thread_text():
    """Generates a realistic long email thread of ≥ 50 000 characters."""
    segment = (
        "Re: Budget Review — message with financial data. "
        "Revenue figures show positive trends this quarter. All departments are "
        "on track with their allocations and no major deviations are expected. "
    )
    return segment * 600  # ≈ 54 000 chars


@pytest.fixture
def html_email_text():
    return (
        "<html><body>"
        "<p>Dear Team,</p>"
        "<p>Please review the <b>attached Q3 report</b> at your earliest convenience.</p>"
        "<br/>"
        "<p>Best regards,<br/>Alice</p>"
        "</body></html>"
    )


# ── Shared mock helpers ───────────────────────────────────────────────────────

@pytest.fixture
def fake_embeddings_384():
    """Deterministic (3, 384) float32 array usable as a drop-in embedding batch."""
    rng = np.random.default_rng(42)
    return rng.random((3, 384)).astype(np.float32)


@pytest.fixture
def mock_embedder(mocker, fake_embeddings_384):
    """MagicMock that mimics EmbeddingModel without loading a real model."""
    mock = mocker.MagicMock()
    mock.embed_chunks.return_value = fake_embeddings_384
    mock.embed_query.return_value = fake_embeddings_384[0]
    return mock


@pytest.fixture
def mock_vector_store(mocker):
    """MagicMock that mimics VectorStore.search() returning two plausible results."""
    mock = mocker.MagicMock()
    mock.search.return_value = {
        "documents": [["chunk text one", "chunk text two"]],
        "distances": [[0.1, 0.3]],
        "metadatas": [
            [
                {"doc_name": "test_doc", "chunk_id": 0},
                {"doc_name": "test_doc", "chunk_id": 1},
            ]
        ],
    }
    return mock
