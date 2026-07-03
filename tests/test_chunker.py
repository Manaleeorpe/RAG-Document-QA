"""
test_chunker.py — tests for app/chunker.py :: chunk_text()

Categories covered
──────────────────
• Normal email body
• Empty / whitespace-only body
• Very long thread (≥ 50 000 chars)
• Non-English text  (parametrize)
• HTML stripping (chunker does NOT strip; tests document both paths)
• Attachment-only email
• Data-driven edge cases via @pytest.mark.parametrize
"""
import re

import pytest

from chunker import chunk_text


# ── Normal email ──────────────────────────────────────────────────────────────

class TestNormalEmail:
    def test_produces_at_least_one_chunk(self, normal_email_text):
        chunks = chunk_text(normal_email_text, "email.txt", "email")
        assert len(chunks) >= 1

    def test_all_required_metadata_keys_present(self, normal_email_text):
        chunks = chunk_text(normal_email_text, "email.txt", "email")
        required = {"source", "doc_name", "chunk_id", "char_count", "total_chunks", "uploaded_at"}
        for chunk in chunks:
            assert required.issubset(chunk.metadata.keys())

    def test_source_and_doc_name_match_args(self, normal_email_text):
        chunks = chunk_text(normal_email_text, "email.txt", "email")
        assert chunks[0].metadata["source"] == "email.txt"
        assert chunks[0].metadata["doc_name"] == "email"

    def test_chunk_ids_are_sequential_from_zero(self, normal_email_text):
        chunks = chunk_text(normal_email_text, "email.txt", "email")
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_id"] == i

    def test_total_chunks_consistent_across_all_chunks(self, normal_email_text):
        chunks = chunk_text(normal_email_text, "email.txt", "email")
        expected = len(chunks)
        for chunk in chunks:
            assert chunk.metadata["total_chunks"] == expected

    def test_char_count_matches_page_content_length(self, normal_email_text):
        chunks = chunk_text(normal_email_text, "email.txt", "email")
        for chunk in chunks:
            assert chunk.metadata["char_count"] == len(chunk.page_content)

    def test_page_content_is_non_empty_string(self, normal_email_text):
        chunks = chunk_text(normal_email_text, "email.txt", "email")
        for chunk in chunks:
            assert isinstance(chunk.page_content, str)
            assert len(chunk.page_content) > 0


# ── Empty body ────────────────────────────────────────────────────────────────

class TestEmptyBody:
    def test_empty_string_returns_empty_list(self):
        assert chunk_text("", "empty.txt", "empty") == []

    def test_whitespace_only_body_returns_empty_list(self):
        assert chunk_text("   \n\n\t  ", "ws.txt", "ws") == []

    def test_single_space_returns_empty_list(self):
        assert chunk_text(" ", "sp.txt", "sp") == []

    def test_newlines_only_returns_empty_list(self):
        assert chunk_text("\n\n\n", "nl.txt", "nl") == []


# ── Very long thread ──────────────────────────────────────────────────────────

class TestLongThread:
    def test_fixture_is_at_least_50k_chars(self, long_thread_text):
        assert len(long_thread_text) >= 50_000

    def test_long_thread_produces_multiple_chunks(self, long_thread_text):
        chunks = chunk_text(long_thread_text, "thread.txt", "thread", chunk_size=500)
        assert len(chunks) > 1

    def test_no_single_chunk_massively_exceeds_chunk_size(self, long_thread_text):
        chunk_size = 500
        chunks = chunk_text(
            long_thread_text, "thread.txt", "thread", chunk_size=chunk_size
        )
        # LangChain's RecursiveCharacterTextSplitter may marginally exceed
        # chunk_size at hard boundaries — allow a reasonable tolerance.
        for chunk in chunks:
            assert chunk.metadata["char_count"] <= chunk_size + 60

    def test_total_chunks_metadata_matches_list_length(self, long_thread_text):
        chunks = chunk_text(long_thread_text, "thread.txt", "thread")
        for chunk in chunks:
            assert chunk.metadata["total_chunks"] == len(chunks)

    def test_overlap_does_not_create_duplicate_chunk_ids(self, long_thread_text):
        chunks = chunk_text(
            long_thread_text, "thread.txt", "thread", chunk_size=500, chunk_overlap=100
        )
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert ids == list(range(len(chunks)))


# ── Non-English text (data-driven with parametrize) ───────────────────────────

@pytest.mark.parametrize(
    "text, filename, doc_name",
    [
        pytest.param(
            "Cher Jean,\n\nRapport financier Q3. Revenus +15%.\n\nCordialement,\nMarie",
            "fr.txt",
            "fr_email",
            id="french",
        ),
        pytest.param(
            "亲爱的团队，\n\n请查看本季度财务报告。收入增长了15%。\n\n此致，\n王明",
            "zh.txt",
            "zh_email",
            id="chinese",
        ),
        pytest.param(
            "प्रिय टीम,\n\nकृपया तिमाही रिपोर्ट देखें। राजस्व 15% बढ़ा।\n\nधन्यवाद,\nराज",
            "hi.txt",
            "hi_email",
            id="hindi",
        ),
        pytest.param(
            "Estimado equipo,\n\nReporte Q3: ingresos +15%.\n\nAtentamente,\nMaria",
            "es.txt",
            "es_email",
            id="spanish",
        ),
    ],
)
def test_non_english_text_produces_valid_chunks(text, filename, doc_name):
    chunks = chunk_text(text, filename, doc_name)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert isinstance(chunk.page_content, str)
        assert len(chunk.page_content) > 0


# ── HTML email ────────────────────────────────────────────────────────────────

class TestHTMLEmail:
    def test_chunker_does_not_strip_html_tags(self, html_email_text):
        """chunk_text is tag-agnostic; raw HTML tags survive into page_content."""
        chunks = chunk_text(html_email_text, "html.txt", "html_email")
        combined = "".join(c.page_content for c in chunks)
        assert "<" in combined, "HTML tags should be preserved (chunker does not strip)"

    def test_html_email_produces_at_least_one_chunk(self, html_email_text):
        chunks = chunk_text(html_email_text, "html.txt", "html_email")
        assert len(chunks) >= 1

    def test_pre_strip_html_before_chunking_removes_tags(self, html_email_text):
        """
        Recommended pipeline: strip HTML *before* calling chunk_text.
        This test documents the correct pre-processing step.
        """
        clean_text = re.sub(r"<[^>]+>", " ", html_email_text).strip()
        chunks = chunk_text(clean_text, "html.txt", "html_email")
        combined = "".join(c.page_content for c in chunks)
        assert "<" not in combined
        assert ">" not in combined

    def test_stripped_html_still_contains_visible_words(self, html_email_text):
        clean_text = re.sub(r"<[^>]+>", " ", html_email_text).strip()
        chunks = chunk_text(clean_text, "html.txt", "html_email")
        combined = " ".join(c.page_content for c in chunks)
        assert "Dear Team" in combined or "Q3" in combined or "Alice" in combined


# ── Attachment-only email ─────────────────────────────────────────────────────

class TestAttachmentOnlyEmail:
    def test_attachment_placeholder_produces_at_least_one_chunk(self):
        text = "[Attachment: quarterly_report.pdf]\n[Attachment: budget_2024.xlsx]"
        chunks = chunk_text(text, "attach.txt", "attach_email")
        assert len(chunks) >= 1

    def test_attachment_keyword_preserved_in_chunk_content(self):
        text = "[Attachment: report.pdf]"
        chunks = chunk_text(text, "attach.txt", "attach_email")
        assert "[Attachment:" in chunks[0].page_content

    def test_attachment_chunk_has_correct_doc_name(self):
        text = "[Attachment: report.pdf]"
        chunks = chunk_text(text, "attach.txt", "attach_email")
        assert chunks[0].metadata["doc_name"] == "attach_email"


# ── Data-driven edge cases ────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "text, filename, doc_name, min_chunks, max_chunks",
    [
        pytest.param("Short email body.", "a.txt", "a", 1, 1, id="short_text"),
        pytest.param("Word " * 200, "b.txt", "b", 1, 5, id="~1000_chars"),
        pytest.param(
            "Sentence one. Sentence two. " * 40, "c.txt", "c", 1, 20, id="medium_text"
        ),
        pytest.param(
            "Re: FWD: Q3. " * 5, "d.txt", "d", 1, 3, id="short_thread_prefix"
        ),
    ],
)
def test_parametrised_chunk_count_bounds(text, filename, doc_name, min_chunks, max_chunks):
    chunks = chunk_text(text, filename, doc_name)
    assert min_chunks <= len(chunks) <= max_chunks, (
        f"Expected [{min_chunks}, {max_chunks}] chunks, got {len(chunks)} for input length {len(text)}"
    )
