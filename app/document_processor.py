from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(pdf_path: Path) -> List[Document]:
    """Load a PDF into LangChain Documents (one per page, with source/page metadata)."""
    return PyPDFLoader(str(pdf_path)).load()


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Concatenate all page text (kept for callers that want a plain string)."""
    return "".join(page.page_content for page in load_pdf(pdf_path))


def test_extraction(pdf_path: Path) -> str:
    text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(text)} characters")
    return text
