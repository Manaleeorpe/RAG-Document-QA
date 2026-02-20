from PyPDF2 import PdfReader
from pathlib import Path
from typing import List


def extract_text_from_pdf(pdf_path:Path) -> str:
    #extract text
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def test_extraction(pdf_path:Path) -> str:
    text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(text)} characters")
    return text