"""
PDF parser. Extracts text from each page, splits into overlapping chunks,
and preserves page number and nearest section header for every chunk.
"""
import re
import hashlib
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF


CHUNK_SIZE = 400      # target characters per chunk
CHUNK_OVERLAP = 80    # overlap between consecutive chunks


def _detect_header(text: str) -> str | None:
    """Return the first line that looks like a section heading, or None."""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if (
            line.isupper()
            or re.match(r"^\d+\.\s+[A-Z]", line)
            or re.match(r"^[IVXLC]+\.\s+[A-Z]", line)
        ):
            return line
    return None


def _split_page(text: str, page_number: int, section_header: str | None) -> list[dict]:
    """Split a single page's text into overlapping chunks."""
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    chunk_index = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "page_number": page_number,
                "chunk_index": chunk_index,
                "text": chunk_text,
                "section_header": section_header,
                "token_count": len(chunk_text.split()),
            })
            chunk_index += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def parse_pdf(filepath: str) -> dict:
    """
    Open a PDF and return a dict with:
      - checksum: SHA256 of file bytes
      - total_pages: int
      - chunks: list of chunk dicts ready for database insertion
    """
    path = Path(filepath)
    file_bytes = path.read_bytes()
    checksum = hashlib.sha256(file_bytes).hexdigest()

    doc = fitz.open(str(path))
    total_pages = doc.page_count

    all_chunks: list[dict] = []
    current_header: str | None = None

    for page_index in range(total_pages):
        page = doc[page_index]
        text = page.get_text()
        page_number = page_index + 1  # 1-based

        detected = _detect_header(text)
        if detected:
            current_header = detected

        page_chunks = _split_page(text, page_number, current_header)
        all_chunks.extend(page_chunks)

    doc.close()

    return {
        "checksum": checksum,
        "total_pages": total_pages,
        "chunks": all_chunks,
    }
