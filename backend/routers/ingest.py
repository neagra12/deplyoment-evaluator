"""
Ingest router. Accepts a PDF file path or uploaded file, runs the parser,
stores chunks in the database, and computes embeddings.
"""
import os
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from pathlib import Path
import tempfile

from database import get_connection
from ingestion.pdf_parser import parse_pdf
from ingestion.embeddings import embed_texts, serialize_embedding

router = APIRouter(prefix="/ingest", tags=["ingest"])


class IngestPathRequest(BaseModel):
    filepath: str
    title: str | None = None
    equipment_type: str | None = None


@router.post("/path")
def ingest_from_path(req: IngestPathRequest):
    """Ingest a PDF from an absolute file path on the server."""
    if not Path(req.filepath).exists():
        raise HTTPException(status_code=404, detail="File not found at the given path.")
    return _run_ingest(req.filepath, req.title, req.equipment_type)


@router.post("/upload")
async def ingest_from_upload(
    file: UploadFile = File(...),
    title: str | None = None,
    equipment_type: str | None = None,
):
    """Ingest a PDF uploaded directly via multipart form."""
    suffix = Path(file.filename or "manual.pdf").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        return _run_ingest(tmp_path, title or file.filename, equipment_type)
    finally:
        os.unlink(tmp_path)


def _run_ingest(filepath: str, title: str | None, equipment_type: str | None) -> dict:
    parsed = parse_pdf(filepath)
    checksum = parsed["checksum"]

    conn = get_connection()
    existing = conn.execute(
        "SELECT id FROM manuals WHERE checksum = ?", (checksum,)
    ).fetchone()
    if existing:
        conn.close()
        return {
            "manual_id": existing["id"],
            "status": "already_ingested",
            "total_chunks": 0,
        }

    filename = Path(filepath).name
    cur = conn.execute(
        """INSERT INTO manuals (filename, title, equipment_type, total_pages, checksum)
           VALUES (?, ?, ?, ?, ?)""",
        (filename, title or filename, equipment_type, parsed["total_pages"], checksum),
    )
    manual_id = cur.lastrowid
    conn.commit()

    chunks = parsed["chunks"]
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    for chunk, emb in zip(chunks, embeddings):
        conn.execute(
            """INSERT INTO chunks
               (manual_id, page_number, chunk_index, text, embedding, section_header, token_count)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                manual_id,
                chunk["page_number"],
                chunk["chunk_index"],
                chunk["text"],
                serialize_embedding(emb),
                chunk["section_header"],
                chunk["token_count"],
            ),
        )

    conn.commit()
    conn.close()

    return {
        "manual_id": manual_id,
        "status": "ingested",
        "total_pages": parsed["total_pages"],
        "total_chunks": len(chunks),
    }
