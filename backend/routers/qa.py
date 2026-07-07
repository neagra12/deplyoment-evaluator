"""
QA router. Accepts a question plus manual_id, runs the LangGraph agent,
and returns a structured response with confidence, citation, and
escalation information when applicable.
"""
from fastapi import APIRouter, HTTPException
from database import get_connection
from agent.qa_agent import ask
from models import QueryRequest, QueryResponse, ChunkRef, ManualInfo

router = APIRouter(prefix="/qa", tags=["qa"])


@router.get("/manuals", response_model=list[ManualInfo])
def list_manuals():
    """List all ingested manuals."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, filename, title, equipment_type, total_pages, ingested_at FROM manuals"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.post("/ask", response_model=QueryResponse)
def ask_question(req: QueryRequest):
    """Submit a question and receive a grounded answer or escalation."""
    conn = get_connection()
    manual = conn.execute(
        "SELECT id FROM manuals WHERE id = ?", (req.manual_id,)
    ).fetchone()
    conn.close()
    if not manual:
        raise HTTPException(status_code=404, detail="Manual not found.")

    state = ask(req.question, req.manual_id)

    chunks = [
        ChunkRef(
            chunk_id=c["chunk_id"],
            page_number=c["page_number"],
            text=c["text"],
            similarity=round(c["similarity"], 4),
        )
        for c in state.get("retrieved_chunks", [])
    ]

    return QueryResponse(
        question=req.question,
        answer=state.get("answer"),
        confidence=state.get("confidence", 0.0),
        cited_page=state.get("cited_page"),
        retrieved_chunks=chunks,
        escalated=state.get("escalated", False),
        escalation_reason=state.get("escalation_reason"),
        suggested_next_step=state.get("suggested_next_step"),
        latency_ms=state.get("latency_ms", 0),
    )
