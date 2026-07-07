"""
Eval router. Endpoints to generate questions, run the harness, retrieve
results, and export a JSON report.
"""
import json
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from database import get_connection
from eval.question_gen import generate_questions_for_manual
from eval.harness import run_eval
from models import EvalRunSummary, EvalResultDetail

router = APIRouter(prefix="/eval", tags=["eval"])


class GenerateQuestionsRequest(BaseModel):
    manual_id: int


class RunEvalRequest(BaseModel):
    manual_id: int
    label: str | None = None


@router.post("/generate-questions")
def generate_questions(req: GenerateQuestionsRequest):
    """Generate eval questions for a manual. Takes 1 to 3 minutes."""
    conn = get_connection()
    manual = conn.execute(
        "SELECT id FROM manuals WHERE id = ?", (req.manual_id,)
    ).fetchone()
    conn.close()
    if not manual:
        raise HTTPException(status_code=404, detail="Manual not found.")

    count = generate_questions_for_manual(req.manual_id)
    return {"manual_id": req.manual_id, "questions_generated": count}


@router.post("/run")
def run_eval_endpoint(req: RunEvalRequest):
    """
    Run the eval harness synchronously. Takes several minutes for 150 questions
    due to LLM calls. Returns the run_id on completion.
    """
    conn = get_connection()
    q_count = conn.execute(
        "SELECT COUNT(*) as n FROM eval_questions WHERE manual_id = ?",
        (req.manual_id,),
    ).fetchone()["n"]
    conn.close()

    if q_count == 0:
        raise HTTPException(
            status_code=400,
            detail="No eval questions found. Run /eval/generate-questions first.",
        )

    run_id = run_eval(manual_id=req.manual_id, label=req.label)
    return {"run_id": run_id}


@router.get("/runs", response_model=list[EvalRunSummary])
def list_runs(manual_id: int | None = None):
    """List all eval runs, optionally filtered by manual."""
    conn = get_connection()
    if manual_id:
        rows = conn.execute(
            "SELECT * FROM eval_runs WHERE manual_id = ? AND dealership_id IS NULL ORDER BY started_at DESC",
            (manual_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM eval_runs WHERE dealership_id IS NULL ORDER BY started_at DESC"
        ).fetchall()
    conn.close()

    results = []
    for row in rows:
        summary = json.loads(row["summary_json"]) if row["summary_json"] else {}
        results.append(EvalRunSummary(
            run_id=row["id"],
            label=row["label"],
            dealership_id=row["dealership_id"],
            total_questions=row["total_questions"] or 0,
            attempted=row["attempted"] or 0,
            coverage=summary.get("coverage", 0.0),
            retrieval_accuracy=summary.get("retrieval_accuracy", 0.0),
            citation_accuracy=summary.get("citation_accuracy", 0.0),
            hallucination_rate=summary.get("hallucination_rate", 0.0),
            avg_groundedness=summary.get("avg_groundedness", 0.0),
            avg_latency_ms=summary.get("avg_latency_ms", 0.0),
            avg_confidence=summary.get("avg_confidence", 0.0),
            calibration_error=summary.get("calibration_error", 0.0),
            started_at=row["started_at"],
            completed_at=row["completed_at"],
        ))
    return results


@router.get("/runs/{run_id}", response_model=EvalRunSummary)
def get_run(run_id: int):
    conn = get_connection()
    row = conn.execute("SELECT * FROM eval_runs WHERE id = ?", (run_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Run not found.")
    summary = json.loads(row["summary_json"]) if row["summary_json"] else {}
    return EvalRunSummary(
        run_id=row["id"],
        label=row["label"],
        dealership_id=row["dealership_id"],
        total_questions=row["total_questions"] or 0,
        attempted=row["attempted"] or 0,
        coverage=summary.get("coverage", 0.0),
        retrieval_accuracy=summary.get("retrieval_accuracy", 0.0),
        citation_accuracy=summary.get("citation_accuracy", 0.0),
        hallucination_rate=summary.get("hallucination_rate", 0.0),
        avg_groundedness=summary.get("avg_groundedness", 0.0),
        avg_latency_ms=summary.get("avg_latency_ms", 0.0),
        avg_confidence=summary.get("avg_confidence", 0.0),
        calibration_error=summary.get("calibration_error", 0.0),
        started_at=row["started_at"],
        completed_at=row["completed_at"],
    )


@router.get("/runs/{run_id}/results", response_model=list[EvalResultDetail])
def get_results(run_id: int, category: str | None = None):
    """Per-question results for a run, optionally filtered by category."""
    conn = get_connection()
    base_query = """
        SELECT er.id as result_id, er.*, eq.category, eq.question_text,
               eq.ground_truth_answer, eq.ground_truth_page
        FROM eval_results er
        JOIN eval_questions eq ON eq.id = er.question_id
        WHERE er.run_id = ?
    """
    params: list = [run_id]
    if category:
        base_query += " AND eq.category = ?"
        params.append(category)
    rows = conn.execute(base_query, params).fetchall()
    conn.close()

    return [
        EvalResultDetail(
            result_id=r["result_id"],
            question_id=r["question_id"],
            category=r["category"],
            question_text=r["question_text"],
            ground_truth_answer=r["ground_truth_answer"],
            ground_truth_page=r["ground_truth_page"],
            agent_answer=r["agent_answer"],
            escalated=bool(r["escalated"]),
            escalation_reason=r["escalation_reason"],
            cited_page=r["cited_page"],
            confidence_score=r["confidence_score"],
            retrieval_correct=bool(r["retrieval_correct"]) if r["retrieval_correct"] is not None else None,
            citation_correct=bool(r["citation_correct"]) if r["citation_correct"] is not None else None,
            hallucination_flag=bool(r["hallucination_flag"]) if r["hallucination_flag"] is not None else None,
            groundedness_score=r["groundedness_score"],
            latency_ms=r["latency_ms"],
        )
        for r in rows
    ]


@router.get("/runs/{run_id}/export")
def export_run(run_id: int):
    """Export the full eval run as raw JSON."""
    conn = get_connection()
    run = conn.execute("SELECT * FROM eval_runs WHERE id = ?", (run_id,)).fetchone()
    if not run:
        conn.close()
        raise HTTPException(status_code=404, detail="Run not found.")

    results = conn.execute(
        """SELECT er.*, eq.category, eq.question_text, eq.ground_truth_answer,
                  eq.ground_truth_page
           FROM eval_results er
           JOIN eval_questions eq ON eq.id = er.question_id
           WHERE er.run_id = ?""",
        (run_id,),
    ).fetchall()
    conn.close()

    payload = {
        "run": {
            "id": run["id"],
            "label": run["label"],
            "started_at": run["started_at"],
            "completed_at": run["completed_at"],
            "total_questions": run["total_questions"],
            "attempted": run["attempted"],
            "summary": json.loads(run["summary_json"]) if run["summary_json"] else {},
        },
        "results": [dict(r) for r in results],
    }
    return JSONResponse(content=payload)
