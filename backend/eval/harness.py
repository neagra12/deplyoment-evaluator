"""
Eval harness. Runs a set of eval questions through the QA agent,
scores each result, computes aggregate metrics, and persists everything
to the database. Every run is independently queryable.
"""
import json
import time
from datetime import datetime
from database import get_connection
from agent.qa_agent import ask
from eval.scoring import (
    score_retrieval,
    score_citation,
    score_hallucination,
    compute_calibration_error,
)


def run_eval(
    manual_id: int,
    label: str | None = None,
    question_ids: list[int] | None = None,
    dealership_id: int | None = None,
) -> int:
    """
    Run the eval harness for a manual. If question_ids is provided, only
    those questions are included (used by the dealership simulator).
    Returns the run_id.
    """
    conn = get_connection()

    if question_ids is not None:
        placeholders = ",".join("?" * len(question_ids))
        questions = conn.execute(
            f"SELECT * FROM eval_questions WHERE id IN ({placeholders})",
            question_ids,
        ).fetchall()
    else:
        questions = conn.execute(
            "SELECT * FROM eval_questions WHERE manual_id = ?",
            (manual_id,),
        ).fetchall()

    run_label = label or f"Eval run {datetime.utcnow().isoformat()}"
    cur = conn.execute(
        """INSERT INTO eval_runs (manual_id, label, dealership_id, total_questions)
           VALUES (?, ?, ?, ?)""",
        (manual_id, run_label, dealership_id, len(questions)),
    )
    run_id = cur.lastrowid
    conn.commit()

    raw_results: list[dict] = []

    for q in questions:
        q_id = q["id"]
        gt_page = q["ground_truth_page"]
        gt_chunk_id = q["ground_truth_chunk_id"]

        state = ask(q["question_text"], manual_id)

        retrieved = state["retrieved_chunks"]
        escalated = state["escalated"]
        answer = state.get("answer")
        confidence = state.get("confidence", 0.0)
        cited_page = state.get("cited_page")
        latency_ms = state.get("latency_ms", 0)
        escalation_reason = state.get("escalation_reason")

        retrieval_correct = score_retrieval(gt_page, retrieved)
        citation_correct = score_citation(gt_page, cited_page)

        hallucination_flag = None
        groundedness_score = None
        if answer and not escalated:
            scores = score_hallucination(q["question_text"], answer, retrieved)
            hallucination_flag = scores["hallucination_flag"]
            groundedness_score = scores["groundedness_score"]

        conn.execute(
            """INSERT INTO eval_results
               (run_id, question_id, agent_answer, escalated, escalation_reason,
                cited_page, confidence_score, retrieval_correct, citation_correct,
                hallucination_flag, groundedness_score, latency_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id, q_id, answer, int(escalated), escalation_reason,
                cited_page, confidence, int(retrieval_correct), int(citation_correct),
                int(hallucination_flag) if hallucination_flag is not None else None,
                groundedness_score, latency_ms,
            ),
        )
        conn.commit()

        raw_results.append({
            "escalated": escalated,
            "confidence_score": confidence,
            "retrieval_correct": retrieval_correct,
            "citation_correct": citation_correct,
            "hallucination_flag": hallucination_flag,
            "groundedness_score": groundedness_score,
            "latency_ms": latency_ms,
        })

    answered = [r for r in raw_results if not r["escalated"]]
    attempted = len(answered)
    total = len(raw_results)

    coverage = attempted / total if total > 0 else 0.0
    retrieval_acc = (
        sum(1 for r in answered if r["retrieval_correct"]) / attempted
        if attempted > 0 else 0.0
    )
    citation_acc = (
        sum(1 for r in answered if r["citation_correct"]) / attempted
        if attempted > 0 else 0.0
    )
    halluc_flagged = [r for r in answered if r["hallucination_flag"] is not None]
    hallucination_rate = (
        sum(1 for r in halluc_flagged if r["hallucination_flag"]) / len(halluc_flagged)
        if halluc_flagged else 0.0
    )
    grounded = [r for r in answered if r["groundedness_score"] is not None]
    avg_groundedness = (
        sum(r["groundedness_score"] for r in grounded) / len(grounded)
        if grounded else 0.0
    )
    avg_latency = (
        sum(r["latency_ms"] for r in raw_results) / total if total > 0 else 0.0
    )
    avg_confidence = (
        sum(r["confidence_score"] for r in answered) / attempted if attempted > 0 else 0.0
    )
    calibration_error = compute_calibration_error(raw_results)

    summary = {
        "coverage": round(coverage, 4),
        "retrieval_accuracy": round(retrieval_acc, 4),
        "citation_accuracy": round(citation_acc, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "avg_groundedness": round(avg_groundedness, 4),
        "avg_latency_ms": round(avg_latency, 1),
        "avg_confidence": round(avg_confidence, 4),
        "calibration_error": calibration_error,
    }

    conn.execute(
        """UPDATE eval_runs
           SET completed_at = CURRENT_TIMESTAMP,
               attempted = ?,
               summary_json = ?
           WHERE id = ?""",
        (attempted, json.dumps(summary), run_id),
    )
    conn.commit()
    conn.close()
    return run_id
