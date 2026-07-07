"""
Per-result scoring functions used by the eval harness.

retrieval_correct: True if the ground truth chunk page appears among the
                   top-K retrieved chunk pages.
citation_correct:  True if the agent's cited page matches ground truth page.
hallucination and groundedness: determined by an LLM judge call to GPT-4o.
"""
import json
import os
from openai import OpenAI

JUDGE_PROMPT = """\
You are an impartial evaluator checking whether an AI answer is grounded in the
provided source passages.

Source passages:
{context}

Question: {question}

AI answer: {answer}

Evaluate the answer and respond with a JSON object and nothing else:
{{
  "hallucination_flag": <true if the answer contains any claim not supported by the passages, false otherwise>,
  "groundedness_score": <float 0.0 to 1.0, where 1.0 means every claim is directly supported>,
  "explanation": "<one sentence>"
}}"""


def score_retrieval(ground_truth_page: int | None, retrieved_chunks: list[dict]) -> bool:
    """True if any retrieved chunk is from the ground truth page."""
    if ground_truth_page is None:
        return False
    retrieved_pages = {c["page_number"] for c in retrieved_chunks}
    return ground_truth_page in retrieved_pages


def score_citation(ground_truth_page: int | None, cited_page: int | None) -> bool:
    """True if the agent's cited page matches the ground truth page."""
    if ground_truth_page is None or cited_page is None:
        return False
    return ground_truth_page == cited_page


def score_hallucination(
    question: str,
    answer: str,
    retrieved_chunks: list[dict],
) -> dict:
    """
    Returns {"hallucination_flag": bool, "groundedness_score": float}.
    Uses GPT-4o as a judge. Falls back to safe defaults on error.
    """
    context = "\n\n".join(
        f"[Page {c['page_number']}] {c['text']}" for c in retrieved_chunks
    )
    prompt = JUDGE_PROMPT.format(
        context=context,
        question=question,
        answer=answer,
    )
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content)
        return {
            "hallucination_flag": bool(parsed.get("hallucination_flag", False)),
            "groundedness_score": float(parsed.get("groundedness_score", 0.0)),
        }
    except Exception:
        return {"hallucination_flag": False, "groundedness_score": 0.0}


def compute_calibration_error(results: list[dict]) -> float:
    """
    Compute Expected Calibration Error (ECE) across answered results.

    Bucket results by confidence into 5 bins. Within each bin, compare
    mean confidence to actual accuracy (citation correct rate).
    ECE is the weighted mean absolute difference.
    """
    answered = [
        r for r in results
        if not r["escalated"] and r["confidence_score"] is not None
    ]
    if not answered:
        return 0.0

    buckets: dict[int, list] = {i: [] for i in range(5)}
    for r in answered:
        idx = min(int(r["confidence_score"] * 5), 4)
        correct = 1 if r["citation_correct"] else 0
        buckets[idx].append((r["confidence_score"], correct))

    total = len(answered)
    ece = 0.0
    for items in buckets.values():
        if not items:
            continue
        mean_conf = sum(x[0] for x in items) / len(items)
        mean_acc = sum(x[1] for x in items) / len(items)
        ece += (len(items) / total) * abs(mean_conf - mean_acc)

    return round(ece, 4)
