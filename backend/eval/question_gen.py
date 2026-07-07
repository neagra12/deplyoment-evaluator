"""
Automatic eval question generation.

Given the ingested chunks for a manual, generates 100 to 200 realistic
dealer and technician questions across five categories. For each question,
a grounded answer and the source page number are generated at the same time,
providing the ground truth needed to score the QA agent later.
"""
import json
import os
from openai import OpenAI
from database import get_connection

CATEGORIES = [
    "parts_compatibility",
    "installation",
    "torque_spec",
    "troubleshooting",
    "warranty_maintenance",
]

TARGET_PER_CATEGORY = 30  # 5 categories x 30 = 150 questions


QUESTION_GEN_PROMPT = """\
You are generating realistic evaluation questions for a Honda Generator E/ES3500 service manual
to test an AI-powered product expert used by dealership staff and service technicians.

Category: {category}
Category description: {description}

Manual passages provided as context (each labeled with page number and section):
{context}

Generate exactly {count} distinct, realistic questions that a dealer or service technician
would actually ask when servicing this equipment. The questions must be answerable from the
provided passages.

For each question, also provide the ground truth answer (grounded only in the passages)
and the page number the answer primarily comes from.

Respond with a JSON array. Each element must have:
  "question": string,
  "answer": string,
  "page": integer

Do not include any text outside the JSON array."""

CATEGORY_DESCRIPTIONS = {
    "parts_compatibility": (
        "Questions about which parts fit which model variants (K0 vs K1, E3500 vs ES3500), "
        "part interchangeability, and component substitutions."
    ),
    "installation": (
        "Questions about assembly, removal, and installation procedures for components "
        "such as the belt, generator, control box, and fuel system."
    ),
    "torque_spec": (
        "Questions about specific torque values, resistance specifications, belt tension, "
        "and other numeric tolerances from the specifications or torques table."
    ),
    "troubleshooting": (
        "Questions about diagnosing faults: no AC output, unit will not start, "
        "circuit breaker behavior, and component test procedures."
    ),
    "warranty_maintenance": (
        "Questions about the maintenance schedule, service intervals, recommended parts, "
        "and which components require periodic inspection or replacement."
    ),
}


def _build_context_for_category(manual_id: int, category: str) -> str:
    """
    Pull all chunks for this manual. For some categories we weight pages
    that are most relevant, but we always include all chunks so the LLM
    can make accurate page citations.
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT page_number, section_header, text FROM chunks "
        "WHERE manual_id = ? ORDER BY page_number, chunk_index",
        (manual_id,),
    ).fetchall()
    conn.close()

    parts = []
    for row in rows:
        header = f" [{row['section_header']}]" if row["section_header"] else ""
        parts.append(f"[Page {row['page_number']}{header}]\n{row['text']}")

    # Limit to avoid exceeding context window: 80 chunks max (roughly 32k tokens)
    selected = parts[:80]
    return "\n\n".join(selected)


def generate_questions_for_manual(manual_id: int) -> int:
    """
    Generate eval questions for all categories and store in the database.
    Returns the total number of questions inserted.
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    conn = get_connection()
    total_inserted = 0

    for category in CATEGORIES:
        context = _build_context_for_category(manual_id, category)
        description = CATEGORY_DESCRIPTIONS[category]
        prompt = QUESTION_GEN_PROMPT.format(
            category=category,
            description=description,
            context=context,
            count=TARGET_PER_CATEGORY,
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        try:
            parsed = json.loads(raw)
            # The model may return {"questions": [...]} or a bare array.
            if isinstance(parsed, list):
                items = parsed
            else:
                items = next(
                    (v for v in parsed.values() if isinstance(v, list)), []
                )
        except Exception:
            items = []

        for item in items:
            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()
            page = item.get("page")
            if not question or not answer:
                continue

            # Find the closest chunk on that page to link ground truth chunk.
            chunk_id = None
            if page:
                row = conn.execute(
                    "SELECT id FROM chunks WHERE manual_id = ? AND page_number = ? LIMIT 1",
                    (manual_id, page),
                ).fetchone()
                if row:
                    chunk_id = row["id"]

            conn.execute(
                """INSERT INTO eval_questions
                   (manual_id, category, question_text, ground_truth_answer,
                    ground_truth_page, ground_truth_chunk_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (manual_id, category, question, answer, page, chunk_id),
            )
            total_inserted += 1

        conn.commit()

    conn.close()
    return total_inserted
