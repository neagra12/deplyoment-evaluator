"""
LangGraph QA agent. State flows through three nodes:

  retrieve -> generate -> [escalate | END]

The generate node asks GPT-4o for a structured answer that includes
a self-reported confidence score. If confidence is below the threshold,
the conditional edge routes to the escalate node instead of END.
"""
import os
import time
import json
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, END
from openai import OpenAI

from agent.retrieval import retrieve_chunks

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))

SYSTEM_PROMPT = """You are a technical service expert for Honda generator equipment.
You answer questions from dealers and service technicians by citing the shop manual exactly.

Rules:
1. Only use information present in the provided manual passages.
2. Never invent specifications, torques, part numbers, or procedures.
3. If the passages do not contain enough information to answer reliably, say so
   and set confidence below 0.65.
4. Always identify the page number your answer primarily draws from."""

GENERATION_PROMPT = """\
Manual passages (each labeled with its page number):

{context}

Technician question: {question}

Respond with a JSON object and nothing else:
{{
  "answer": "<your answer, written for a trained technician>",
  "confidence": <float 0.0 to 1.0>,
  "confidence_reasoning": "<one sentence explaining why you are this confident>",
  "cited_page": <integer page number, or null>,
  "escalation_reason": <null, or a short string if confidence is below 0.65>
}}

Confidence calibration guide:
  0.90 and above: answer is stated explicitly and completely in the passages
  0.70 to 0.89: answer is clearly implied or reliably inferred
  0.50 to 0.69: answer requires interpretation or passages are incomplete
  below 0.50: answer cannot be reliably determined from the provided text"""


class AgentState(TypedDict):
    query: str
    manual_id: int
    retrieved_chunks: list
    answer: Optional[str]
    confidence: float
    escalated: bool
    escalation_reason: Optional[str]
    suggested_next_step: Optional[str]
    cited_page: Optional[int]
    latency_start: float
    latency_ms: int


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        header = f" [{c['section_header']}]" if c.get("section_header") else ""
        parts.append(f"[Page {c['page_number']}{header}]\n{c['text']}")
    return "\n\n".join(parts)


def retrieve_node(state: AgentState) -> AgentState:
    chunks = retrieve_chunks(state["query"], state["manual_id"])
    return {**state, "retrieved_chunks": chunks}


def generate_node(state: AgentState) -> AgentState:
    context = _build_context(state["retrieved_chunks"])
    prompt = GENERATION_PROMPT.format(
        context=context,
        question=state["query"],
    )

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {
            "answer": raw,
            "confidence": 0.0,
            "confidence_reasoning": "Failed to parse structured response.",
            "cited_page": None,
            "escalation_reason": "Internal parsing error.",
        }

    elapsed = int((time.time() - state["latency_start"]) * 1000)
    return {
        **state,
        "answer": parsed.get("answer"),
        "confidence": float(parsed.get("confidence", 0.0)),
        "cited_page": parsed.get("cited_page"),
        "escalation_reason": parsed.get("escalation_reason"),
        "latency_ms": elapsed,
    }


def escalate_node(state: AgentState) -> AgentState:
    reason = state.get("escalation_reason") or "Confidence below threshold."
    confidence = state["confidence"]

    if confidence < 0.3:
        next_step = (
            "Escalate to Honda factory technical support. "
            "The manual does not contain sufficient information to answer this question."
        )
    elif confidence < 0.5:
        next_step = (
            "Consult a certified Honda service technician. "
            "Review related sections of the shop manual before proceeding."
        )
    else:
        next_step = (
            "Cross-reference the shop manual sections closest to this topic. "
            "If still unclear, contact the regional service representative."
        )

    return {
        **state,
        "escalated": True,
        "answer": None,
        "suggested_next_step": next_step,
        "escalation_reason": reason,
    }


def _should_escalate(state: AgentState) -> str:
    if state["confidence"] < CONFIDENCE_THRESHOLD:
        return "escalate"
    return "end"


def build_qa_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("escalate", escalate_node)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_conditional_edges(
        "generate",
        _should_escalate,
        {"escalate": "escalate", "end": END},
    )
    workflow.add_edge("escalate", END)
    return workflow.compile()


QA_GRAPH = build_qa_graph()


def ask(question: str, manual_id: int) -> dict:
    """
    Public entry point. Returns a dict matching the QueryResponse model fields.
    """
    initial_state: AgentState = {
        "query": question,
        "manual_id": manual_id,
        "retrieved_chunks": [],
        "answer": None,
        "confidence": 0.0,
        "escalated": False,
        "escalation_reason": None,
        "suggested_next_step": None,
        "cited_page": None,
        "latency_start": time.time(),
        "latency_ms": 0,
    }
    final_state = QA_GRAPH.invoke(initial_state)
    return final_state
