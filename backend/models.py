from pydantic import BaseModel
from typing import Optional, List, Any
from datetime import datetime


class ChunkRef(BaseModel):
    chunk_id: int
    page_number: int
    text: str
    similarity: float


class QueryRequest(BaseModel):
    question: str
    manual_id: int


class QueryResponse(BaseModel):
    question: str
    answer: Optional[str]
    confidence: float
    cited_page: Optional[int]
    retrieved_chunks: List[ChunkRef]
    escalated: bool
    escalation_reason: Optional[str]
    suggested_next_step: Optional[str]
    latency_ms: int


class ManualInfo(BaseModel):
    id: int
    filename: str
    title: Optional[str]
    equipment_type: Optional[str]
    total_pages: Optional[int]
    ingested_at: str


class EvalRunSummary(BaseModel):
    run_id: int
    label: Optional[str]
    dealership_id: Optional[int]
    total_questions: int
    attempted: int
    coverage: float
    retrieval_accuracy: float
    citation_accuracy: float
    hallucination_rate: float
    avg_groundedness: float
    avg_latency_ms: float
    avg_confidence: float
    calibration_error: float
    started_at: str
    completed_at: Optional[str]


class EvalResultDetail(BaseModel):
    result_id: int
    question_id: int
    category: str
    question_text: str
    ground_truth_answer: str
    ground_truth_page: Optional[int]
    agent_answer: Optional[str]
    escalated: bool
    escalation_reason: Optional[str]
    cited_page: Optional[int]
    confidence_score: Optional[float]
    retrieval_correct: Optional[bool]
    citation_correct: Optional[bool]
    hallucination_flag: Optional[bool]
    groundedness_score: Optional[float]
    latency_ms: Optional[int]


class DealershipMetrics(BaseModel):
    dealership_id: int
    name: str
    skew_category: str
    unanswered_pct: float
    avg_response_quality: float
    avg_confidence: float
    avg_latency_ms: float
    top_failure_categories: List[str]
    recommendation: str
    run_id: Optional[int]


class TroubleshootNode(BaseModel):
    id: str
    title: str
    description: str
    confidence: float
    cited_page: int
    cited_text: str
    question: Optional[str]
    branches: Optional[List[dict]]
    result: Optional[dict]
