from __future__ import annotations

from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=500)
    top_k: int = Field(default=4, ge=1, le=8)
    simulate_hallucination: bool = Field(
        default=True,
        description=(
            "When true, asks the generator to slip one fabricated sentence into "
            "the answer (a genuine adversarial prompt against the LLM when "
            "GEMINI_API_KEY is set, or a fixed injected sentence in template "
            "mode) so the detector has something real to catch in demos."
        ),
    )

class DetectRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=500)
    answer: str = Field(..., min_length=5, max_length=4000)
    top_k: int = Field(default=4, ge=1, le=8)


class EvidenceChunk(BaseModel):
    chunk_id: str
    title: str
    source: str
    text: str
    score: float


class ClaimResult(BaseModel):
    claim: str
    verdict: Literal["grounded", "unsupported", "contradicted"]
    support_score: float
    evidence: EvidenceChunk
    grounding_method: Literal["lexical", "nli"]


class CorrectionStep(BaseModel):
    original_claim: str
    action: Literal["kept", "replaced", "removed"]
    rewritten_claim: str | None = None
    reason: str
    evidence: EvidenceChunk | None = None


class BaseDetectionResponse(BaseModel):
    run_id: str
    created_at: datetime
    question: str
    answer: str
    corrected_answer: str
    retrieval_mode: str
    generation_mode: Literal["llm", "template", "user_provided"]
    generation_model: str | None = None
    generation_warning: str | None = None
    grounding_mode: Literal["lexical", "nli"]
    answer_confidence: float
    corrected_confidence: float
    confidence_delta: float
    hallucination_score: float
    grounded_claim_count: int
    unsupported_claim_count: int
    contradicted_claim_count: int
    claims: list[ClaimResult]
    correction_steps: list[CorrectionStep]
    retrieved_chunks: list[EvidenceChunk]


class QueryResponse(BaseDetectionResponse):
    pass


class DetectResponse(BaseDetectionResponse):
    pass


class RunRecord(BaseModel):
    run_id: str
    created_at: datetime
    mode: Literal["query", "detect"]
    question: str
    hallucination_score: float
    answer_confidence: float
    corrected_confidence: float
    unsupported_claim_count: int
    contradicted_claim_count: int
    retrieval_mode: str
    grounding_mode: Literal["lexical", "nli"]


class RunHistoryResponse(BaseModel):
    total_runs: int
    runs: list[RunRecord]


class AnalyticsSummaryResponse(BaseModel):
    total_runs: int
    query_runs: int
    detect_runs: int
    avg_hallucination_score: float
    avg_confidence_delta: float
    latest_run: RunRecord | None
