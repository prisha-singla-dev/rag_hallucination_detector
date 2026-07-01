from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=500)
    top_k: int = Field(default=4, ge=1, le=8)
    simulate_hallucination: bool = Field(
        default=True,
        description="When true, injects one unsupported sentence so the detector has something to catch in demos.",
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
    verdict: Literal["grounded", "unsupported"]
    support_score: float
    evidence: EvidenceChunk


class BaseDetectionResponse(BaseModel):
    question: str
    answer: str
    corrected_answer: str
    answer_confidence: float
    corrected_confidence: float
    confidence_delta: float
    hallucination_score: float
    claims: list[ClaimResult]
    retrieved_chunks: list[EvidenceChunk]


class QueryResponse(BaseDetectionResponse):
    pass


class DetectResponse(BaseDetectionResponse):
    pass
