from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.pipeline import HallucinationPipeline
from backend.core.schemas import DetectRequest, DetectResponse, QueryRequest, QueryResponse


app = FastAPI(
    title="RAG Hallucination Detector",
    description="A free MVP for detecting and auto-correcting unsupported claims in a RAG answer.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = HallucinationPipeline()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/knowledge-base")
def knowledge_base() -> dict[str, object]:
    return {
        "documents": pipeline.list_documents(),
        "document_count": len(pipeline.list_documents()),
    }


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    return pipeline.run_query(payload)


@app.post("/detect", response_model=DetectResponse)
def detect(payload: DetectRequest) -> DetectResponse:
    return pipeline.run_detection(payload)
