from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from backend.core.logging_config import configure_logging, get_logger, now_ms
from backend.core.pipeline import HallucinationPipeline
from backend.core.schemas import (
    AnalyticsSummaryResponse,
    DetectRequest,
    DetectResponse,
    QueryRequest,
    QueryResponse,
    RunHistoryResponse,
)


configure_logging(os.getenv("LOG_LEVEL"))
logger = get_logger("backend.app")


# The frontend is served by this same FastAPI app (see `index()` below), so
# same-origin browser traffic never needs CORS at all. CORS only matters for
# other origins calling this API directly -- e.g. the Streamlit dashboard
# during local dev, or a separately hosted frontend in the future. Configure
# real origins via the ALLOWED_ORIGINS env var (comma-separated); default to
# common local dev origins only. No cookies/sessions are used by this API,
# so allow_credentials stays False -- flip it on only if that ever changes.
DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:8001",
    "http://127.0.0.1:8001",
    "http://localhost:8501",  # Streamlit dashboard default port
    "http://127.0.0.1:8501",
]


def _allowed_origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS")
    if not raw:
        return DEFAULT_ALLOWED_ORIGINS
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app = FastAPI(
    title="RAG Hallucination Detector",
    description="A free MVP for detecting and auto-correcting unsupported claims in a RAG answer.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Persist run history to a real file so it survives container restarts
# (common on free hosting tiers) instead of the old in-memory deque, which
# wiped every run on redeploy. Override via DATABASE_PATH in production if
# the deployment target needs a different writable location (e.g. a mounted
# volume).
DATABASE_PATH = os.getenv("DATABASE_PATH", "data/app.db")
pipeline = HallucinationPipeline(db_path=DATABASE_PATH)
FRONTEND_PATH = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Assigns a request ID and logs one structured line per request.

    The request ID is echoed back in the X-Request-ID response header and
    in any error body (see the exception handler below), so a user-reported
    error can be correlated with the exact server-side log line -- there
    was previously no logging at all outside the Phase 1 generation-fallback
    warnings, so a deployed instance was effectively undebuggable.
    """
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start = now_ms()
    try:
        response = await call_next(request)
    except Exception:
        # Let the global exception_handler below build the client-facing
        # response; here we only need to make sure this failure is still
        # logged with timing/request_id before it propagates.
        logger.exception(
            "request_failed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "duration_ms": round(now_ms() - start, 2),
            },
        )
        raise
    duration_ms = round(now_ms() - start, 2)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all so an uncaught error never leaks a raw traceback to the
    client. The full exception is already logged with its request_id by the
    middleware above; this just guarantees a clean, consistent error
    contract instead of FastAPI's default plaintext 500.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "Something went wrong processing your request.",
            "request_id": request_id,
        },
    )


# adding endpoints
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(FRONTEND_PATH)


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


@app.get("/runs/recent", response_model=RunHistoryResponse)
def recent_runs(limit: int = 8) -> RunHistoryResponse:
    return pipeline.recent_runs(limit=limit)


@app.get("/analytics/summary", response_model=AnalyticsSummaryResponse)
def analytics_summary() -> AnalyticsSummaryResponse:
    return pipeline.analytics_summary()