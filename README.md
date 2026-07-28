# RAG Hallucination Detector & Auto-Corrector
Free MVP for catching unsupported claims in a retrieval-augmented generation pipeline, then rewriting the answer so it stays grounded in retrieved evidence.

## Delivery Phases

### Phase 1: Backend intelligence
- claim extraction
- lexical retrieval
- embedding-ready retrieval
- FAISS retrieval mode
- claim-level support scoring
- repair loop for unsupported claims
- run history and analytics endpoints

### Phase 2: Demo data

- realistic product-doc knowledge base
- believable questions
- visible hallucination and correction cases

### Phase 3: Frontend experience

- dark product-style interface
- query and inspect modes
- correction trace
- evidence review
- recent runs analytics

### Phase 4: Hardening and deployment

- tests
- docs
- Docker path
- Render config

## MVP Scope

This repo is optimized for a no-cost demo between July 15 and July 20.

What the MVP does:

- retrieves relevant knowledge chunks for a user query
- generates an answer from retrieved context
- splits the answer into sentence-level claims
- scores each claim against evidence
- flags unsupported claims
- auto-corrects unsupported claims by re-querying and rewriting toward stronger evidence
- shows hallucination risk, confidence, evidence, and correction in a custom frontend

What we are intentionally not doing in the first ship:

- paid LLM APIs such as Azure OpenAI
- hosted vector databases
- large-scale ingestion pipelines
- multi-tenant auth or production-grade observability

## Free Stack
- Python
- FastAPI
- HTML/CSS/JavaScript
- sentence-transformers
- FAISS
- LangChain

## Current Demo Domain

The project currently uses a fictional but realistic product-doc knowledge base for `HelixCloud`, a workspace platform for search, document chat, and support workflows.

Notes:

- The code uses lexical retrieval by default so the MVP stays runnable on a clean machine.
- Set `ENABLE_SENTENCE_TRANSFORMERS=1` later when your local Python environment is ready for embedding models.
- Set `RETRIEVAL_MODE=faiss` to use FAISS-backed embedding retrieval, or `RETRIEVAL_MODE=embedding` to compare direct embedding ranking without FAISS.

## Project Structure

- [backend/app.py](backend/app.py): FastAPI app, API routes, and frontend entry route
- [backend/core/claim_extractor.py](backend/core/claim_extractor.py): claim splitting and normalization
- [backend/core/pipeline.py](backend/core/pipeline.py): query, claim scoring, and auto-correction flow
- [backend/core/retriever.py](backend/core/retriever.py): local retrieval with embedding fallback
- [frontend/index.html](frontend/index.html): primary demo UI served by FastAPI
- [data/knowledge_base.json](data/knowledge_base.json): seed knowledge base for local testing
- [data/sample_queries.json](data/sample_queries.json): sample scenarios for future expansion
- [tests/test_pipeline.py](tests/test_pipeline.py): regression tests for the pipeline

## API Contract

### `POST /query`

Input:

```json
{
  "question": "How can a RAG system detect unsupported claims?",
  "top_k": 4,
  "simulate_hallucination": true
}
```

Output:

```json
{
  "run_id": "run-12345",
  "question": "...",
  "answer": "...",
  "corrected_answer": "...",
  "retrieval_mode": "lexical",
  "answer_confidence": 0.52,
  "corrected_confidence": 0.71,
  "confidence_delta": 0.19,
  "hallucination_score": 0.33,
  "grounded_claim_count": 2,
  "unsupported_claim_count": 1,
  "claims": [],
  "correction_steps": [],
  "retrieved_chunks": []
}
```

### `POST /detect`

Use this when you already have an answer and want claim-level grounding plus correction.

### `GET /runs/recent`

Returns recent query and detect runs for the dashboard history panel.

### `GET /analytics/summary`

Returns aggregate run counts plus average hallucination and confidence-delta metrics.

## UI Flow

- `GET /`: opens the main product-style demo UI
- query mode: generate an answer, score claims, and show auto-correction
- inspect mode: paste an answer you already have and audit it directly
- analytics view: inspect recent runs and average risk trends


## Local Run

### 1. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate       
# Windows PowerShell: .venv\Scripts\Activate.ps1
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
This installs only what the API needs to run in its default lexical
retrieval mode. Two optional extras exist and are **not** required for
the default demo path:
- `requirements-ml.txt` -- adds sentence-transformers/faiss/numpy for
  embedding or FAISS retrieval (see `RETRIEVAL_MODE` below).
- `requirements-dashboard.txt` -- adds Streamlit for the secondary
  dashboard at `dashboard/app.py`.

### 3. (Optional) Configure environment variables
Copy `.env.example` to `.env` and adjust `ALLOWED_ORIGINS`,
`RETRIEVAL_MODE`, and `ENABLE_SENTENCE_TRANSFORMERS` as needed, then
export them into your shell (this project does not auto-load `.env`
files). Defaults work fine for local dev without any of this.

### 4. Start the API
```bash
uvicorn backend.app:app --reload --port 8001
```

### 5. Open the app
Open [http://127.0.0.1:8001](http://127.0.0.1:8001)

### 6. Run tests
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### 7. Run with Docker
```bash
docker build -t rag-hallucination-detector .
docker run -p 8000:8000 rag-hallucination-detector
```
