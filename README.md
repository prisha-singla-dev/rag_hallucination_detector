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

### Phase 2: Demo data -- working

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
- generates an answer from retrieved context using the real Gemini API (falls
  back to a clearly-labeled template if no `GEMINI_API_KEY` is configured)
- splits the answer into sentence-level claims
- scores each claim against evidence using either a lexical (token-overlap)
  scorer or an NLI cross-encoder, verdict is `grounded` / `unsupported` /
  `contradicted`
- flags unsupported and contradicted claims
- auto-corrects flagged claims by re-querying and rewriting toward stronger
  evidence
- shows hallucination risk, confidence, evidence, and correction in a custom
  frontend

What we are intentionally not doing yet:

- persistent storage (run history is in-memory and resets on restart)
- authentication or multi-tenant support
- production-grade observability (structured logging, metrics, tracing)
- hosted vector databases or large-scale ingestion pipelines

## Free Stack

- Python
- FastAPI
- HTML/CSS/JavaScript
- Google Gemini API (free tier) for answer generation, with a
  dependency-free template fallback
- sentence-transformers (NLI cross-encoder) for grounding, with a
  dependency-free lexical fallback
- FAISS (optional embedding retrieval)

## Current Demo Domain

The project currently uses a fictional but realistic product-doc knowledge base for `HelixCloud`, a workspace platform for search, document chat, and support workflows.

Notes:

- The code uses lexical retrieval and lexical grounding by default so the MVP
  stays runnable on a clean machine with zero ML dependencies.
- Set `GEMINI_API_KEY` to get real LLM-generated answers (`generation_mode:
  "llm"` in every response) instead of the template fallback
  (`generation_mode: "template"`). Get a free-tier key at
  [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
- Set `GROUNDING_MODE=nli` (plus `pip install -r requirements-ml.txt`) to use
  an NLI cross-encoder for claim verification instead of token overlap. This
  is what makes the `contradicted` verdict meaningful -- the lexical scorer
  can only ever return `grounded`/`unsupported` (see `eval/results/README.md`
  for measured numbers on why).
- Set `ENABLE_SENTENCE_TRANSFORMERS=1` to enable embedding-based retrieval.
- Set `RETRIEVAL_MODE=faiss` to use FAISS-backed embedding retrieval, or `RETRIEVAL_MODE=embedding` to compare direct embedding ranking without FAISS.

## Project Structure

- [backend/app.py](backend/app.py): FastAPI app, API routes, and frontend entry route
- [backend/core/claim_extractor.py](backend/core/claim_extractor.py): claim splitting and normalization
- [backend/core/generator.py](backend/core/generator.py): Gemini-backed answer generation with template fallback
- [backend/core/grounding.py](backend/core/grounding.py): lexical and NLI claim-grounding scorers
- [backend/core/pipeline.py](backend/core/pipeline.py): query, claim scoring, and auto-correction flow
- [backend/core/retriever.py](backend/core/retriever.py): local retrieval with embedding fallback
- [frontend/index.html](frontend/index.html): primary demo UI served by FastAPI
- [data/knowledge_base.json](data/knowledge_base.json): seed knowledge base for local testing
- [data/sample_queries.json](data/sample_queries.json): sample scenarios for future expansion
- [eval/labeled_claims.json](eval/labeled_claims.json): hand-labeled (claim, evidence, verdict) set
- [eval/run_eval.py](eval/run_eval.py): precision/recall/F1 comparison of lexical vs. NLI grounding
- [tests/](tests/): regression tests for claim extraction, generation, grounding, and the full pipeline

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
  "generation_mode": "llm",
  "generation_model": "gemini-2.5-flash",
  "generation_warning": null,
  "grounding_mode": "nli",
  "answer_confidence": 0.52,
  "corrected_confidence": 0.71,
  "confidence_delta": 0.19,
  "hallucination_score": 0.33,
  "grounded_claim_count": 2,
  "unsupported_claim_count": 1,
  "contradicted_claim_count": 0,
  "claims": [],
  "correction_steps": [],
  "retrieved_chunks": []
}
```

`generation_mode` is `"llm"` when `GEMINI_API_KEY` is set and the call
succeeded, `"template"` when no key is set (or the call failed and the
response degraded gracefully -- check `generation_warning` for why).
`grounding_mode` is `"lexical"` or `"nli"` depending on `GROUNDING_MODE` and
whether `requirements-ml.txt` is installed. Every claim's `verdict` is one of
`grounded` / `unsupported` / `contradicted` -- see `eval/results/README.md`
for why the lexical scorer can never actually return `contradicted`.
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
Copy `.env.example` to `.env` and adjust as needed, then export them into
your shell (this project does not auto-load `.env` files). Defaults work
fine for local dev without any of this -- the app runs in template
generation + lexical grounding mode with zero configuration.

To get real LLM generation and NLI-based grounding instead:
```bash
export GEMINI_API_KEY=your-key-here      # https://aistudio.google.com/apikey
export GROUNDING_MODE=nli                # requires requirements-ml.txt
pip install -r requirements-ml.txt
```

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

### 7. Run the grounding eval
```bash
python -m eval.run_eval               # both scorers, if requirements-ml.txt is installed
python -m eval.run_eval --scorer lexical
```
Prints precision/recall/F1 per verdict class and writes
`eval/results/eval_report.json`. See `eval/results/README.md` for the
current baseline numbers and what they mean.

### 8. Run with Docker
```bash
docker build -t rag-hallucination-detector .
docker run -p 8000:8000 -e GEMINI_API_KEY=your-key-here rag-hallucination-detector
```
