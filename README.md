# RAG Hallucination Detector & Auto-Corrector

Free MVP for catching unsupported claims in a retrieval-augmented generation pipeline, then rewriting the answer so it stays grounded in retrieved evidence.

## MVP Scope

This repo is optimized for a no-cost demo between July 15 and July 20.

What the MVP does:

- retrieves relevant knowledge chunks for a user query
- generates an answer from retrieved context
- splits the answer into sentence-level claims
- scores each claim against evidence
- flags unsupported claims
- auto-corrects the answer by removing unsupported content
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

Notes:

- The code uses lexical retrieval by default so the MVP stays runnable on a clean machine.
- Set `ENABLE_SENTENCE_TRANSFORMERS=1` later when your local Python environment is ready for embedding models.

## Project Structure

- [backend/app.py](C:/Users/prish/Downloads/rag-hallucinator-detector/rag_hallucination_detector/backend/app.py): FastAPI app, API routes, and frontend entry route
- [backend/core/pipeline.py](C:/Users/prish/Downloads/rag-hallucinator-detector/rag_hallucination_detector/backend/core/pipeline.py): query, claim scoring, and auto-correction flow
- [backend/core/retriever.py](C:/Users/prish/Downloads/rag-hallucinator-detector/rag_hallucination_detector/backend/core/retriever.py): local retrieval with embedding fallback
- [frontend/index.html](C:/Users/prish/Downloads/rag-hallucinator-detector/rag_hallucination_detector/frontend/index.html): primary demo UI served by FastAPI
- [data/knowledge_base.json](C:/Users/prish/Downloads/rag-hallucinator-detector/rag_hallucination_detector/data/knowledge_base.json): seed knowledge base for local testing
- [data/sample_queries.json](C:/Users/prish/Downloads/rag-hallucinator-detector/rag_hallucination_detector/data/sample_queries.json): sample scenarios for future expansion

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
  "question": "...",
  "answer": "...",
  "corrected_answer": "...",
  "answer_confidence": 0.52,
  "corrected_confidence": 0.71,
  "confidence_delta": 0.19,
  "hallucination_score": 0.33,
  "claims": [],
  "retrieved_chunks": []
}
```

### `POST /detect`

Use this when you already have an answer and want claim-level grounding plus correction.

## UI Flow

- `GET /`: opens the main product-style demo UI
- query mode: generate an answer, score claims, and show auto-correction
- inspect mode: paste an answer you already have and audit it directly

## Today's Work

### Before 11

- create and activate a fresh virtual environment
- install dependencies from `requirements.txt`
- run the API and open the custom frontend

### 12 to 3

- run `query` mode against the seeded knowledge base
- run `inspect` mode with a manually written risky answer
- note where the claim scoring feels weak or too optimistic

### After 5

- replace the seed knowledge base with your chosen demo domain
- add 10 to 15 realistic questions for that domain
- pick 3 demo scenarios you want to rehearse by July 14

## Local Run

### 1. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Use a fresh virtual environment for this project. The system Python on this machine already has incompatible scientific Python binaries, so the clean venv matters.

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Start the API

```powershell
uvicorn backend.app:app --reload --port 8001
```

### 4. Open the app

Open [http://127.0.0.1:8001](http://127.0.0.1:8001)

## Near-Term Goal

By end of today, the repo should have:

- a running FastAPI app
- a usable custom frontend
- a local seed knowledge base
- a claim-level detector flow
- a visible corrected answer path
