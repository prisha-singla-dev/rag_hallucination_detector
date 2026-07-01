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
- shows answer confidence and correction delta in a Streamlit dashboard

What we are intentionally not doing in the first ship:

- paid LLM APIs such as Azure OpenAI
- hosted vector databases
- large-scale ingestion pipelines
- multi-tenant auth or production-grade observability

## Free Stack

- Python
- FastAPI
- Streamlit
- sentence-transformers
- FAISS
- LangChain

Notes:

- The code uses lexical retrieval by default so the MVP stays runnable on a clean machine.
- Set `ENABLE_SENTENCE_TRANSFORMERS=1` later when your local Python environment is ready for embedding models.

## Project Structure

- [backend/app.py](C:/Users/prish/Downloads/rag-hallucinator-detector/rag_hallucination_detector/backend/app.py): FastAPI app and API routes
- [backend/core/pipeline.py](C:/Users/prish/Downloads/rag-hallucinator-detector/rag_hallucination_detector/backend/core/pipeline.py): query, claim scoring, and auto-correction flow
- [backend/core/retriever.py](C:/Users/prish/Downloads/rag-hallucinator-detector/rag_hallucination_detector/backend/core/retriever.py): local retrieval with embedding fallback
- [dashboard/app.py](C:/Users/prish/Downloads/rag-hallucinator-detector/rag_hallucination_detector/dashboard/app.py): Streamlit demo UI
- [data/knowledge_base.json](C:/Users/prish/Downloads/rag-hallucinator-detector/rag_hallucination_detector/data/knowledge_base.json): seed knowledge base for local testing
- [data/sample_queries.json](C:/Users/prish/Downloads/rag-hallucinator-detector/rag_hallucination_detector/data/sample_queries.json): dashboard examples

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


### Block 1: before 11

- create and activate a virtual environment
- install dependencies
- run the FastAPI health endpoint

### Block 2: 12 to 3

- verify `POST /query` returns a generated answer
- verify unsupported sentence injection is getting flagged
- open the Streamlit dashboard and run the sample queries

### Block 3: after 5

- replace the seed knowledge base with your chosen demo domain
- add 10 to 15 realistic questions for that domain
- note the first bugs or weak spots in the README or issues list

## Local Run

### 1. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Use a fresh virtual environment for this project. The system Python on this machine already has incompatible `numpy/scipy/pandas` binaries, and Streamlit will fail if it picks those up.

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Start the API

```powershell
uvicorn backend.app:app --reload
```

### 4. Start the dashboard

```powershell
streamlit run dashboard/app.py
```

## Near-Term Goal

By end of today, the repo should have:

- a running FastAPI app
- a running Streamlit dashboard
- a local seed knowledge base
- a claim-level detector flow
- a visible corrected answer path
