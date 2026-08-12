from __future__ import annotations

import json
from pathlib import Path

import requests
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_QUERIES_PATH = ROOT_DIR / "data" / "sample_queries.json"


def load_sample_queries() -> list[dict[str, str]]:
    with SAMPLE_QUERIES_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


st.set_page_config(page_title="RAG Hallucination Detector", page_icon=":mag:", layout="wide")
st.title("RAG Hallucination Detector")

with st.sidebar:
    api_base_url = st.text_input("API base URL", value="http://127.0.0.1:8000")
    top_k = st.slider("Retrieved chunks", min_value=1, max_value=8, value=4)
    simulate_hallucination = st.toggle("Inject unsupported claim", value=True)

sample_queries = load_sample_queries()
sample_lookup = {item["label"]: item["question"] for item in sample_queries}
selected_label = st.selectbox("Example query", options=list(sample_lookup.keys()))
question = st.text_area("Question", value=sample_lookup[selected_label], height=120)


def render_claim(claim: dict[str, object]) -> None:
    verdict = claim["verdict"]
    color = "#dc2626" if verdict == "unsupported" else "#15803d"
    st.markdown(
        f"""
        <div style="border-left: 4px solid {color}; padding: 0.6rem 1rem; margin-bottom: 0.75rem; background: #11182708;">
            <div style="font-weight: 700; color: {color}; text-transform: capitalize;">{verdict}</div>
            <div style="margin-top: 0.41rem;">{claim["claim"]}</div>
            <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #475569;">
                Support score: {claim["support_score"]} | Evidence: {claim["evidence"]["title"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if st.button("Run pipeline", type="primary", use_container_width=True):
    payload = {
        "question": question,
        "top_k": top_k,
        "simulate_hallucination": simulate_hallucination,
    }
    try:
        response = requests.post(f"{api_base_url}/query", json=payload, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Could not reach the API: {exc}")
    else:
        result = response.json()
        metrics = st.columns(3)
        metrics[0].metric("Hallucination score", result["hallucination_score"])
        metrics[1].metric("Answer confidence", result["answer_confidence"])
        metrics[2].metric("Confidence delta", result["confidence_delta"])

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original answer")
            st.write(result["answer"])
        with col2:
            st.subheader("Corrected answer")
            st.write(result["corrected_answer"])

        st.subheader("Claim grounding")
        for claim in result["claims"]:
            render_claim(claim)

        st.subheader("Retrieved evidence")
        for chunk in result["retrieved_chunks"]:
            with st.expander(f'{chunk["title"]} ({chunk["score"]})'):
                st.write(chunk["text"])
                st.caption(chunk["source"])
