from __future__ import annotations

from statistics import mean

from backend.core.claim_extractor import extract_claims
from backend.core.knowledge_base import KnowledgeChunk, list_document_sources, load_knowledge_chunks
from backend.core.retriever import HybridRetriever
from backend.core.schemas import ClaimResult, DetectRequest, DetectResponse, EvidenceChunk, QueryRequest, QueryResponse


MIN_GROUNDED_SCORE = 0.42


class HallucinationPipeline:
    def __init__(self) -> None:
        self._chunks = load_knowledge_chunks()
        self._retriever = HybridRetriever(self._chunks)

    def list_documents(self) -> list[dict[str, str]]:
        return list_document_sources(self._chunks)

    def _to_evidence(self, chunk: KnowledgeChunk, score: float) -> EvidenceChunk:
        return EvidenceChunk(
            chunk_id=chunk.chunk_id,
            title=chunk.title,
            source=chunk.source,
            text=chunk.text,
            score=round(score, 4),
        )

    def _split_claims(self, answer: str) -> list[str]:
        return extract_claims(answer)

    def _generate_answer(self, question: str, retrieved_chunks: list[EvidenceChunk], simulate_hallucination: bool) -> str:
        if not retrieved_chunks:
            return "I could not retrieve any relevant context from the knowledge base."

        answer_parts: list[str] = []
        lead = retrieved_chunks[0].text.rstrip(".")
        answer_parts.append(f"The retrieved evidence suggests that {lead}.")

        supporting_chunks = retrieved_chunks[1:3]
        for chunk in supporting_chunks:
            supporting_text = chunk.text.rstrip(".")
            answer_parts.append(f"It also shows that {supporting_text.lower()}.")

        if simulate_hallucination:
            answer_parts.append(
                "The system has already proven a verified 99.9 percent factual accuracy in live production deployments."
            )
        return " ".join(answer_parts)

    def _score_claims(self, claims: list[str], chunks: list[KnowledgeChunk]) -> list[ClaimResult]:
        results: list[ClaimResult] = []
        candidate_chunks = chunks or self._chunks
        for claim in claims:
            best_match = self._retriever.best_supporting_chunk(claim, candidate_chunks)
            verdict = "grounded" if best_match.score >= MIN_GROUNDED_SCORE else "unsupported"
            results.append(
                ClaimResult(
                    claim=claim,
                    verdict=verdict,
                    support_score=round(best_match.score, 4),
                    evidence=self._to_evidence(best_match.chunk, best_match.score),
                )
            )
        return results

    def _correct_answer(self, claim_results: list[ClaimResult]) -> str:
        grounded_claims = [result.claim for result in claim_results if result.verdict == "grounded"]
        if grounded_claims:
            return " ".join(grounded_claims)
        return "I could not produce a grounded answer from the retrieved evidence."

    def _confidence(self, claim_results: list[ClaimResult], verdict: str | None = None) -> float:
        if verdict is None:
            scores = [claim.support_score for claim in claim_results]
        else:
            scores = [claim.support_score for claim in claim_results if claim.verdict == verdict]
        if not scores:
            return 0.0
        return round(mean(scores), 4)

    def _retrieved_evidence(self, question: str, top_k: int) -> tuple[list[KnowledgeChunk], list[EvidenceChunk]]:
        retrieval_results = self._retriever.retrieve(question, top_k=top_k)
        raw_chunks = [item.chunk for item in retrieval_results]
        evidence = [self._to_evidence(item.chunk, item.score) for item in retrieval_results]
        return raw_chunks, evidence

    def run_query(self, payload: QueryRequest) -> QueryResponse:
        raw_chunks, evidence = self._retrieved_evidence(payload.question, payload.top_k)
        answer = self._generate_answer(payload.question, evidence, payload.simulate_hallucination)
        claim_results = self._score_claims(self._split_claims(answer), raw_chunks)
        corrected_answer = self._correct_answer(claim_results)
        answer_confidence = self._confidence(claim_results)
        corrected_confidence = self._confidence(claim_results, verdict="grounded")
        hallucination_score = round(
            len([claim for claim in claim_results if claim.verdict == "unsupported"]) / max(len(claim_results), 1),
            4,
        )

        return QueryResponse(
            question=payload.question,
            answer=answer,
            corrected_answer=corrected_answer,
            retrieval_mode=self._retriever.last_retrieval_mode,
            answer_confidence=answer_confidence,
            corrected_confidence=corrected_confidence,
            confidence_delta=round(corrected_confidence - answer_confidence, 4),
            hallucination_score=hallucination_score,
            claims=claim_results,
            retrieved_chunks=evidence,
        )

    def run_detection(self, payload: DetectRequest) -> DetectResponse:
        raw_chunks, evidence = self._retrieved_evidence(payload.question, payload.top_k)
        claim_results = self._score_claims(self._split_claims(payload.answer), raw_chunks)
        corrected_answer = self._correct_answer(claim_results)
        answer_confidence = self._confidence(claim_results)
        corrected_confidence = self._confidence(claim_results, verdict="grounded")
        hallucination_score = round(
            len([claim for claim in claim_results if claim.verdict == "unsupported"]) / max(len(claim_results), 1),
            4,
        )

        return DetectResponse(
            question=payload.question,
            answer=payload.answer,
            corrected_answer=corrected_answer,
            retrieval_mode=self._retriever.last_retrieval_mode,
            answer_confidence=answer_confidence,
            corrected_confidence=corrected_confidence,
            confidence_delta=round(corrected_confidence - answer_confidence, 4),
            hallucination_score=hallucination_score,
            claims=claim_results,
            retrieved_chunks=evidence,
        )
