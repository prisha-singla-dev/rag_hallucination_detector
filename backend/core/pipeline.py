from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from statistics import mean
from uuid import uuid4

from backend.core.claim_extractor import extract_claims, normalize_claim
from backend.core.generator import AnswerGenerator, GeneratedAnswer
from backend.core.grounding import GroundingScorer, build_default_grounding_scorer
from backend.core.knowledge_base import KnowledgeChunk, list_document_sources, load_knowledge_chunks
from backend.core.retriever import HybridRetriever
from backend.core.schemas import (
    AnalyticsSummaryResponse,
    ClaimResult,
    CorrectionStep,
    DetectRequest,
    DetectResponse,
    EvidenceChunk,
    QueryRequest,
    QueryResponse,
    RunHistoryResponse,
    RunRecord,
)


MAX_RUN_HISTORY = 50


class HallucinationPipeline:
    def __init__(
        self,
        generator: AnswerGenerator | None = None,
        grounding_scorer: GroundingScorer | None = None,
    ) -> None:
        self._chunks = load_knowledge_chunks()
        self._retriever = HybridRetriever(self._chunks)
        # Both are overridable for tests (avoids hitting the real Gemini API
        # or downloading the NLI model in unit tests); production code builds
        # them from env vars via the same factories the API layer uses.
        self._generator = generator or AnswerGenerator()
        self._grounding_scorer = grounding_scorer or build_default_grounding_scorer()
        self._run_history: deque[RunRecord] = deque(maxlen=MAX_RUN_HISTORY)

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

    def _generate_answer(
        self, question: str, retrieved_chunks: list[EvidenceChunk], simulate_hallucination: bool
    ) -> GeneratedAnswer:
        return self._generator.generate(question, retrieved_chunks, adversarial=simulate_hallucination)

    def _score_claims(self, claims: list[str], chunks: list[KnowledgeChunk]) -> list[ClaimResult]:
        results: list[ClaimResult] = []
        candidate_chunks = chunks or self._chunks
        for claim in claims:
            grounding = self._grounding_scorer.score(claim, candidate_chunks)
            results.append(
                ClaimResult(
                    claim=claim,
                    verdict=grounding.verdict,
                    support_score=grounding.support_score,
                    evidence=self._to_evidence(grounding.chunk, grounding.support_score),
                    grounding_method=grounding.method,
                )
            )
        return results

    @staticmethod
    def _dedupe_chunks(chunks: list[KnowledgeChunk]) -> list[KnowledgeChunk]:
        seen: set[str] = set()
        deduped: list[KnowledgeChunk] = []
        for chunk in chunks:
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            deduped.append(chunk)
        return deduped

    def _repair_unsupported_claims(
        self,
        question: str,
        claim_results: list[ClaimResult],
        top_k: int,
    ) -> tuple[list[str], list[CorrectionStep], list[KnowledgeChunk]]:
        corrected_claims: list[str] = []
        correction_steps: list[CorrectionStep] = []
        repair_chunks: list[KnowledgeChunk] = []
        seen_claims: set[str] = set()

        def remember(claim: str) -> bool:
            key = normalize_claim(claim).lower()
            if not key or key in seen_claims:
                return False
            seen_claims.add(key)
            corrected_claims.append(claim)
            return True

        for claim_result in claim_results:
            if claim_result.verdict == "grounded":
                remember(claim_result.claim)
                correction_steps.append(
                    CorrectionStep(
                        original_claim=claim_result.claim,
                        action="kept",
                        rewritten_claim=claim_result.claim,
                        reason="Claim was already grounded in the initial retrieved evidence.",
                        evidence=claim_result.evidence,
                    )
                )
                continue

            repair_query = f"{question} {claim_result.claim}"
            follow_up_results = self._retriever.retrieve(repair_query, top_k=max(2, top_k))
            if not follow_up_results:
                correction_steps.append(
                    CorrectionStep(
                        original_claim=claim_result.claim,
                        action="removed",
                        rewritten_claim=None,
                        reason="No stronger supporting evidence was found during the repair pass.",
                        evidence=None,
                    )
                )
                continue

            repair_chunks.extend([result.chunk for result in follow_up_results])
            rescored = self._score_claims([claim_result.claim], [result.chunk for result in follow_up_results])[0]

            if rescored.verdict == "grounded":
                remember(claim_result.claim)
                correction_steps.append(
                    CorrectionStep(
                        original_claim=claim_result.claim,
                        action="kept",
                        rewritten_claim=claim_result.claim,
                        reason="The repair pass found tighter evidence that supports the original claim.",
                        evidence=rescored.evidence,
                    )
                )
                continue

            replacement_candidates = extract_claims(follow_up_results[0].chunk.text) or [follow_up_results[0].chunk.text.strip()]
            replacement_scores = self._score_claims(replacement_candidates, [result.chunk for result in follow_up_results])
            replacement_choice = next(
                (
                    scored
                    for scored in replacement_scores
                    if scored.verdict == "grounded" and normalize_claim(scored.claim).lower() not in seen_claims
                ),
                None,
            )

            if replacement_choice is not None and remember(replacement_choice.claim):
                correction_steps.append(
                    CorrectionStep(
                        original_claim=claim_result.claim,
                        action="replaced",
                        rewritten_claim=replacement_choice.claim,
                        reason="The original claim stayed unsupported, so the answer was rewritten to match the strongest retrieved evidence.",
                        evidence=replacement_choice.evidence,
                    )
                )
            else:
                correction_steps.append(
                    CorrectionStep(
                        original_claim=claim_result.claim,
                        action="removed",
                        rewritten_claim=None,
                        reason="Even after re-querying, the claim could not be grounded strongly enough to keep without repeating existing evidence.",
                        evidence=follow_up_results[0] and self._to_evidence(follow_up_results[0].chunk, follow_up_results[0].score),
                    )
                )

        return corrected_claims, correction_steps, self._dedupe_chunks(repair_chunks)

    def _correct_answer(self, claim_results: list[ClaimResult]) -> str:
        grounded_claims = [result.claim for result in claim_results if result.verdict == "grounded"]
        if grounded_claims:
            return " ".join(grounded_claims)
        return "I could not produce a grounded answer from the retrieved evidence."

    @staticmethod
    def _dedupe_claim_texts(claims: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for claim in claims:
            key = normalize_claim(claim).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(claim.strip())
        return deduped

    @staticmethod
    def _render_claims_as_answer(claims: list[str]) -> str:
        sentences: list[str] = []
        for claim in claims:
            text = claim.strip()
            if not text:
                continue
            if text[0].islower():
                text = text[0].upper() + text[1:]
            if text[-1] not in ".!?":
                text = f"{text}."
            sentences.append(text)
        return " ".join(sentences)

    def _confidence(self, claim_results: list[ClaimResult], verdict: str | None = None) -> float:
        if verdict is None:
            scores = [claim.support_score for claim in claim_results]
        else:
            scores = [claim.support_score for claim in claim_results if claim.verdict == verdict]
        if not scores:
            return 0.0
        return round(mean(scores), 4)

    @staticmethod
    def _claim_counts(claim_results: list[ClaimResult]) -> tuple[int, int, int]:
        """Returns (grounded, unsupported, contradicted) counts.

        Counted explicitly per verdict rather than derived as
        `total - unsupported` -- with the verdict now three-way
        (grounded/unsupported/contradicted), that subtraction would silently
        misclassify contradicted claims as grounded.
        """
        grounded = len([c for c in claim_results if c.verdict == "grounded"])
        unsupported = len([c for c in claim_results if c.verdict == "unsupported"])
        contradicted = len([c for c in claim_results if c.verdict == "contradicted"])
        return grounded, unsupported, contradicted

    def _retrieved_evidence(self, question: str, top_k: int) -> tuple[list[KnowledgeChunk], list[EvidenceChunk]]:
        retrieval_results = self._retriever.retrieve(question, top_k=top_k)
        raw_chunks = [item.chunk for item in retrieval_results]
        evidence = [self._to_evidence(item.chunk, item.score) for item in retrieval_results]
        return raw_chunks, evidence

    def _record_run(
        self,
        run_id: str,
        created_at: datetime,
        mode: str,
        question: str,
        hallucination_score: float,
        answer_confidence: float,
        corrected_confidence: float,
        unsupported_claim_count: int,
        contradicted_claim_count: int,
    ) -> None:
        self._run_history.appendleft(
            RunRecord(
                run_id=run_id,
                created_at=created_at,
                mode=mode,
                question=question,
                hallucination_score=hallucination_score,
                answer_confidence=answer_confidence,
                corrected_confidence=corrected_confidence,
                unsupported_claim_count=unsupported_claim_count,
                contradicted_claim_count=contradicted_claim_count,
                retrieval_mode=self._retriever.last_retrieval_mode,
                grounding_mode=self._grounding_scorer.method_name,
            )
        )

    def recent_runs(self, limit: int = 10) -> RunHistoryResponse:
        runs = list(self._run_history)[:limit]
        return RunHistoryResponse(total_runs=len(self._run_history), runs=runs)

    def analytics_summary(self) -> AnalyticsSummaryResponse:
        runs = list(self._run_history)
        if not runs:
            return AnalyticsSummaryResponse(
                total_runs=0,
                query_runs=0,
                detect_runs=0,
                avg_hallucination_score=0.0,
                avg_confidence_delta=0.0,
                latest_run=None,
            )

        query_runs = len([run for run in runs if run.mode == "query"])
        detect_runs = len(runs) - query_runs
        confidence_deltas = [run.corrected_confidence - run.answer_confidence for run in runs]

        return AnalyticsSummaryResponse(
            total_runs=len(runs),
            query_runs=query_runs,
            detect_runs=detect_runs,
            avg_hallucination_score=round(mean(run.hallucination_score for run in runs), 4),
            avg_confidence_delta=round(mean(confidence_deltas), 4),
            latest_run=runs[0],
        )

    def run_query(self, payload: QueryRequest) -> QueryResponse:
        raw_chunks, evidence = self._retrieved_evidence(payload.question, payload.top_k)
        generated = self._generate_answer(payload.question, evidence, payload.simulate_hallucination)
        answer = generated.text
        claim_results = self._score_claims(self._split_claims(answer), raw_chunks)
        repaired_claims, correction_steps, repair_chunks = self._repair_unsupported_claims(
            payload.question,
            claim_results,
            payload.top_k,
        )
        repaired_claims = self._dedupe_claim_texts(repaired_claims)
        final_chunks = self._dedupe_chunks(raw_chunks + repair_chunks)
        corrected_claim_results = self._score_claims(repaired_claims, final_chunks) if repaired_claims else []
        corrected_answer = self._render_claims_as_answer([claim.claim for claim in corrected_claim_results]) or self._correct_answer(claim_results)
        answer_confidence = self._confidence(claim_results)
        corrected_confidence = self._confidence(corrected_claim_results) if corrected_claim_results else self._confidence(
            claim_results,
            verdict="grounded",
        )
        grounded_claim_count, unsupported_claim_count, contradicted_claim_count = self._claim_counts(claim_results)
        hallucination_score = round(
            (unsupported_claim_count + contradicted_claim_count) / max(len(claim_results), 1), 4
        )
        run_id = f"run-{uuid4().hex[:10]}"
        created_at = datetime.now(timezone.utc)
        response_evidence = [self._to_evidence(chunk, 0.0) for chunk in final_chunks]

        self._record_run(
            run_id=run_id,
            created_at=created_at,
            mode="query",
            question=payload.question,
            hallucination_score=hallucination_score,
            answer_confidence=answer_confidence,
            corrected_confidence=corrected_confidence,
            unsupported_claim_count=unsupported_claim_count,
            contradicted_claim_count=contradicted_claim_count,
        )

        return QueryResponse(
            run_id=run_id,
            created_at=created_at,
            question=payload.question,
            answer=answer,
            corrected_answer=corrected_answer,
            retrieval_mode=self._retriever.last_retrieval_mode,
            generation_mode=generated.mode,
            generation_model=generated.model,
            generation_warning=generated.warning,
            grounding_mode=self._grounding_scorer.method_name,
            answer_confidence=answer_confidence,
            corrected_confidence=corrected_confidence,
            confidence_delta=round(corrected_confidence - answer_confidence, 4),
            hallucination_score=hallucination_score,
            grounded_claim_count=grounded_claim_count,
            unsupported_claim_count=unsupported_claim_count,
            contradicted_claim_count=contradicted_claim_count,
            claims=claim_results,
            correction_steps=correction_steps,
            retrieved_chunks=response_evidence,
        )

    def run_detection(self, payload: DetectRequest) -> DetectResponse:
        raw_chunks, evidence = self._retrieved_evidence(payload.question, payload.top_k)
        claim_results = self._score_claims(self._split_claims(payload.answer), raw_chunks)
        repaired_claims, correction_steps, repair_chunks = self._repair_unsupported_claims(
            payload.question,
            claim_results,
            payload.top_k,
        )
        repaired_claims = self._dedupe_claim_texts(repaired_claims)
        final_chunks = self._dedupe_chunks(raw_chunks + repair_chunks)
        corrected_claim_results = self._score_claims(repaired_claims, final_chunks) if repaired_claims else []
        corrected_answer = self._render_claims_as_answer([claim.claim for claim in corrected_claim_results]) or self._correct_answer(claim_results)
        answer_confidence = self._confidence(claim_results)
        corrected_confidence = self._confidence(corrected_claim_results) if corrected_claim_results else self._confidence(
            claim_results,
            verdict="grounded",
        )
        grounded_claim_count, unsupported_claim_count, contradicted_claim_count = self._claim_counts(claim_results)
        hallucination_score = round(
            (unsupported_claim_count + contradicted_claim_count) / max(len(claim_results), 1), 4
        )
        run_id = f"run-{uuid4().hex[:10]}"
        created_at = datetime.now(timezone.utc)
        response_evidence = [self._to_evidence(chunk, 0.0) for chunk in final_chunks]

        self._record_run(
            run_id=run_id,
            created_at=created_at,
            mode="detect",
            question=payload.question,
            hallucination_score=hallucination_score,
            answer_confidence=answer_confidence,
            corrected_confidence=corrected_confidence,
            unsupported_claim_count=unsupported_claim_count,
            contradicted_claim_count=contradicted_claim_count,
        )

        return DetectResponse(
            run_id=run_id,
            created_at=created_at,
            question=payload.question,
            answer=payload.answer,
            corrected_answer=corrected_answer,
            retrieval_mode=self._retriever.last_retrieval_mode,
            generation_mode="user_provided",
            generation_model=None,
            generation_warning=None,
            grounding_mode=self._grounding_scorer.method_name,
            answer_confidence=answer_confidence,
            corrected_confidence=corrected_confidence,
            confidence_delta=round(corrected_confidence - answer_confidence, 4),
            hallucination_score=hallucination_score,
            grounded_claim_count=grounded_claim_count,
            unsupported_claim_count=unsupported_claim_count,
            contradicted_claim_count=contradicted_claim_count,
            claims=claim_results,
            correction_steps=correction_steps,
            retrieved_chunks=response_evidence,
        )