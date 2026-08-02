"""Evaluates grounding scorers against a hand-labeled set of (claim, evidence)
pairs, reporting precision/recall/F1 per verdict class plus macro-F1 and
accuracy.

Deliberately scores each scorer against a single (claim, evidence_text) pair
at a time rather than running full retrieval first. This isolates the
grounding scorer's own accuracy from retrieval quality -- if we ran full
`/detect` through the pipeline instead, a retrieval miss and a grounding
miss would look identical in the results, and we'd have no way to tell which
component to improve.

Usage:
    python -m eval.run_eval                # lexical only if ML deps missing
    python -m eval.run_eval --scorer nli    # NLI only
    python -m eval.run_eval --scorer both   # side-by-side (default)

Requires `pip install -r requirements-ml.txt` for the NLI scorer. The first
run downloads cross-encoder/nli-deberta-v3-base (~440MB) from Hugging Face,
so it needs real network access -- this could not be executed in the
sandbox this project was built in (huggingface.co is not in that sandbox's
network allowlist). Run it yourself to get real numbers; see
eval/results/README.md for what to expect and how to sanity-check the output.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from backend.core.grounding import GroundingScorer, LexicalGroundingScorer, NLIGroundingScorer
from backend.core.knowledge_base import KnowledgeChunk

LABELS = ("grounded", "unsupported", "contradicted")
DATA_PATH = Path(__file__).parent / "labeled_claims.json"
RESULTS_DIR = Path(__file__).parent / "results"


@dataclass(frozen=True)
class LabeledExample:
    example_id: str
    label: str
    claim: str
    evidence_title: str
    evidence_text: str

    def as_chunk(self) -> KnowledgeChunk:
        return KnowledgeChunk(
            chunk_id=self.example_id,
            source=f"eval://{self.example_id}",
            title=self.evidence_title,
            text=self.evidence_text,
        )


def load_examples() -> list[LabeledExample]:
    raw = json.loads(DATA_PATH.read_text())
    return [
        LabeledExample(
            example_id=row["id"],
            label=row["label"],
            claim=row["claim"],
            evidence_title=row["evidence_title"],
            evidence_text=row["evidence_text"],
        )
        for row in raw
    ]


def _assert_label_order(scorer: NLIGroundingScorer) -> None:
    """Runs the NLI model against three unambiguous hand-labeled pairs before
    trusting it for the real eval. If cross-encoder/nli-deberta-v3-base's
    label order were ever misread in grounding.py, this fails loudly here
    instead of silently producing inverted precision/recall below.
    """
    entailed = scorer.predict(
        claim="HelixCloud retains audit logs for 90 days.",
        evidence_text="HelixCloud retains audit logs for 90 days.",
    )
    contradicted = scorer.predict(
        claim="HelixCloud retains audit logs for 30 days.",
        evidence_text="HelixCloud retains audit logs for 90 days.",
    )
    neutral = scorer.predict(
        claim="HelixCloud supports PDF uploads.",
        evidence_text="HelixCloud retains audit logs for 90 days.",
    )
    checks = [
        entailed["entailment"] > 0.5,
        contradicted["contradiction"] > 0.5,
        neutral["neutral"] > neutral["entailment"] and neutral["neutral"] > neutral["contradiction"],
    ]
    if not all(checks):
        raise RuntimeError(
            "NLI label-order smoke test failed -- entailment/contradiction/neutral "
            f"labels do not match expectations (entailed={entailed}, "
            f"contradicted={contradicted}, neutral={neutral}). Refusing to run the "
            "full eval with a scorer whose labels may be inverted."
        )


def confusion_and_metrics(predictions: list[tuple[str, str]]) -> dict:
    """predictions: list of (true_label, predicted_label)."""
    confusion: dict[str, dict[str, int]] = {t: {p: 0 for p in LABELS} for t in LABELS}
    for true_label, predicted_label in predictions:
        confusion[true_label][predicted_label] += 1

    per_class = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in LABELS if other != label)
        fn = sum(confusion[label][other] for other in LABELS if other != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[label] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

    accuracy = sum(1 for t, p in predictions if t == p) / len(predictions)
    macro_f1 = sum(m["f1"] for m in per_class.values()) / len(per_class)

    return {
        "confusion_matrix": confusion,
        "per_class": per_class,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
    }


def evaluate_scorer(scorer: GroundingScorer, examples: list[LabeledExample]) -> dict:
    predictions: list[tuple[str, str]] = []
    for example in examples:
        result = scorer.score(example.claim, [example.as_chunk()])
        predictions.append((example.label, result.verdict))
    return confusion_and_metrics(predictions)


def print_report(name: str, report: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"Accuracy: {report['accuracy']:.2%}   Macro-F1: {report['macro_f1']:.4f}")
    print(f"{'label':<14}{'precision':>10}{'recall':>10}{'f1':>10}")
    for label, metrics in report["per_class"].items():
        print(f"{label:<14}{metrics['precision']:>10.2f}{metrics['recall']:>10.2f}{metrics['f1']:>10.2f}")
    print("Confusion matrix (rows=true, cols=predicted):")
    header = "".join(f"{label:>14}" for label in LABELS)
    print(f"{'':<14}{header}")
    for true_label in LABELS:
        row = "".join(f"{report['confusion_matrix'][true_label][pred]:>14}" for pred in LABELS)
        print(f"{true_label:<14}{row}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer", choices=["lexical", "nli", "both"], default="both")
    args = parser.parse_args()

    examples = load_examples()
    print(f"Loaded {len(examples)} labeled examples "
          f"({sum(1 for e in examples if e.label == 'grounded')} grounded / "
          f"{sum(1 for e in examples if e.label == 'unsupported')} unsupported / "
          f"{sum(1 for e in examples if e.label == 'contradicted')} contradicted)")

    reports: dict[str, dict] = {}

    if args.scorer in {"lexical", "both"}:
        lexical_report = evaluate_scorer(LexicalGroundingScorer(), examples)
        print_report("Lexical (token-overlap) scorer", lexical_report)
        reports["lexical"] = lexical_report

    if args.scorer in {"nli", "both"}:
        nli_scorer = NLIGroundingScorer()
        if not nli_scorer.is_available:
            print(
                "\n=== NLI scorer ===\n"
                "sentence-transformers / the model weights are not available. "
                "Run `pip install -r requirements-ml.txt` and re-run this script "
                "(the first run downloads cross-encoder/nli-deberta-v3-base, ~440MB)."
            )
        else:
            _assert_label_order(nli_scorer)
            nli_report = evaluate_scorer(nli_scorer, examples)
            print_report("NLI (cross-encoder) scorer", nli_report)
            reports["nli"] = nli_report

    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / "eval_report.json"
    output_path.write_text(
        json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "example_count": len(examples), "reports": reports},
            indent=2,
        )
    )
    print(f"\nSaved report to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())