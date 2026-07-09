from __future__ import annotations

import re


SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+|;\s+")
LEADING_MARKER_PATTERN = re.compile(r"^(?:[-*]\s+|\d+[.)]\s+)+")
MULTISPACE_PATTERN = re.compile(r"\s+")
INTRO_PREFIXES = (
    "based on the retrieved sources,",
    "based on the retrieved evidence,",
    "the retrieved evidence shows that",
    "the retrieved evidence suggests that",
    "the evidence shows that",
    "the evidence suggests that",
    "it also shows that",
    "it also suggests that",
    "overall,",
)


def normalize_claim(text: str) -> str:
    claim = LEADING_MARKER_PATTERN.sub("", text.strip())
    claim = MULTISPACE_PATTERN.sub(" ", claim)

    lowered = claim.lower()
    for prefix in INTRO_PREFIXES:
        if lowered.startswith(prefix):
            claim = claim[len(prefix) :].strip()
            break

    return claim.strip(" -")


def extract_claims(answer: str) -> list[str]:
    if not answer.strip():
        return []

    raw_parts = [part.strip() for part in SENTENCE_BOUNDARY_PATTERN.split(answer) if part.strip()]
    claims: list[str] = []
    seen: set[str] = set()

    for part in raw_parts:
        normalized = normalize_claim(part)
        if len(normalized.split()) < 4:
            continue

        dedupe_key = normalized.lower()
        if dedupe_key in seen:
            continue

        claims.append(normalized)
        seen.add(dedupe_key)

    return claims
