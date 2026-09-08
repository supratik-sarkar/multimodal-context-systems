"""Pipeline stages: dedup, PII, contamination, quality.

Each stage takes records and returns (kept, rejected). Stages are pure and
deterministic -- no model calls, no network -- so a dataset build is
reproducible from the inputs alone.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .records import Record, Rejection, sha256_text

StageOutput = tuple[list[Record], list[Rejection]]


# -- exact duplicates ------------------------------------------------------
def exact_dedup(records: Sequence[Record]) -> StageOutput:
    """Remove records whose normalised text is byte-identical to an earlier one."""
    seen: dict[str, str] = {}
    kept, rejected = [], []
    for r in records:
        key = sha256_text(" ".join(r.text.split()).lower())
        if key in seen:
            rejected.append(Rejection(r.record_id, r.source_id, "exact_dedup",
                                      "duplicate of an earlier record",
                                      {"duplicate_of": seen[key]}))
            continue
        seen[key] = r.record_id
        kept.append(r.applied("exact_dedup", "1.0", normalised_hash=key[:16]))
    return kept, rejected


# -- near duplicates -------------------------------------------------------
def _shingles(text: str, k: int = 5) -> set[str]:
    words = " ".join(text.split()).lower().split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def near_dedup(records: Sequence[Record], *, threshold: float = 0.7,
               k: int = 5) -> StageOutput:
    """Remove records whose k-shingle Jaccard similarity exceeds the threshold.

    This is lexical, not semantic. It catches boilerplate and lightly edited
    copies. It does not catch a paraphrase, and this repository does not claim
    that it does -- doing so honestly needs embeddings, which would make the
    stage non-deterministic and is out of scope here.
    """
    kept, rejected = [], []
    signatures: list[tuple[str, set[str]]] = []
    for r in records:
        sig = _shingles(r.text, k)
        match = next(((rid, round(s, 4)) for rid, other in signatures
                      if (s := jaccard(sig, other)) >= threshold), None)
        if match is not None:
            rejected.append(Rejection(r.record_id, r.source_id, "near_dedup",
                                      f"similar to {match[0]}",
                                      {"similar_to": match[0],
                                       "jaccard": match[1],
                                       "threshold": threshold}))
            continue
        signatures.append((r.record_id, sig))
        kept.append(r.applied("near_dedup", "1.0", threshold=threshold,
                              shingle_k=k))
    return kept, rejected


# -- PII -------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PIIRule:
    name: str
    pattern: re.Pattern[str]
    replacement: str


PII_RULES: tuple[PIIRule, ...] = (
    PIIRule("email", re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
            "[EMAIL]"),
    PIIRule("phone_e164", re.compile(r"\+\d{1,3}[\s-]?\d{6,12}\b"), "[PHONE]"),
    PIIRule("phone_grouped",
            re.compile(r"\b\d{3}[\s.-]\d{3}[\s.-]\d{4}\b"), "[PHONE]"),
    PIIRule("ipv4",
            re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[IP]"),
    PIIRule("credit_card_like",
            re.compile(r"\b(?:\d{4}[\s-]){3}\d{4}\b"), "[CARD]"),
)


def scan_pii(text: str) -> dict[str, int]:
    return {rule.name: n for rule in PII_RULES
            if (n := len(rule.pattern.findall(text))) > 0}


def redact_pii(records: Sequence[Record], *, drop_if_over: int | None = None
               ) -> StageOutput:
    """Replace matched PII with typed placeholders.

    Rules are transparent regexes listed in ``PII_RULES``. This is a
    demonstrable, auditable mechanism, not a claim of completeness: regex PII
    detection misses names, addresses and anything context-dependent.
    """
    kept, rejected = [], []
    for r in records:
        found = scan_pii(r.text)
        total = sum(found.values())
        if drop_if_over is not None and total > drop_if_over:
            rejected.append(Rejection(r.record_id, r.source_id, "pii",
                                      f"{total} PII matches exceeds limit",
                                      {"matches": found}))
            continue
        if found:
            text = r.text
            for rule in PII_RULES:
                text = rule.pattern.sub(rule.replacement, text)
            r.text = text
        kept.append(r.applied("pii_redaction", "1.0", matches=found,
                              redacted=bool(found)))
    return kept, rejected


# -- contamination ---------------------------------------------------------
def contamination_check(records: Sequence[Record],
                        held_out: Iterable[str], *,
                        threshold: float = 0.5, k: int = 5) -> StageOutput:
    """Remove training records overlapping a held-out evaluation set.

    Train/eval leakage silently inflates every downstream number, so the
    default is to remove rather than warn.
    """
    eval_sigs = [_shingles(t, k) for t in held_out]
    kept, rejected = [], []
    for r in records:
        sig = _shingles(r.text, k)
        best = max((jaccard(sig, e) for e in eval_sigs), default=0.0)
        if best >= threshold:
            rejected.append(Rejection(r.record_id, r.source_id, "contamination",
                                      "overlaps the held-out set",
                                      {"overlap": round(best, 4),
                                       "threshold": threshold}))
            continue
        kept.append(r.applied("contamination_check", "1.0",
                              max_overlap=round(best, 4), threshold=threshold))
    return kept, rejected


# -- quality gates ---------------------------------------------------------
@dataclass(frozen=True, slots=True)
class QualityPolicy:
    min_words: int = 5
    max_words: int = 5_000
    min_distinct_word_ratio: float = 0.25
    max_non_alpha_ratio: float = 0.6


def quality_gate(records: Sequence[Record],
                 policy: QualityPolicy | None = None) -> StageOutput:
    """Drop records failing explicit, inspectable thresholds."""
    pol = policy or QualityPolicy()
    kept, rejected = [], []
    for r in records:
        words = r.text.split()
        n = len(words)
        distinct_ratio = len(set(w.lower() for w in words)) / n if n else 0.0
        alpha = sum(c.isalpha() or c.isspace() for c in r.text)
        non_alpha_ratio = 1 - (alpha / len(r.text)) if r.text else 1.0

        failures = []
        if n < pol.min_words:
            failures.append(f"too short ({n} < {pol.min_words} words)")
        if n > pol.max_words:
            failures.append(f"too long ({n} > {pol.max_words} words)")
        if n and distinct_ratio < pol.min_distinct_word_ratio:
            failures.append(f"repetitive (distinct ratio {distinct_ratio:.2f})")
        if non_alpha_ratio > pol.max_non_alpha_ratio:
            failures.append(f"mostly non-alphabetic ({non_alpha_ratio:.2f})")

        if failures:
            rejected.append(Rejection(r.record_id, r.source_id, "quality",
                                      "; ".join(failures),
                                      {"words": n,
                                       "distinct_ratio": round(distinct_ratio, 4),
                                       "non_alpha_ratio": round(non_alpha_ratio, 4)}))
            continue
        kept.append(r.applied("quality_gate", "1.0", words=n,
                              distinct_ratio=round(distinct_ratio, 4)))
    return kept, rejected


# -- hard negatives --------------------------------------------------------
def mine_hard_negatives(records: Sequence[Record], query: str, *,
                        top_k: int = 3, k: int = 5) -> list[tuple[str, float]]:
    """Records lexically close to a query but not its answer.

    Useful for retrieval training sets: a negative that shares vocabulary with
    the query is far more informative than a random one.
    """
    q = _shingles(query, k)
    scored = [(r.record_id, round(jaccard(q, _shingles(r.text, k)), 4))
              for r in records]
    ranked = sorted(scored, key=lambda kv: (-kv[1], kv[0]))
    return [(rid, s) for rid, s in ranked if 0.0 < s < 1.0][:top_k]


__all__ = [
    "PIIRule", "PII_RULES", "QualityPolicy", "StageOutput",
    "contamination_check", "exact_dedup", "jaccard", "mine_hard_negatives",
    "near_dedup", "quality_gate", "redact_pii", "scan_pii",
]
