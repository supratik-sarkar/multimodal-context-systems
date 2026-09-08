"""Stage behaviour: dedup, PII, contamination, quality, hard negatives."""
from __future__ import annotations

from gdf import (
    QualityPolicy,
    contamination_check,
    exact_dedup,
    jaccard,
    mine_hard_negatives,
    near_dedup,
    quality_gate,
    redact_pii,
    scan_pii,
)
from gdf.records import Record


def rec(rid: str, text: str) -> Record:
    return Record(record_id=rid, text=text, source_id="s", source_hash="h")


def test_exact_dedup_ignores_whitespace_and_case() -> None:
    kept, rejected = exact_dedup([
        rec("a", "Hello   World"), rec("b", "hello world"), rec("c", "other"),
    ])
    assert [r.record_id for r in kept] == ["a", "c"]
    assert rejected[0].detail["duplicate_of"] == "a"
    assert rejected[0].stage == "exact_dedup"


def test_exact_dedup_records_lineage() -> None:
    kept, _ = exact_dedup([rec("a", "some text here")])
    assert "exact_dedup" in kept[0].transformation_names()


def test_near_dedup_catches_a_lightly_edited_copy() -> None:
    base = "the quick brown fox jumps over the lazy dog in the garden today"
    edited = "the quick brown fox jumps over the lazy dog in the garden now"
    kept, rejected = near_dedup([rec("a", base), rec("b", edited)],
                                threshold=0.5)
    assert [r.record_id for r in kept] == ["a"]
    assert rejected[0].detail["jaccard"] >= 0.5


def test_near_dedup_keeps_genuinely_different_text() -> None:
    kept, rejected = near_dedup([
        rec("a", "distributed training correctness on local processes"),
        rec("b", "retrieval evaluation with citation grounded answers"),
    ], threshold=0.7)
    assert len(kept) == 2 and not rejected


def test_jaccard_bounds() -> None:
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a"}, set()) == 0.0
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_pii_scan_finds_each_rule() -> None:
    found = scan_pii("mail a@b.invalid call 555-010-9999 host 10.0.0.14")
    assert found["email"] == 1
    assert found["phone_grouped"] == 1
    assert found["ipv4"] == 1


def test_pii_redaction_replaces_and_records() -> None:
    kept, _ = redact_pii([rec("a", "write to a@b.invalid now please today")])
    assert "a@b.invalid" not in kept[0].text
    assert "[EMAIL]" in kept[0].text
    lineage = [t for t in kept[0].lineage if t.name == "pii_redaction"][0]
    assert lineage.detail["redacted"] is True


def test_pii_over_limit_is_dropped() -> None:
    text = "a@b.invalid c@d.invalid e@f.invalid g@h.invalid extra words here"
    kept, rejected = redact_pii([rec("a", text)], drop_if_over=2)
    assert not kept
    assert rejected[0].stage == "pii"


def test_contamination_removes_overlap_with_holdout() -> None:
    shared = "evaluation records must never appear inside the training corpus"
    kept, rejected = contamination_check([rec("a", shared), rec("b", "unique text about sharding")],
                                         [shared], threshold=0.5)
    assert [r.record_id for r in kept] == ["b"]
    assert rejected[0].detail["overlap"] >= 0.5


def test_quality_gate_rejects_short_repetitive_and_symbolic() -> None:
    kept, rejected = quality_gate([
        rec("short", "two words"),
        rec("repetitive", " ".join(["buy now"] * 30)),
        rec("symbols", "### $$$ %%% ^^^ &&& *** ((( ))) !!! ??? ~~~ +++"),
        rec("good", "a perfectly reasonable sentence with enough distinct words"),
    ], QualityPolicy())
    assert [r.record_id for r in kept] == ["good"]
    reasons = {r.record_id: r.reason for r in rejected}
    assert "too short" in reasons["short"]
    assert "repetitive" in reasons["repetitive"]


def test_hard_negatives_exclude_exact_matches() -> None:
    records = [
        rec("a", "distributed checkpoint recovery after a process restart"),
        rec("b", "distributed checkpoint sharding across a device mesh"),
        rec("c", "completely unrelated content about gardening"),
    ]
    out = mine_hard_negatives(records, "distributed checkpoint recovery after a process restart")
    ids = [rid for rid, _ in out]
    assert "a" not in ids, "an exact match is not a negative"
