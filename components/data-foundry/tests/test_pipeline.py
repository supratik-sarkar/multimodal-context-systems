"""End-to-end build: determinism, lineage, manifests, diffing."""
from __future__ import annotations

from pathlib import Path

from gdf import FoundryConfig, build_dataset, diff_manifests, ingest

CORPUS = Path(__file__).resolve().parent.parent / "examples" / "corpus"


def test_ingest_parses_every_supported_format() -> None:
    records = ingest(CORPUS)
    formats = {r.metadata["format"] for r in records}
    assert {"md", "txt", "jsonl"} <= formats
    assert all(r.source_hash for r in records)


def test_markdown_is_split_on_headings() -> None:
    records = [r for r in ingest(CORPUS) if r.source_id == "handbook.md"]
    assert len(records) >= 3
    headings = {r.metadata["heading"] for r in records}
    assert "Style" in headings


def test_build_produces_all_three_splits() -> None:
    result = build_dataset(CORPUS, "v1")
    assert set(result.splits) == {"retrieval", "post_training", "evaluation"}
    assert result.manifest.total_records > 0


def test_build_is_deterministic() -> None:
    a = build_dataset(CORPUS, "v1").manifest
    b = build_dataset(CORPUS, "v1").manifest
    assert a.content_digest == b.content_digest
    assert a.stage_counts == b.stage_counts


def test_duplicate_section_is_removed() -> None:
    result = build_dataset(CORPUS, "v1")
    stages = {r.stage for r in result.rejections}
    assert "exact_dedup" in stages or "near_dedup" in stages


def test_low_quality_records_are_rejected_with_reasons() -> None:
    result = build_dataset(CORPUS, "v1")
    quality = [r for r in result.rejections if r.stage == "quality"]
    assert quality
    assert all(r.reason for r in quality)


def test_pii_is_redacted_everywhere_in_the_output() -> None:
    result = build_dataset(CORPUS, "v1")
    blob = " ".join(r.text for recs in result.splits.values() for r in recs)
    assert "support@example.invalid" not in blob
    assert "maintainer@example.invalid" not in blob
    assert "555-010-9999" not in blob
    assert "10.0.0.14" not in blob


def test_every_output_record_has_full_lineage() -> None:
    result = build_dataset(CORPUS, "v1")
    for rec in result.splits["retrieval"]:
        names = rec.transformation_names()
        assert names[0] == "parse"
        assert "exact_dedup" in names
        assert "pii_redaction" in names
        assert "quality_gate" in names
        assert rec.source_hash and rec.source_id


def test_evaluation_records_do_not_appear_in_training() -> None:
    result = build_dataset(CORPUS, "v1")
    train = {r.record_id for r in result.splits["post_training"]}
    holdout = {r.record_id for r in result.splits["evaluation"]}
    assert not (train & holdout)


def test_split_membership_is_stable_when_the_corpus_grows(tmp_path) -> None:
    import shutil
    grown = tmp_path / "corpus"
    shutil.copytree(CORPUS, grown)
    before = build_dataset(grown, "v1")
    (grown / "extra.txt").write_text(
        "An additional document added later to verify that split membership "
        "of existing records does not move when the corpus grows."
    )
    after = build_dataset(grown, "v2")

    def split_of(result, rid):
        for name in ("post_training", "evaluation"):
            if any(r.record_id == rid for r in result.splits[name]):
                return name
        return None

    common = {r.record_id for r in before.splits["post_training"]} | \
             {r.record_id for r in before.splits["evaluation"]}
    moved = [rid for rid in common
             if split_of(before, rid) and split_of(after, rid)
             and split_of(before, rid) != split_of(after, rid)]
    assert not moved, f"records changed split: {moved}"


def test_manifest_records_rejections_and_counts() -> None:
    m = build_dataset(CORPUS, "v1").manifest.to_dict()
    assert m["schema_version"] == 1
    assert m["stage_counts"]["parsed"] > m["stage_counts"]["after_quality"]
    assert m["rejections"]
    assert m["content_digest"]


def test_manifest_digest_is_order_independent(tmp_path) -> None:
    from gdf.manifest import digest_records
    result = build_dataset(CORPUS, "v1")
    recs = result.splits["retrieval"]
    assert digest_records(recs) == digest_records(list(reversed(recs)))


def test_diff_reports_added_records(tmp_path) -> None:
    import shutil
    grown = tmp_path / "corpus"
    shutil.copytree(CORPUS, grown)
    a = build_dataset(grown, "v1").manifest
    (grown / "new.txt").write_text(
        "A newly added document with enough distinct words to survive the "
        "quality gate and appear in the resulting dataset manifest diff."
    )
    b = build_dataset(grown, "v2").manifest
    d = diff_manifests(a, b)
    assert d["content_digest_changed"] is True
    added = sum(len(s["added"]) for s in d["splits"].values())
    assert added >= 1


def test_config_is_recorded_in_the_manifest() -> None:
    cfg = FoundryConfig(near_dedup_threshold=0.55, eval_fraction=0.3)
    m = build_dataset(CORPUS, "v1", config=cfg).manifest
    assert m.config["near_dedup_threshold"] == 0.55
    assert m.config["eval_fraction"] == 0.3
