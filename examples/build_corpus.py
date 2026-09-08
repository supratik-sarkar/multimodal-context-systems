"""Build the sample corpus and print what the pipeline did.

    python examples/build_corpus.py
"""
from __future__ import annotations

from pathlib import Path

from gdf import build_dataset

CORPUS = Path(__file__).resolve().parent / "corpus"


def main() -> int:
    result = build_dataset(CORPUS, "v1")
    m = result.manifest

    print("stage counts")
    for stage, n in m.stage_counts.items():
        print(f"  {stage:<24} {n}")

    print("\nrejected, with reasons")
    for r in result.rejections:
        print(f"  {r.record_id:<20} {r.stage:<14} {r.reason}")

    print("\nsplits")
    for name, s in sorted(m.splits.items()):
        print(f"  {name:<14} {s.record_count} records  digest={s.content_digest[:12]}")

    print("\nlineage of one surviving record")
    rec = result.splits["retrieval"][0]
    print(f"  {rec.record_id}  from {rec.source_id} ({rec.source_hash[:12]})")
    for t in rec.lineage:
        print(f"    {t.name:<20} v{t.version}  {t.detail}")

    print(f"\ndataset content digest: {m.content_digest[:32]}")
    again = build_dataset(CORPUS, "v1").manifest
    print(f"rebuild produces the same digest: {again.content_digest == m.content_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
