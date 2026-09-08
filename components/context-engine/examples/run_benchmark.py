"""Run the labelled benchmark and print the summary.

    python examples/run_benchmark.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fixtures  # noqa: E402

from gce.bench import run_benchmark  # noqa: E402


def main() -> int:
    engine = fixtures.engine()
    report = run_benchmark(engine, fixtures.QUERIES)

    caps = report["environment"]["embedder"]
    print(f"embedder: {caps['name']}  semantic={caps['semantic']}  "
          f"image={caps['image']}")
    if not caps["semantic"]:
        print("  note: lexical signal only. These numbers show the pipeline "
              "retrieves the right objects,\n        not that the embeddings "
              "carry meaning.")

    print("\nsummary")
    for k, v in report["summary"].items():
        print(f"  {k:<28} {v}")

    print("\nper query")
    for q in report["per_query"]:
        got = ", ".join(q["retrieved"][:3])
        print(f"  [{q['kind']:<9}] recall@5={q['metrics']['recall_at_5']:.2f}  {got}")

    s = report["summary"]
    ok = (s["multi_hop_correct"] == s["multi_hop_total"]
          and s["citation_attribution_rate"] >= 0.8)
    print(f"\nmulti-hop {s['multi_hop_correct']}/{s['multi_hop_total']}  "
          f"attribution {s['citation_attribution_rate']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
