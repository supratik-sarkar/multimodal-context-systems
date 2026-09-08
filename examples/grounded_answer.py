"""Citation grounding, including the case where nothing supports a claim.

    python examples/grounded_answer.py
"""
from __future__ import annotations

from gce import DOCUMENTS, GraphContextEngine, attribution_rate, build_graph


def main() -> int:
    engine = GraphContextEngine()
    engine.graph = build_graph()
    engine.add_documents(DOCUMENTS)

    grounded = "The Durable Store is an append only log."
    invented = "The Durable Store replicates across nineteen continents."

    for label, answer in (("grounded", grounded), ("unsupported", invented)):
        result = engine.retrieve("append only log durable store", answer=answer)
        print(f"{label} answer: {answer}")
        print(f"  attribution rate: "
              f"{attribution_rate(answer, result.bundle.citations):.2f}")
        for c in result.bundle.citations:
            print(f"  cited: {c.doc_id} [{c.start}:{c.end}]")
            print(f"    source says: {c.resolve(engine.documents)!r}")
        if not result.bundle.citations:
            print("  no citation issued: nothing in the retrieved evidence "
                  "supports this sentence")
        report = result.bundle.verify_all(engine.documents)
        print(f"  all citations verify against source: "
              f"{report['all_verified']}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
