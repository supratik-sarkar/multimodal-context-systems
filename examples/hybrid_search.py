"""Show each channel's contribution to a fused result.

    python examples/hybrid_search.py
"""
from __future__ import annotations

from gce import DOCUMENTS, GraphContextEngine, RetrievalConfig, build_graph


def main() -> int:
    engine = GraphContextEngine(config=RetrievalConfig(top_k=5))
    engine.graph = build_graph()
    engine.add_documents(DOCUMENTS)

    query = "what happens when retries are exhausted"
    print(f"query: {query}\n")

    for label, cfg in (
        ("lexical only", RetrievalConfig(use_dense=False, use_graph=False)),
        ("dense only", RetrievalConfig(use_lexical=False, use_graph=False)),
        ("all three, fused", RetrievalConfig()),
    ):
        engine.config = cfg
        result = engine.retrieve(query)
        print(f"{label}:")
        for i, r in enumerate(result.results, 1):
            print(f"  {i}. {r.chunk.chunk_id:<28} "
                  f"score={r.score:.5f} channels={','.join(r.channels)}")
        print()

    engine.config = RetrievalConfig()
    result = engine.retrieve(query)
    print("agreement across channels is what RRF rewards:")
    for r in result.results[:3]:
        print(f"  {r.chunk.chunk_id:<28} ranks={r.ranks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
