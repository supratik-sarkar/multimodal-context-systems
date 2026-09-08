"""Decompose a dependent question and follow the graph.

    python examples/multi_hop.py
"""
from __future__ import annotations

from gce import DOCUMENTS, QUERIES, GraphContextEngine, build_graph


def main() -> int:
    engine = GraphContextEngine()
    engine.graph = build_graph()
    engine.add_documents(DOCUMENTS)

    ok = 0
    for q in (q for q in QUERIES if q.multi_hop):
        result = engine.retrieve(q.query)
        plan = result.decomposition
        print(f"query: {q.query}")
        print(f"  strategy: {plan.strategy}  depth: {plan.max_depth}")
        for sq in plan.sub_queries:
            dep = f" (depends on #{sq.depends_on})" if sq.depends_on is not None else ""
            print(f"    #{sq.index} {sq.text}{dep}")
        print(f"  retrieved: {result.doc_ids}")
        terminal = q.expected_path[-1]
        reached = terminal in result.doc_ids
        ok += reached
        print(f"  terminal document {terminal!r} reached: {reached}")
        if result.bundle.paths:
            print("  traversal:")
            for p in result.bundle.paths[:3]:
                print(f"    {' -> '.join(p['nodes'])}")
        print()

    print(f"{ok}/2 multi-hop queries reached their terminal document")
    return 0 if ok == 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
