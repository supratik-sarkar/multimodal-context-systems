"""Command line interface."""
from __future__ import annotations

import argparse
import json
import sys

from .benchmark import run_benchmark
from .corpus import DOCUMENTS, build_graph
from .engine import GraphContextEngine, RetrievalConfig
from .multimodal import MLXImageEmbedder, MultimodalIndex


def _engine(top_k: int = 5) -> GraphContextEngine:
    e = GraphContextEngine(config=RetrievalConfig(top_k=top_k))
    e.graph = build_graph()
    e.add_documents(DOCUMENTS)
    return e


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gce", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("search", help="hybrid retrieval over the fixture corpus")
    p.add_argument("query")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--explain", action="store_true",
                   help="show per-channel ranks and traversal paths")
    p.add_argument("--answer", default=None,
                   help="ground this answer against the retrieved evidence")

    p = sub.add_parser("graph", help="print the document graph")
    p.add_argument("--format", choices=["json", "mermaid"], default="json")

    p = sub.add_parser("benchmark", help="run the labelled benchmark")
    p.add_argument("--out", default=None)

    sub.add_parser("capabilities", help="what is and is not validated here")

    p = sub.add_parser("multimodal", help="multimodal retrieval (Apple Silicon)")
    p.add_argument("--images", required=True)
    p.add_argument("--query", required=True)

    args = ap.parse_args(argv)

    match args.cmd:
        case "search":
            engine = _engine(args.top_k)
            result = engine.retrieve(args.query, answer=args.answer)
            plan = result.decomposition
            if plan.is_multi_hop:
                print(f"decomposed ({plan.strategy}):")
                for sq in plan.sub_queries:
                    dep = f"  <- depends on #{sq.depends_on}" if sq.depends_on is not None else ""
                    print(f"  #{sq.index} {sq.text}{dep}")
                print()
            for i, r in enumerate(result.results, 1):
                print(f"{i}. {r.chunk.chunk_id}  score={r.score:.5f}  "
                      f"channels={','.join(r.channels)}")
                print(f"   {r.chunk.text[:100].replace(chr(10), ' ')}...")
                if args.explain:
                    print(f"   ranks={r.ranks}")
            if args.explain and result.bundle.paths:
                print("\ntraversal paths:")
                for path in result.bundle.paths[:6]:
                    print(f"  {' -> '.join(path['nodes'])} ({path['hops']} hops)")
            if args.answer:
                print("\ncitations:")
                for c in result.bundle.citations:
                    print(f"  {c.doc_id} [{c.start}:{c.end}] {c.text[:70]}")
                report = result.bundle.verify_all(engine.documents)
                print(f"  all verified: {report['all_verified']}")
                if not result.bundle.citations:
                    print("  (no sentence in the answer is supported by the "
                          "retrieved evidence)")
        case "graph":
            g = build_graph()
            if args.format == "json":
                print(json.dumps(g.to_dict(), indent=2))
            else:
                print("graph TD")
                for e in g.edges:
                    print(f"    {e.source.replace('-', '_')} "
                          f"-->|{e.edge_type}| {e.target.replace('-', '_')}")
        case "benchmark":
            report = run_benchmark()
            print(f"embedder: {report.embedder}  "
                  f"semantic: {report.embedder_is_semantic}")
            print(json.dumps(report.metrics, indent=2))
            print(f"multi-hop terminal reached: "
                  f"{report.multi_hop['terminal_reached']}/"
                  f"{report.multi_hop['queries']}")
            print(f"attribution: {report.attribution['fully_grounded']}/"
                  f"{report.attribution['answers_checked']}")
            print(f"multimodal: {report.multimodal_status}")
            if args.out:
                print(f"wrote {report.write(args.out)}", file=sys.stderr)
        case "capabilities":
            engine = _engine()
            report = engine.capability_report()
            report["multimodal"] = MultimodalIndex(
                MLXImageEmbedder()).capability_report()
            print(json.dumps(report, indent=2))
        case "multimodal":
            from pathlib import Path
            emb = MLXImageEmbedder()
            if not emb.available():
                print("MLX unavailable on this machine. Multimodal retrieval "
                      "requires Apple Silicon; see "
                      "scripts/validate_multimodal_mac.sh", file=sys.stderr)
                return 2
            idx = MultimodalIndex(emb)
            idx.add_images(sorted(Path(args.images).glob("*")))
            for hit in idx.search(args.query):
                print(f"{hit.score:.4f}  {hit.chunk.chunk_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
