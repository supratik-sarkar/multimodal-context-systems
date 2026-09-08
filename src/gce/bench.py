"""Benchmark runner over the labelled fixture set."""
from __future__ import annotations

import platform
import sys
import time
from typing import Any

from .engine import ContextEngine
from .metrics import evaluate_ranking, mean_metrics


def run_benchmark(engine: ContextEngine, queries=None) -> dict[str, Any]:
    if queries is None:
        sys.path.insert(0, "examples")
        import fixtures
        queries = fixtures.QUERIES

    per_query, metric_sets = [], []
    latencies: list[float] = []
    attributed = 0
    multi_hop_correct = 0
    multi_hop_total = 0

    for text, relevant, kind in queries:
        t0 = time.perf_counter()
        bundle = engine.retrieve(text)
        latencies.append((time.perf_counter() - t0) * 1000)

        ms = evaluate_ranking(bundle.object_ids, relevant)
        metric_sets.append(ms)

        cited = {c.object_id for c in bundle.citations}
        if cited & relevant:
            attributed += 1

        if kind == "multi_hop":
            multi_hop_total += 1
            if len(set(bundle.object_ids) & relevant) >= 2:
                multi_hop_correct += 1

        per_query.append({
            "query": text, "kind": kind, "relevant": sorted(relevant),
            "retrieved": bundle.object_ids, "metrics": ms.to_dict(),
            "citations": [c.object_id for c in bundle.citations],
            "graph_paths": [p.to_dict() for p in bundle.paths],
        })

    return {
        "schema_version": 1,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "embedder": engine.embedder.capabilities().to_dict(),
        },
        "summary": {
            **mean_metrics(metric_sets),
            "queries": len(per_query),
            "citation_attribution_rate": round(attributed / len(per_query), 4),
            "multi_hop_correct": multi_hop_correct,
            "multi_hop_total": multi_hop_total,
            "mean_latency_ms": round(sum(latencies) / len(latencies), 3),
            "p95_latency_ms": round(sorted(latencies)[int(0.95 * len(latencies)) - 1], 3),
        },
        "per_query": per_query,
    }


__all__ = ["run_benchmark"]
