"""Benchmark runner over the labelled fixture set.

Records the embedder's semantic status alongside every number, so a result
file cannot be read as a semantic retrieval measurement when it was produced
by a deterministic stand-in.
"""
from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .corpus import DOCUMENTS, QUERIES, build_graph
from .engine import GraphContextEngine, RetrievalConfig
from .evidence import attribution_rate
from .metrics import summarise


@dataclass(slots=True)
class BenchmarkReport:
    embedder: str
    embedder_is_semantic: bool
    config: dict[str, Any]
    metrics: dict[str, Any]
    multi_hop: dict[str, Any]
    attribution: dict[str, Any]
    per_query: list[dict[str, Any]] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    schema_version: int = 1
    multimodal_status: str = "MAC_VALIDATION_REQUIRED"

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        return p


def run_benchmark(engine: GraphContextEngine | None = None) -> BenchmarkReport:
    if engine is None:
        engine = GraphContextEngine(config=RetrievalConfig(top_k=5))
        engine.graph = build_graph()
        engine.add_documents(DOCUMENTS)

    per_query: list[dict[str, Any]] = []
    pairs: list[tuple[list[str], set[str]]] = []
    hop_ok = 0
    hop_total = 0

    for q in QUERIES:
        result = engine.retrieve(q.query)
        got = result.doc_ids
        pairs.append((got, set(q.relevant_docs)))
        entry: dict[str, Any] = {
            "query": q.query, "retrieved": got,
            "relevant": sorted(q.relevant_docs),
            "hit": bool(set(got) & set(q.relevant_docs)),
            "multi_hop": q.multi_hop,
            "strategy": result.decomposition.strategy,
        }
        if q.multi_hop:
            hop_total += 1
            terminal = q.expected_path[-1]
            reached = terminal in got
            hop_ok += int(reached)
            entry["terminal_document"] = terminal
            entry["terminal_reached"] = reached
        per_query.append(entry)

    # attribution measured on answers quoted verbatim from the corpus
    grounded = 0
    checked = 0
    for doc in DOCUMENTS[:4]:
        sentence = doc.text.split(".")[0].strip() + "."
        res = engine.retrieve(sentence, answer=sentence)
        rate = attribution_rate(sentence, res.bundle.citations)
        verified = res.bundle.verify_all(engine.documents)["all_verified"]
        checked += 1
        grounded += int(rate == 1.0 and verified)

    return BenchmarkReport(
        embedder=engine.embedder.name,
        embedder_is_semantic=engine.embedder.is_semantic,
        config=engine.config.to_dict(),
        metrics=summarise(pairs).to_dict(),
        multi_hop={"queries": hop_total, "terminal_reached": hop_ok,
                   "rate": round(hop_ok / hop_total, 4) if hop_total else 0.0},
        attribution={"answers_checked": checked, "fully_grounded": grounded,
                     "rate": round(grounded / checked, 4) if checked else 0.0},
        per_query=per_query,
        environment={"python": platform.python_version(),
                     "platform": platform.platform(),
                     "machine": platform.machine()},
    )


__all__ = ["BenchmarkReport", "run_benchmark"]
