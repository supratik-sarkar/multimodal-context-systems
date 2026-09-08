"""The engine: ingestion, hybrid retrieval, multi-hop execution."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .embedding import DenseIndex, Embedder, HashingEmbedder
from .evidence import EvidenceBundle, ground_citations
from .fusion import FusionResult, Reranker, reciprocal_rank_fusion
from .graph import DocumentGraph, EdgeType, extract_mentions
from .lexical import BM25Index
from .model import Chunk, Document, ScoredChunk, chunk_document
from .multihop import Decomposition, decompose


@dataclass(slots=True)
class RetrievalConfig:
    top_k: int = 5
    per_channel_k: int = 10
    rrf_k: int = 60
    weights: dict[str, float] = field(default_factory=dict)
    graph_hops: int = 2
    graph_decay: float = 0.5
    use_graph: bool = True
    use_dense: bool = True
    use_lexical: bool = True
    rerank: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"top_k": self.top_k, "per_channel_k": self.per_channel_k,
                "rrf_k": self.rrf_k, "weights": self.weights,
                "graph_hops": self.graph_hops, "graph_decay": self.graph_decay,
                "channels": [n for n, on in
                             (("lexical", self.use_lexical),
                              ("dense", self.use_dense),
                              ("graph", self.use_graph)) if on],
                "rerank": self.rerank}


@dataclass(slots=True)
class RetrievalResult:
    query: str
    results: list[FusionResult]
    decomposition: Decomposition
    bundle: EvidenceBundle
    channel_counts: dict[str, int] = field(default_factory=dict)

    @property
    def chunk_ids(self) -> list[str]:
        return [r.chunk.chunk_id for r in self.results]

    @property
    def doc_ids(self) -> list[str]:
        seen: list[str] = []
        for r in self.results:
            if r.chunk.doc_id not in seen:
                seen.append(r.chunk.doc_id)
        return seen

    def to_dict(self) -> dict[str, Any]:
        return {"query": self.query,
                "decomposition": self.decomposition.to_dict(),
                "channel_counts": self.channel_counts,
                "results": [r.to_dict() for r in self.results],
                "evidence": self.bundle.to_dict()}


@dataclass(slots=True)
class GraphContextEngine:
    embedder: Embedder = field(default_factory=HashingEmbedder)
    config: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: Reranker | None = None
    documents: dict[str, Document] = field(default_factory=dict)
    chunks: dict[str, Chunk] = field(default_factory=dict)
    graph: DocumentGraph = field(default_factory=DocumentGraph)
    bm25: BM25Index = field(default_factory=BM25Index)
    dense: DenseIndex | None = None

    def __post_init__(self) -> None:
        if self.dense is None:
            self.dense = DenseIndex(self.embedder)

    # -- ingestion ---------------------------------------------------------
    def add_documents(self, docs: list[Document], *,
                      target_chars: int = 320) -> int:
        new: list[Chunk] = []
        for doc in docs:
            self.documents[doc.doc_id] = doc
            self.graph.add_node(doc.doc_id)
            for chunk in chunk_document(doc, target_chars=target_chars):
                self.chunks[chunk.chunk_id] = chunk
                new.append(chunk)
        self.bm25.add(new)
        assert self.dense is not None
        self.dense.add(new)
        extract_mentions(list(self.documents.values()), self.graph)
        return len(new)

    def add_edge(self, source: str, target: str, edge_type: EdgeType,
                 weight: float = 1.0, **detail: Any) -> None:
        self.graph.add_edge(source, target, edge_type, weight, **detail)

    # -- channels ----------------------------------------------------------
    def _graph_channel(self, seeds: list[str],
                       k: int) -> tuple[list[ScoredChunk], list[dict[str, Any]]]:
        """Expand from seed documents, scoring by hop distance."""
        scored: dict[str, ScoredChunk] = {}
        paths: list[dict[str, Any]] = []
        for seed in seeds:
            for doc_id, path in self.graph.bfs(
                seed, max_hops=self.config.graph_hops
            ).items():
                if path.hops == 0:
                    continue
                paths.append({"from": seed, **path.to_dict()})
                weight = self.config.graph_decay ** path.hops
                for chunk in (c for c in self.chunks.values()
                              if c.doc_id == doc_id):
                    prev = scored.get(chunk.chunk_id)
                    if prev is None or weight > prev.score:
                        scored[chunk.chunk_id] = ScoredChunk(
                            chunk, weight, "graph",
                            {"hops": path.hops, "from": seed,
                             "path": path.nodes},
                        )
        out = sorted(scored.values(),
                     key=lambda s: (-s.score, s.chunk.chunk_id))
        return out[:k], paths

    def _retrieve_once(self, query: str, *, seeds: list[str] | None = None
                       ) -> tuple[list[FusionResult], dict[str, list[str]],
                                  list[dict[str, Any]], dict[str, int]]:
        cfg = self.config
        channels: dict[str, list[ScoredChunk]] = {}
        if cfg.use_lexical:
            channels["lexical"] = self.bm25.search(query, cfg.per_channel_k)
        if cfg.use_dense:
            assert self.dense is not None
            channels["dense"] = self.dense.search(query, cfg.per_channel_k)

        paths: list[dict[str, Any]] = []
        if cfg.use_graph:
            if seeds is None:
                seeds = []
                for res in list(channels.values()):
                    for s in res[:3]:
                        if s.chunk.doc_id not in seeds:
                            seeds.append(s.chunk.doc_id)
            graph_hits, paths = self._graph_channel(seeds, cfg.per_channel_k)
            if graph_hits:
                channels["graph"] = graph_hits

        fused = reciprocal_rank_fusion(channels, k=cfg.rrf_k,
                                       weights=cfg.weights, top_k=cfg.top_k)
        if cfg.rerank and self.reranker is not None:
            fused = self.reranker.rerank(query, fused, top_k=cfg.top_k)

        provenance = {name: [s.chunk.chunk_id for s in res]
                      for name, res in channels.items()}
        counts = {name: len(res) for name, res in channels.items()}
        return fused, provenance, paths, counts

    # -- public API --------------------------------------------------------
    def retrieve(self, query: str, *, answer: str | None = None
                 ) -> RetrievalResult:
        """Retrieve, following multi-hop dependencies where present."""
        plan = decompose(query)
        seeds: list[str] | None = None
        results: list[FusionResult] = []
        provenance: dict[str, list[str]] = {}
        paths: list[dict[str, Any]] = []
        counts: dict[str, int] = {}

        per_hop: list[list[FusionResult]] = []
        for sub in plan.sub_queries:
            hop_seeds = seeds if sub.depends_on is not None else None
            r, prov, p, c = self._retrieve_once(sub.text, seeds=hop_seeds)
            seeds = [fr.chunk.doc_id for fr in r[:3]]
            for name, ids in prov.items():
                provenance.setdefault(name, []).extend(ids)
            paths.extend(p)
            for name, n in c.items():
                counts[name] = counts.get(name, 0) + n
            per_hop.append(r)

        # For a dependent chain the final hop carries the answer and the
        # earlier hops are supporting context: "which store backs the
        # component that X depends on" is asking for the store, not for X.
        # Merging every hop into one ranking lets the well-matched
        # intermediate documents outrank the terminal one, which is the
        # opposite of what was asked. So later hops are placed first, and
        # earlier hops follow as context.
        if plan.max_depth > 0:
            ordered = [r for hop in reversed(per_hop)
                       for r in sorted(hop, key=lambda x: (-x.score,
                                                           x.chunk.chunk_id))]
        else:
            ordered = sorted((r for hop in per_hop for r in hop),
                             key=lambda x: (-x.score, x.chunk.chunk_id))

        seen_ids: set[str] = set()
        for fr in ordered:
            if fr.chunk.chunk_id not in seen_ids:
                seen_ids.add(fr.chunk.chunk_id)
                results.append(fr)
        results = results[:self.config.top_k]

        chunks = [r.chunk for r in results]
        citations = ground_citations(answer, chunks) if answer else []
        bundle = EvidenceBundle(query=query, chunks=chunks,
                                citations=citations, paths=paths,
                                channels=provenance)
        return RetrievalResult(query, results, plan, bundle, counts)

    def capability_report(self) -> dict[str, Any]:
        return {
            "documents": len(self.documents), "chunks": len(self.chunks),
            "graph_nodes": len(self.graph.nodes),
            "graph_edges": len(self.graph.edges),
            "embedder": self.embedder.name,
            "embedder_is_semantic": self.embedder.is_semantic,
            "config": self.config.to_dict(),
            "semantic_retrieval_validated": self.embedder.is_semantic,
        }


__all__ = ["GraphContextEngine", "RetrievalConfig", "RetrievalResult"]
