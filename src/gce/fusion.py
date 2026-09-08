"""Combining ranked lists from independent channels.

Reciprocal rank fusion is used rather than a weighted score sum because the
channels produce incomparable scales: BM25 is unbounded and corpus-dependent,
cosine similarity sits in [-1, 1], and graph proximity is a hop count.
Normalising those onto a common scale requires assumptions that break whenever
the corpus changes. RRF uses only rank, so it needs none.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .model import Chunk, ScoredChunk


@dataclass(slots=True)
class FusionResult:
    chunk: Chunk
    score: float
    ranks: dict[str, int] = field(default_factory=dict)
    channel_scores: dict[str, float] = field(default_factory=dict)

    @property
    def channels(self) -> list[str]:
        return sorted(self.ranks)

    def to_dict(self) -> dict[str, object]:
        return {"chunk_id": self.chunk.chunk_id, "doc_id": self.chunk.doc_id,
                "score": round(self.score, 6), "ranks": self.ranks,
                "channel_scores": {k: round(v, 6)
                                   for k, v in self.channel_scores.items()},
                "channels": self.channels}


def reciprocal_rank_fusion(
    channels: dict[str, list[ScoredChunk]], *, k: int = 60,
    weights: dict[str, float] | None = None, top_k: int = 10,
) -> list[FusionResult]:
    """RRF: score = sum over channels of weight / (k + rank).

    ``k`` damps the influence of top ranks so a single channel's first result
    cannot dominate agreement between two others.
    """
    w = weights or {}
    acc: dict[str, FusionResult] = {}
    for channel, results in sorted(channels.items()):
        for rank, scored in enumerate(results, start=1):
            cid = scored.chunk.chunk_id
            entry = acc.get(cid)
            if entry is None:
                entry = FusionResult(scored.chunk, 0.0)
                acc[cid] = entry
            entry.score += w.get(channel, 1.0) / (k + rank)
            entry.ranks[channel] = rank
            entry.channel_scores[channel] = scored.score
    out = sorted(acc.values(), key=lambda r: (-r.score, r.chunk.chunk_id))
    return out[:top_k]


@dataclass(slots=True)
class Reranker:
    """Optional second-stage reordering.

    The protocol exists so a cross-encoder can be dropped in. The default
    implementation is a transparent lexical-overlap heuristic, and it says so:
    ``is_model_based`` is False, so a result file can record which kind ran.
    """

    name: str = "lexical-overlap"

    @property
    def is_model_based(self) -> bool:
        return False

    def rerank(self, query: str, results: list[FusionResult],
               *, top_k: int | None = None) -> list[FusionResult]:
        from .lexical import tokenize
        q = set(tokenize(query))
        if not q:
            return results[:top_k] if top_k else results

        def blended(r: FusionResult) -> tuple[float, str]:
            overlap = len(q & set(tokenize(r.chunk.text))) / len(q)
            return (-(0.7 * r.score * 100 + 0.3 * overlap), r.chunk.chunk_id)

        out = sorted(results, key=blended)
        return out[:top_k] if top_k else out


__all__ = ["FusionResult", "Reranker", "reciprocal_rank_fusion"]
