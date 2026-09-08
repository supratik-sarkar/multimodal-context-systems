"""Hybrid retrieval: decomposition, dense/lexical/graph, fusion, evidence."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from .embed import Embedder, UnsupportedModality, cosine
from .graph import Path
from .lexical import Hit, tokenize
from .objects import CanonicalObject

# -- decomposition ---------------------------------------------------------
_CONNECTIVES = re.compile(
    r"\s+(?:and then|and also|then|;|,\s*and\b|\band\b)\s+", re.IGNORECASE
)


@dataclass(slots=True)
class SubQuery:
    text: str
    index: int

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "index": self.index}


def decompose(query: str, *, max_parts: int = 4) -> list[SubQuery]:
    """Split a compound question into retrievable parts.

    Rule based and deterministic. A model-backed decomposer would slot in
    behind the same signature; keeping it lexical here means the multi-hop
    tests measure the graph, not a model.
    """
    parts = [p.strip(" ?.") for p in _CONNECTIVES.split(query) if p.strip(" ?.")]
    if len(parts) <= 1:
        return [SubQuery(query.strip(), 0)]
    return [SubQuery(p, i) for i, p in enumerate(parts[:max_parts])]


# -- dense index -----------------------------------------------------------
class DenseIndex:
    """Cosine search with an explicit similarity floor.

    Without a floor, a dense channel returns *something* for every query --
    cosine is rarely exactly zero -- and those near-zero matches then enter
    fusion as if they were evidence. ``min_score`` is the honest place to say
    "this channel found nothing".
    """

    def __init__(self, objects: list[CanonicalObject], embedder: Embedder, *,
                 min_score: float = 0.15) -> None:
        self.embedder = embedder
        self.min_score = min_score
        self.objects: list[CanonicalObject] = []
        self._vectors: dict[str, list[float]] = {}
        self.skipped: list[tuple[str, str]] = []
        for o in objects:
            try:
                self._vectors[o.object_id] = embedder.embed_object(o)
                self.objects.append(o)
            except UnsupportedModality as exc:
                self.skipped.append((o.object_id, str(exc)))

    def __len__(self) -> int:
        return len(self._vectors)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        qv = self.embedder.embed_text([query])[0]
        scored = [Hit(oid, cosine(qv, v), "dense")
                  for oid, v in self._vectors.items()]
        scored = [h for h in scored if h.score >= self.min_score]
        scored.sort(key=lambda h: (-h.score, h.object_id))
        return scored[:k]


# -- reranker --------------------------------------------------------------
class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit],
               objects: dict[str, CanonicalObject]) -> list[Hit]: ...


class LexicalOverlapReranker:
    """Deterministic reranker scoring query-term coverage of the object text.

    Not a cross-encoder. It exists so the interface is exercised end to end
    without pulling in a model; a real reranker implements the same protocol.
    """

    name = "lexical-overlap"

    def rerank(self, query: str, hits: list[Hit],
               objects: dict[str, CanonicalObject]) -> list[Hit]:
        q = set(tokenize(query))
        out = []
        for h in hits:
            obj = objects.get(h.object_id)
            toks = set(tokenize(obj.text)) if obj else set()
            coverage = len(q & toks) / len(q) if q else 0.0
            out.append(Hit(h.object_id, h.score * (1 + coverage), "reranked"))
        out.sort(key=lambda h: (-h.score, h.object_id))
        return out


# -- fusion ----------------------------------------------------------------
def reciprocal_rank_fusion(rankings: list[list[Hit]], *, k: int = 60,
                           top_k: int = 10) -> list[Hit]:
    """RRF: score = sum over rankings of 1/(k + rank).

    Chosen over score normalisation because BM25 and cosine scores are not on
    comparable scales, and normalising them invents a relationship that is not
    there.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.object_id] = scores.get(hit.object_id, 0.0) + 1 / (k + rank)
    fused = [Hit(oid, s, "fused") for oid, s in scores.items()]
    fused.sort(key=lambda h: (-h.score, h.object_id))
    return fused[:top_k]


# -- evidence --------------------------------------------------------------
@dataclass(slots=True)
class Citation:
    object_id: str
    document_id: str
    span: dict[str, int] | None
    quote: str

    def to_dict(self) -> dict[str, Any]:
        return {"object_id": self.object_id, "document_id": self.document_id,
                "span": self.span, "quote": self.quote}


@dataclass(slots=True)
class EvidenceBundle:
    """What retrieval returns. Generation is optional and reads from this."""

    query: str
    sub_queries: list[SubQuery]
    hits: list[Hit]
    citations: list[Citation]
    paths: list[Path] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def object_ids(self) -> list[str]:
        return [h.object_id for h in self.hits]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "sub_queries": [s.to_dict() for s in self.sub_queries],
            "hits": [h.to_dict() for h in self.hits],
            "citations": [c.to_dict() for c in self.citations],
            "paths": [p.to_dict() for p in self.paths],
            "diagnostics": self.diagnostics,
        }


def ground_citation(obj: CanonicalObject, query: str, *,
                    window: int = 160) -> Citation:
    """Narrow a chunk to the sentence carrying the most query terms.

    A citation pointing at a 400-character chunk is weak evidence. Narrowing to
    the sentence, with offsets kept relative to the source document, is what
    makes the claim checkable.
    """
    q = set(tokenize(query))
    sentences: list[tuple[int, str]] = []
    cursor = 0
    for sent in re.split(r"(?<=[.!?])\s+", obj.text):
        if not sent.strip():
            continue
        start = obj.text.index(sent, cursor)
        sentences.append((start, sent))
        cursor = start + len(sent)

    best_local, best_text, best_score = 0, obj.text[:window], -1.0
    for start, sent in sentences:
        score = len(q & set(tokenize(sent)))
        if score > best_score:
            best_local, best_text, best_score = start, sent, score

    base = obj.span.start if obj.span else 0
    span = {"start": base + best_local,
            "end": base + best_local + len(best_text)}
    return Citation(object_id=obj.object_id, document_id=obj.document_id,
                    span=span, quote=best_text.strip()[:window])


__all__ = ["Citation", "DenseIndex", "EvidenceBundle", "LexicalOverlapReranker",
           "Reranker", "SubQuery", "decompose", "ground_citation",
           "reciprocal_rank_fusion"]
