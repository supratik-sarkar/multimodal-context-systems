"""BM25 lexical retrieval.

Implemented directly rather than pulled from a library, because the parameters
matter to the fusion story downstream and a reader should be able to see what
the scores mean.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .model import Chunk, ScoredChunk

_TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset([
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "to",
    "was", "were", "will", "with", "this", "these", "those", "which", "who",
    "whom", "what", "when", "where", "how",
])


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    return [t for t in toks if t not in STOPWORDS] if drop_stopwords else toks


@dataclass(slots=True)
class BM25Index:
    """Okapi BM25 with the standard k1/b parameterisation."""

    k1: float = 1.5
    b: float = 0.75
    chunks: list[Chunk] = field(default_factory=list)
    _tf: list[Counter[str]] = field(default_factory=list)
    _df: Counter[str] = field(default_factory=Counter)
    _lengths: list[int] = field(default_factory=list)
    _avg_len: float = 0.0

    def add(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            toks = tokenize(c.text)
            tf = Counter(toks)
            self.chunks.append(c)
            self._tf.append(tf)
            self._lengths.append(len(toks))
            for term in tf:
                self._df[term] += 1
        self._avg_len = (sum(self._lengths) / len(self._lengths)
                         if self._lengths else 0.0)

    def _idf(self, term: str) -> float:
        n = len(self.chunks)
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 10) -> list[ScoredChunk]:
        terms = tokenize(query)
        if not terms or not self.chunks:
            return []
        scored: list[ScoredChunk] = []
        for i, chunk in enumerate(self.chunks):
            tf, length = self._tf[i], self._lengths[i]
            score = 0.0
            matched: list[str] = []
            for term in terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                matched.append(term)
                denom = f + self.k1 * (
                    1 - self.b + self.b * length / (self._avg_len or 1)
                )
                score += self._idf(term) * (f * (self.k1 + 1)) / denom
            if score > 0:
                scored.append(ScoredChunk(chunk, score, "bm25",
                                          {"matched_terms": sorted(set(matched))}))
        scored.sort(key=lambda s: (-s.score, s.chunk.chunk_id))
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self.chunks)


__all__ = ["BM25Index", "STOPWORDS", "tokenize"]
