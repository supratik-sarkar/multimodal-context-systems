"""Dense retrieval behind a narrow protocol.

The interface is what matters here. ``HashingEmbedder`` is deterministic and
dependency-free so CI can exercise the dense path and the fusion logic on any
machine. It is a real embedder in the mechanical sense -- stable vectors,
cosine similarity, honest scores -- but it carries no semantics: it will not
match "car" to "automobile", and nothing in this repository claims otherwise.

Semantic behaviour requires a real model, which is what the MLX backend in
``multimodal.py`` is for and why its validation is a Mac gate.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .model import Chunk, ScoredChunk


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into a fixed-width vector."""

    @property
    def name(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    @property
    def is_semantic(self) -> bool:
        """True only for embedders with learned semantics.

        Exposed so callers, tests and result files can distinguish a real
        model from a deterministic stand-in without inspecting the class.
        """
        ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(slots=True)
class HashingEmbedder:
    """Deterministic hashed bag-of-words with sublinear term weighting.

    Not semantic. Present so the dense channel, the fusion and the whole
    pipeline are testable offline and reproducibly.
    """

    dims: int = 256
    seed: int = 17

    @property
    def name(self) -> str:
        return f"hashing-{self.dims}d"

    @property
    def dimensions(self) -> int:
        return self.dims

    @property
    def is_semantic(self) -> bool:
        return False

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dims
            counts: dict[str, int] = {}
            for tok in _TOKEN.findall(text.lower()):
                counts[tok] = counts.get(tok, 0) + 1
            for tok, n in counts.items():
                h = hashlib.blake2b(f"{self.seed}:{tok}".encode(),
                                    digest_size=8).digest()
                idx = int.from_bytes(h[:4], "big") % self.dims
                sign = 1.0 if h[4] & 1 else -1.0
                vec[idx] += sign * (1.0 + math.log(n))
            norm = math.sqrt(sum(v * v for v in vec))
            out.append([v / norm for v in vec] if norm else vec)
        return out


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


@dataclass(slots=True)
class DenseIndex:
    """Exact nearest-neighbour search.

    Exact rather than approximate: at fixture scale an ANN index would add a
    dependency and a recall caveat while measuring nothing new. At real scale
    this is the wrong choice, which is stated in the README rather than hidden.
    """

    embedder: Embedder
    min_score: float = 0.15
    chunks: list[Chunk] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self.chunks.extend(chunks)
        self.vectors.extend(self.embedder.embed([c.text for c in chunks]))

    def search(self, query: str, top_k: int = 10) -> list[ScoredChunk]:
        if not self.chunks:
            return []
        q = self.embedder.embed([query])[0]
        scored = [
            ScoredChunk(c, cosine(q, v), "dense",
                        {"embedder": self.embedder.name,
                         "is_semantic": self.embedder.is_semantic})
            for c, v in zip(self.chunks, self.vectors, strict=True)
        ]
        kept = [s for s in scored if s.score >= self.min_score]
        kept.sort(key=lambda s: (-s.score, s.chunk.chunk_id))
        return kept[:top_k]

    def __len__(self) -> int:
        return len(self.chunks)


__all__ = ["DenseIndex", "Embedder", "HashingEmbedder", "cosine"]
