"""Evidence bundles and citation grounding.

An answer is only as good as the ability to check it. A Citation names the
document, the chunk and the exact character span, so verification is a string
comparison against the source rather than an act of trust.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .model import Chunk, Document, split_sentences


@dataclass(frozen=True, slots=True)
class Citation:
    doc_id: str
    chunk_id: str
    start: int
    end: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "chunk_id": self.chunk_id,
                "start": self.start, "end": self.end, "text": self.text}

    def resolve(self, documents: dict[str, Document]) -> str:
        """Return the span as it appears in the source document."""
        doc = documents.get(self.doc_id)
        if doc is None:
            raise KeyError(f"unknown document: {self.doc_id}")
        return doc.text[self.start:self.end]

    def verify(self, documents: dict[str, Document]) -> bool:
        try:
            return self.resolve(documents).strip() == self.text.strip()
        except (KeyError, IndexError):
            return False


@dataclass(slots=True)
class EvidenceBundle:
    """Everything an answer is permitted to rest on."""

    query: str
    chunks: list[Chunk] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    paths: list[dict[str, Any]] = field(default_factory=list)
    channels: dict[str, list[str]] = field(default_factory=dict)

    @property
    def doc_ids(self) -> list[str]:
        return sorted({c.doc_id for c in self.chunks})

    def to_dict(self) -> dict[str, Any]:
        return {"query": self.query, "doc_ids": self.doc_ids,
                "chunk_ids": [c.chunk_id for c in self.chunks],
                "citations": [c.to_dict() for c in self.citations],
                "paths": self.paths, "channels": self.channels}

    def verify_all(self, documents: dict[str, Document]) -> dict[str, Any]:
        bad = [c.to_dict() for c in self.citations if not c.verify(documents)]
        return {"total": len(self.citations), "unverifiable": bad,
                "all_verified": not bad}


def ground_citations(answer: str, chunks: list[Chunk],
                     *, min_overlap: float = 0.5) -> list[Citation]:
    """Attach each answer sentence to the source sentence that supports it.

    Matching is lexical overlap between the answer sentence and candidate
    source sentences. A sentence with no sufficiently overlapping source gets
    no citation -- silence is the correct output when nothing supports a claim,
    and inventing an approximate match is exactly the failure this guards.
    """
    from .lexical import tokenize

    candidates: list[tuple[Chunk, Any]] = [
        (chunk, sent)
        for chunk in chunks
        for sent in split_sentences(chunk.text, offset=chunk.start)
    ]
    out: list[Citation] = []
    for a_sent in split_sentences(answer):
        a_toks = set(tokenize(a_sent.text))
        if not a_toks:
            continue
        best: tuple[float, Chunk, Any] | None = None
        for chunk, sent in candidates:
            s_toks = set(tokenize(sent.text))
            if not s_toks:
                continue
            overlap = len(a_toks & s_toks) / len(a_toks)
            if best is None or overlap > best[0]:
                best = (overlap, chunk, sent)
        if best and best[0] >= min_overlap:
            _, chunk, sent = best
            out.append(Citation(chunk.doc_id, chunk.chunk_id, sent.start,
                                sent.end, sent.text))
    return out


def attribution_rate(answer: str, citations: list[Citation]) -> float:
    """Fraction of answer sentences carrying a citation."""
    sentences = split_sentences(answer)
    if not sentences:
        return 1.0
    return len(citations) / len(sentences)


def unsupported_sentences(answer: str, citations: list[Citation]) -> list[str]:
    """Answer sentences with no citation, in order."""
    cited = len(citations)
    return [s.text for s in split_sentences(answer)[cited:]]


__all__ = ["Citation", "EvidenceBundle", "attribution_rate", "ground_citations",
           "unsupported_sentences"]
