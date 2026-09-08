"""Canonical objects.

A Document is a source. A Chunk is a retrievable unit that knows exactly where
in its document it came from -- character offsets, not approximate positions.
Those offsets are what make citation grounding checkable rather than plausible.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

SCHEMA_VERSION = 1


class Modality(StrEnum):
    TEXT = "text"
    IMAGE = "image"


@dataclass(slots=True)
class Document:
    doc_id: str
    text: str
    title: str = ""
    modality: Modality = Modality.TEXT
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["modality"] = str(self.modality)
        d["content_hash"] = self.content_hash
        return d


@dataclass(slots=True)
class Chunk:
    """A retrievable span of a document.

    ``start`` and ``end`` are character offsets into ``Document.text``. Every
    downstream citation resolves through them, so a quoted sentence can always
    be checked against the source rather than trusted.
    """

    chunk_id: str
    doc_id: str
    text: str
    start: int
    end: int
    modality: Modality = Modality.TEXT
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["modality"] = str(self.modality)
        return d


@dataclass(slots=True)
class Sentence:
    """A sentence inside a chunk, with offsets relative to the document."""

    text: str
    start: int
    end: int


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str, offset: int = 0) -> list[Sentence]:
    """Split into sentences, preserving absolute document offsets."""
    out: list[Sentence] = []
    pos = 0
    for part in _SENTENCE_END.split(text):
        if not part.strip():
            pos += len(part)
            continue
        idx = text.index(part, pos)
        out.append(Sentence(part.strip(), offset + idx, offset + idx + len(part)))
        pos = idx + len(part)
    return out


@dataclass(slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    channel: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"chunk_id": self.chunk.chunk_id, "doc_id": self.chunk.doc_id,
                "score": round(self.score, 6), "channel": self.channel,
                "detail": self.detail}


def chunk_document(doc: Document, *, target_chars: int = 320,
                   overlap: int = 40) -> list[Chunk]:
    """Split on paragraph boundaries, packing up to a target size.

    Chunking on structure rather than a fixed window keeps offsets meaningful
    and avoids cutting a sentence in half, which would make any citation
    derived from that chunk unverifiable.
    """
    if not doc.text.strip():
        return []
    chunks: list[Chunk] = []
    paras: list[tuple[int, str]] = []
    pos = 0
    for para in doc.text.split("\n\n"):
        idx = doc.text.index(para, pos)
        if para.strip():
            paras.append((idx, para))
        pos = idx + len(para)

    buf: list[tuple[int, str]] = []
    size = 0

    def flush() -> None:
        nonlocal buf, size
        if not buf:
            return
        start = buf[0][0]
        end = buf[-1][0] + len(buf[-1][1])
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}::c{len(chunks)}", doc_id=doc.doc_id,
            text=doc.text[start:end], start=start, end=end,
            modality=doc.modality,
            metadata={"title": doc.title, "chunk_index": len(chunks)},
        ))
        if overlap and len(buf) > 1:
            buf, size = [buf[-1]], len(buf[-1][1])
        else:
            buf, size = [], 0

    for idx, para in paras:
        if size and size + len(para) > target_chars:
            flush()
        buf.append((idx, para))
        size += len(para)
    flush()
    return chunks


__all__ = ["Chunk", "Document", "Modality", "SCHEMA_VERSION", "ScoredChunk",
           "Sentence", "chunk_document", "split_sentences"]
