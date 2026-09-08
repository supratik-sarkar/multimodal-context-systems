"""Canonical object model.

Everything retrievable is a ``CanonicalObject``: a span of text, an image, or a
table, addressed by a stable id and carrying the offsets needed to cite it
precisely. Retrieval returns objects; generation is optional and layered on top.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class Modality(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"


@dataclass(frozen=True, slots=True)
class Span:
    """A character range inside a source document."""

    start: int
    end: int

    def slice(self, text: str) -> str:
        return text[self.start:self.end]

    def to_dict(self) -> dict[str, int]:
        return {"start": self.start, "end": self.end}


@dataclass(slots=True)
class CanonicalObject:
    object_id: str
    document_id: str
    modality: Modality
    text: str = ""
    span: Span | None = None
    image_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        basis = self.text if self.modality is Modality.TEXT else (
            self.image_path or self.text
        )
        return hashlib.sha256(basis.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["modality"] = str(self.modality)
        d["span"] = self.span.to_dict() if self.span else None
        d["content_hash"] = self.content_hash
        return d


@dataclass(slots=True)
class Document:
    document_id: str
    text: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def chunk(self, *, max_chars: int = 400, overlap: int = 40
              ) -> list[CanonicalObject]:
        """Split on paragraph boundaries, keeping exact source offsets.

        Offsets are preserved rather than recomputed, so a citation always
        resolves back into the original document byte-for-byte.
        """
        objects: list[CanonicalObject] = []
        paragraphs: list[tuple[int, str]] = []
        cursor = 0
        for para in self.text.split("\n\n"):
            start = self.text.index(para, cursor)
            paragraphs.append((start, para))
            cursor = start + len(para)

        buf: list[tuple[int, str]] = []
        size = 0
        idx = 0

        def flush() -> None:
            nonlocal buf, size, idx
            if not buf:
                return
            start = buf[0][0]
            end = buf[-1][0] + len(buf[-1][1])
            objects.append(CanonicalObject(
                object_id=f"{self.document_id}#{idx}",
                document_id=self.document_id, modality=Modality.TEXT,
                text=self.text[start:end], span=Span(start, end),
                metadata={"title": self.title, "chunk": idx},
            ))
            idx += 1
            keep = buf[-1] if overlap and len(buf) > 1 else None
            buf = [keep] if keep else []
            size = len(keep[1]) if keep else 0

        for start, para in paragraphs:
            if not para.strip():
                continue
            if size + len(para) > max_chars and buf:
                flush()
            buf.append((start, para))
            size += len(para)
        flush()
        return objects


def load_corpus(root: str | Path) -> list[Document]:
    docs = []
    for p in sorted(Path(root).glob("*.txt")):
        text = p.read_text(encoding="utf-8")
        first = text.strip().split("\n", 1)[0]
        docs.append(Document(document_id=p.stem, text=text, title=first[:80]))
    return docs


def objects_to_json(objects: list[CanonicalObject]) -> str:
    return json.dumps([o.to_dict() for o in objects], indent=2, sort_keys=True)


__all__ = ["CanonicalObject", "Document", "Modality", "Span", "load_corpus",
           "objects_to_json"]
