"""Canonical records and lineage.

Every record carries the identity of its source, the hash of the source bytes,
and the ordered list of transformations applied to it. That chain is what makes
a dataset auditable: for any row in a training corpus you can name the file it
came from and every step that touched it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Self


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, slots=True)
class Transformation:
    """One step applied to a record, named and versioned."""

    name: str
    version: str
    detail: dict[str, Any] = field(default_factory=dict)
    at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Record:
    """A canonical unit of text with its provenance."""

    record_id: str
    text: str
    source_id: str
    source_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    lineage: list[Transformation] = field(default_factory=list)

    @property
    def content_hash(self) -> str:
        return sha256_text(self.text)

    def applied(self, name: str, version: str, **detail: Any) -> Self:
        """Return this record with one more transformation recorded."""
        self.lineage.append(Transformation(name=name, version=version,
                                           detail=dict(detail)))
        return self

    def transformation_names(self) -> list[str]:
        return [t.name for t in self.lineage]

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "text": self.text,
            "source_id": self.source_id,
            "source_hash": self.source_hash,
            "content_hash": self.content_hash,
            "metadata": self.metadata,
            "lineage": [t.to_dict() for t in self.lineage],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Record:
        return cls(
            record_id=d["record_id"], text=d["text"],
            source_id=d["source_id"], source_hash=d["source_hash"],
            metadata=dict(d.get("metadata", {})),
            lineage=[Transformation(**{k: v for k, v in t.items()})
                     for t in d.get("lineage", [])],
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass(slots=True)
class Rejection:
    """A record removed from the corpus, with the reason preserved.

    Rejections are kept rather than discarded. "Why is this document not in the
    training set?" is a question a data pipeline should be able to answer.
    """

    record_id: str
    source_id: str
    stage: str
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["Record", "Rejection", "Transformation", "sha256_bytes", "sha256_text"]
