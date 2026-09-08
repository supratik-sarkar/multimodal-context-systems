"""Versioned dataset manifests.

A manifest is the addressable identity of a dataset: which records it contains,
what produced them, and a content digest over the whole set. Two builds from
the same inputs with the same config produce the same ``content_digest``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .records import Record, Rejection

SCHEMA_VERSION = 1


@dataclass(slots=True)
class SplitManifest:
    name: str
    record_ids: list[str]
    content_digest: str
    record_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DatasetManifest:
    dataset_version: str
    created_at: str
    config: dict[str, Any]
    splits: dict[str, SplitManifest]
    stage_counts: dict[str, int]
    rejections: list[dict[str, Any]] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    @property
    def content_digest(self) -> str:
        joined = "|".join(f"{n}:{s.content_digest}"
                          for n, s in sorted(self.splits.items()))
        return hashlib.sha256(joined.encode()).hexdigest()

    @property
    def total_records(self) -> int:
        return sum(s.record_count for s in self.splits.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_version": self.dataset_version,
            "created_at": self.created_at,
            "content_digest": self.content_digest,
            "total_records": self.total_records,
            "config": self.config,
            "stage_counts": self.stage_counts,
            "splits": {n: s.to_dict() for n, s in sorted(self.splits.items())},
            "rejections": self.rejections,
        }

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return p


def digest_records(records: list[Record]) -> str:
    """Order-independent digest over record content hashes."""
    joined = "|".join(sorted(r.content_hash for r in records))
    return hashlib.sha256(joined.encode()).hexdigest()


def build_manifest(version: str, splits: dict[str, list[Record]],
                   config: dict[str, Any], stage_counts: dict[str, int],
                   rejections: list[Rejection]) -> DatasetManifest:
    return DatasetManifest(
        dataset_version=version,
        created_at=datetime.now(UTC).isoformat(),
        config=config,
        splits={
            name: SplitManifest(
                name=name,
                record_ids=sorted(r.record_id for r in recs),
                content_digest=digest_records(recs),
                record_count=len(recs),
            )
            for name, recs in splits.items()
        },
        stage_counts=stage_counts,
        rejections=[r.to_dict() for r in rejections],
    )


def diff_manifests(a: DatasetManifest, b: DatasetManifest) -> dict[str, Any]:
    """What changed between two dataset versions."""
    out: dict[str, Any] = {
        "from": a.dataset_version, "to": b.dataset_version,
        "content_digest_changed": a.content_digest != b.content_digest,
        "splits": {},
    }
    for name in sorted(set(a.splits) | set(b.splits)):
        old = set(a.splits[name].record_ids) if name in a.splits else set()
        new = set(b.splits[name].record_ids) if name in b.splits else set()
        out["splits"][name] = {
            "added": sorted(new - old),
            "removed": sorted(old - new),
            "unchanged": len(old & new),
        }
    return out


__all__ = ["DatasetManifest", "SCHEMA_VERSION", "SplitManifest",
           "build_manifest", "diff_manifests", "digest_records"]
