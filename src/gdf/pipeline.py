"""The build: sources in, versioned splits out."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .manifest import DatasetManifest, build_manifest
from .parsers import ingest
from .records import Record, Rejection
from .stages import (
    QualityPolicy,
    contamination_check,
    exact_dedup,
    near_dedup,
    quality_gate,
    redact_pii,
)


@dataclass(slots=True)
class FoundryConfig:
    near_dedup_threshold: float = 0.7
    shingle_k: int = 5
    contamination_threshold: float = 0.5
    pii_drop_if_over: int | None = None
    quality: QualityPolicy = field(default_factory=QualityPolicy)
    eval_fraction: float = 0.2
    seed: int = 1234

    def to_dict(self) -> dict[str, Any]:
        return {
            "near_dedup_threshold": self.near_dedup_threshold,
            "shingle_k": self.shingle_k,
            "contamination_threshold": self.contamination_threshold,
            "pii_drop_if_over": self.pii_drop_if_over,
            "quality": {
                "min_words": self.quality.min_words,
                "max_words": self.quality.max_words,
                "min_distinct_word_ratio": self.quality.min_distinct_word_ratio,
                "max_non_alpha_ratio": self.quality.max_non_alpha_ratio,
            },
            "eval_fraction": self.eval_fraction,
            "seed": self.seed,
        }


@dataclass(slots=True)
class BuildResult:
    manifest: DatasetManifest
    splits: dict[str, list[Record]]
    rejections: list[Rejection]

    def split(self, name: str) -> list[Record]:
        return self.splits[name]


def _deterministic_split(records: Sequence[Record], fraction: float,
                         seed: int) -> tuple[list[Record], list[Record]]:
    """Split by a hash of the record id, so membership is stable across runs.

    A random shuffle would move records between splits whenever the corpus
    changes, which quietly invalidates every historical comparison.
    """
    import hashlib
    holdout: list[Record] = []
    train: list[Record] = []
    for r in records:
        h = hashlib.sha256(f"{seed}:{r.record_id}".encode()).hexdigest()
        bucket = int(h[:8], 16) / 0xFFFFFFFF
        (holdout if bucket < fraction else train).append(r)
    return train, holdout


def build_dataset(source_dir: str | Path, version: str, *,
                  config: FoundryConfig | None = None) -> BuildResult:
    cfg = config or FoundryConfig()
    rejections: list[Rejection] = []
    counts: dict[str, int] = {}

    records = ingest(Path(source_dir))
    counts["parsed"] = len(records)

    records, rej = exact_dedup(records)
    rejections += rej
    counts["after_exact_dedup"] = len(records)

    records, rej = near_dedup(records, threshold=cfg.near_dedup_threshold,
                              k=cfg.shingle_k)
    rejections += rej
    counts["after_near_dedup"] = len(records)

    records, rej = redact_pii(records, drop_if_over=cfg.pii_drop_if_over)
    rejections += rej
    counts["after_pii"] = len(records)

    records, rej = quality_gate(records, cfg.quality)
    rejections += rej
    counts["after_quality"] = len(records)

    train, holdout = _deterministic_split(records, cfg.eval_fraction, cfg.seed)

    train, rej = contamination_check(
        train, [r.text for r in holdout],
        threshold=cfg.contamination_threshold, k=cfg.shingle_k,
    )
    rejections += rej
    counts["after_contamination"] = len(train)

    splits = {
        "retrieval": list(train),
        "post_training": list(train),
        "evaluation": list(holdout),
    }
    manifest = build_manifest(version, splits, cfg.to_dict(), counts, rejections)
    return BuildResult(manifest=manifest, splits=splits, rejections=rejections)


__all__ = ["BuildResult", "FoundryConfig", "build_dataset"]
