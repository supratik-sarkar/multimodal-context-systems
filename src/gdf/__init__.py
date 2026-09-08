"""genai-data-foundry: auditable dataset construction for GenAI corpora."""
from __future__ import annotations

from .manifest import (
    DatasetManifest,
    SplitManifest,
    build_manifest,
    diff_manifests,
    digest_records,
)
from .parsers import ingest, parse_file, supported_suffixes
from .pipeline import BuildResult, FoundryConfig, build_dataset
from .records import Record, Rejection, Transformation, sha256_text
from .stages import (
    PII_RULES,
    QualityPolicy,
    contamination_check,
    exact_dedup,
    jaccard,
    mine_hard_negatives,
    near_dedup,
    quality_gate,
    redact_pii,
    scan_pii,
)

__version__ = "0.1.0"

__all__ = [
    "BuildResult", "DatasetManifest", "FoundryConfig", "PII_RULES",
    "QualityPolicy", "Record", "Rejection", "SplitManifest", "Transformation",
    "__version__", "build_dataset", "build_manifest", "contamination_check",
    "diff_manifests", "digest_records", "exact_dedup", "ingest", "jaccard",
    "mine_hard_negatives", "near_dedup", "parse_file", "quality_gate",
    "redact_pii", "scan_pii", "sha256_text", "supported_suffixes",
]
