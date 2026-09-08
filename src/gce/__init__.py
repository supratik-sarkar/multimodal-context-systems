"""graph-context-engine: hybrid lexical, dense and graph retrieval."""
from __future__ import annotations

from .corpus import DOCUMENTS, QUERIES, LabelledQuery, build_graph
from .embedding import DenseIndex, Embedder, HashingEmbedder, cosine
from .engine import GraphContextEngine, RetrievalConfig, RetrievalResult
from .evidence import (
    Citation,
    EvidenceBundle,
    attribution_rate,
    ground_citations,
    unsupported_sentences,
)
from .fusion import FusionResult, Reranker, reciprocal_rank_fusion
from .graph import DocumentGraph, Edge, EdgeType, Path, extract_mentions
from .lexical import BM25Index, tokenize
from .metrics import (
    MetricSummary,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    summarise,
)
from .model import (
    SCHEMA_VERSION,
    Chunk,
    Document,
    Modality,
    ScoredChunk,
    chunk_document,
    split_sentences,
)
from .multihop import Decomposition, SubQuery, decompose
from .multimodal import (
    BackendUnavailable,
    ImageEmbedder,
    MLXImageEmbedder,
    MultimodalIndex,
    StubImageEmbedder,
)

__version__ = "0.1.0"

__all__ = [
    "BM25Index", "BackendUnavailable", "Chunk", "Citation", "DOCUMENTS",
    "Decomposition", "DenseIndex", "Document", "DocumentGraph", "Edge",
    "EdgeType", "Embedder", "EvidenceBundle", "FusionResult",
    "GraphContextEngine", "HashingEmbedder", "ImageEmbedder", "LabelledQuery",
    "MLXImageEmbedder", "MetricSummary", "Modality", "MultimodalIndex",
    "Path", "QUERIES", "Reranker", "RetrievalConfig", "RetrievalResult",
    "SCHEMA_VERSION", "ScoredChunk", "StubImageEmbedder", "SubQuery",
    "__version__", "attribution_rate", "build_graph", "chunk_document",
    "cosine", "decompose", "extract_mentions", "ground_citations",
    "mean_reciprocal_rank", "ndcg_at_k", "precision_at_k",
    "reciprocal_rank_fusion", "recall_at_k", "split_sentences", "summarise",
    "tokenize", "unsupported_sentences",
]
