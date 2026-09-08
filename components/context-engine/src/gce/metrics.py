"""Retrieval metrics."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def mean_reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1)
              for i, item in enumerate(retrieved[:k], start=1)
              if item in relevant)
    ideal = sum(1.0 / math.log2(i + 1)
                for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


@dataclass(slots=True)
class MetricSummary:
    queries: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    ndcg_at_5: float

    def to_dict(self) -> dict[str, Any]:
        return {"queries": self.queries,
                "recall@1": round(self.recall_at_1, 4),
                "recall@3": round(self.recall_at_3, 4),
                "recall@5": round(self.recall_at_5, 4),
                "mrr": round(self.mrr, 4),
                "ndcg@5": round(self.ndcg_at_5, 4)}


def summarise(per_query: list[tuple[list[str], set[str]]]) -> MetricSummary:
    n = len(per_query) or 1
    return MetricSummary(
        queries=len(per_query),
        recall_at_1=sum(recall_at_k(r, g, 1) for r, g in per_query) / n,
        recall_at_3=sum(recall_at_k(r, g, 3) for r, g in per_query) / n,
        recall_at_5=sum(recall_at_k(r, g, 5) for r, g in per_query) / n,
        mrr=sum(mean_reciprocal_rank(r, g) for r, g in per_query) / n,
        ndcg_at_5=sum(ndcg_at_k(r, g, 5) for r, g in per_query) / n,
    )


__all__ = ["MetricSummary", "mean_reciprocal_rank", "ndcg_at_k",
           "precision_at_k", "recall_at_k", "summarise"]
