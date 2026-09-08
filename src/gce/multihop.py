"""Multi-hop query decomposition.

A question like "which service does the component that handles retries depend
on?" cannot be answered by one retrieval: the second clause depends on the
answer to the first. Decomposition splits it into ordered sub-queries where a
later one may consume an earlier one's result.

The decomposer is rule-based and deterministic. That is a deliberate limit, not
an oversight -- an LLM decomposer would make every test in this repository
non-reproducible, and the interface below is where one would be substituted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_CONNECTIVES = (
    r"\s+and then\s+", r"\s+then\s+", r"\s*;\s*", r"\s+and also\s+",
)
_RELATIVE = re.compile(
    r"^(?P<head>.*?)\s+(?:that|which|who)\s+(?P<tail>.+)$", re.IGNORECASE
)


@dataclass(slots=True)
class SubQuery:
    text: str
    index: int
    depends_on: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "index": self.index,
                "depends_on": self.depends_on}


@dataclass(slots=True)
class Decomposition:
    original: str
    sub_queries: list[SubQuery] = field(default_factory=list)
    strategy: str = "single"

    @property
    def is_multi_hop(self) -> bool:
        return len(self.sub_queries) > 1

    @property
    def max_depth(self) -> int:
        depth, seen = 0, {}
        for sq in self.sub_queries:
            d = 0 if sq.depends_on is None else seen.get(sq.depends_on, 0) + 1
            seen[sq.index] = d
            depth = max(depth, d)
        return depth

    def to_dict(self) -> dict[str, Any]:
        return {"original": self.original, "strategy": self.strategy,
                "is_multi_hop": self.is_multi_hop, "max_depth": self.max_depth,
                "sub_queries": [s.to_dict() for s in self.sub_queries]}


def decompose(query: str) -> Decomposition:
    """Split a query into ordered sub-queries.

    Two patterns are recognised. Coordination ("A and then B") produces
    independent sub-queries. A relative clause ("the X that Y") produces a
    dependent pair: resolve the clause first, then the head.
    """
    text = query.strip()
    for pattern in _CONNECTIVES:
        parts = [p.strip() for p in re.split(pattern, text, flags=re.IGNORECASE)
                 if p.strip()]
        if len(parts) > 1:
            return Decomposition(
                query, [SubQuery(p, i) for i, p in enumerate(parts)],
                "coordination",
            )

    m = _RELATIVE.match(text)
    if m:
        head, tail = m.group("head").strip(), m.group("tail").strip()
        if len(head.split()) >= 2 and len(tail.split()) >= 2:
            return Decomposition(
                query,
                [SubQuery(tail, 0), SubQuery(head, 1, depends_on=0)],
                "relative_clause",
            )

    return Decomposition(query, [SubQuery(text, 0)], "single")


__all__ = ["Decomposition", "SubQuery", "decompose"]
