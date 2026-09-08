"""Document graph and traversal.

Edges are typed and directed. They come from two places: explicit relations
declared by the corpus, and implicit links extracted from the text (a document
mentioning another document's title). Extraction is deliberately conservative
-- a wrong edge is worse than a missing one, because traversal amplifies it.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class EdgeType(StrEnum):
    MENTIONS = "mentions"
    DEPENDS_ON = "depends_on"
    SUPERSEDES = "supersedes"
    PART_OF = "part_of"
    RELATED_TO = "related_to"


@dataclass(frozen=True, slots=True)
class Edge:
    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0
    detail: dict[str, Any] = field(default_factory=dict, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "target": self.target,
                "edge_type": str(self.edge_type), "weight": self.weight,
                "detail": self.detail}


@dataclass(slots=True)
class Path:
    """A traversal result: the node sequence and the edges that joined them."""

    nodes: list[str]
    edges: list[Edge]

    @property
    def hops(self) -> int:
        return len(self.edges)

    @property
    def cost(self) -> float:
        """Lower is better. Edge weight is affinity, so cost is its inverse."""
        return sum(1.0 / max(e.weight, 1e-6) for e in self.edges)

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": self.nodes, "hops": self.hops,
                "cost": round(self.cost, 4),
                "edges": [e.to_dict() for e in self.edges]}


@dataclass(slots=True)
class DocumentGraph:
    nodes: set[str] = field(default_factory=set)
    edges: list[Edge] = field(default_factory=list)
    _out: dict[str, list[Edge]] = field(default_factory=dict)
    _in: dict[str, list[Edge]] = field(default_factory=dict)

    def add_node(self, node: str) -> None:
        self.nodes.add(node)
        self._out.setdefault(node, [])
        self._in.setdefault(node, [])

    def add_edge(self, source: str, target: str, edge_type: EdgeType,
                 weight: float = 1.0, **detail: Any) -> Edge:
        self.add_node(source)
        self.add_node(target)
        edge = Edge(source, target, edge_type, weight, dict(detail))
        if edge in self.edges:
            return edge
        self.edges.append(edge)
        self._out[source].append(edge)
        self._in[target].append(edge)
        return edge

    def neighbours(self, node: str, *, types: set[EdgeType] | None = None,
                   undirected: bool = False) -> list[Edge]:
        out = list(self._out.get(node, []))
        if undirected:
            out += [Edge(e.target, e.source, e.edge_type, e.weight, e.detail)
                    for e in self._in.get(node, [])]
        return [e for e in out if types is None or e.edge_type in types]

    def bfs(self, start: str, *, max_hops: int = 2,
            types: set[EdgeType] | None = None,
            undirected: bool = True) -> dict[str, Path]:
        """Shortest paths by hop count, deterministic in edge order."""
        if start not in self.nodes:
            return {}
        found: dict[str, Path] = {start: Path([start], [])}
        queue: deque[str] = deque([start])
        while queue:
            node = queue.popleft()
            path = found[node]
            if path.hops >= max_hops:
                continue
            for edge in sorted(self.neighbours(node, types=types,
                                               undirected=undirected),
                               key=lambda e: (e.target, str(e.edge_type))):
                if edge.target in found:
                    continue
                found[edge.target] = Path(path.nodes + [edge.target],
                                          path.edges + [edge])
                queue.append(edge.target)
        return found

    def shortest_path(self, start: str, end: str, *,
                      max_hops: int = 4) -> Path | None:
        return self.bfs(start, max_hops=max_hops).get(end)

    def connected(self, start: str, end: str, *, max_hops: int = 4) -> bool:
        return self.shortest_path(start, end, max_hops=max_hops) is not None

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": sorted(self.nodes),
                "edges": [e.to_dict() for e in self.edges]}


def extract_mentions(docs: list[Any], graph: DocumentGraph | None = None,
                     *, min_title_words: int = 2) -> DocumentGraph:
    """Link documents whose text contains another document's title.

    Titles shorter than ``min_title_words`` are ignored: a one-word title
    produces edges from coincidence rather than reference, and a false edge
    propagates through every multi-hop query that touches it.
    """
    g = graph or DocumentGraph()
    for d in docs:
        g.add_node(d.doc_id)
    titles = {d.doc_id: d.title for d in docs
              if d.title and len(d.title.split()) >= min_title_words}
    for d in docs:
        low = d.text.lower()
        for other_id, title in titles.items():
            if other_id != d.doc_id and title.lower() in low:
                g.add_edge(d.doc_id, other_id, EdgeType.MENTIONS, 1.0,
                           via="title_match", title=title)
    return g


__all__ = ["DocumentGraph", "Edge", "EdgeType", "Path", "extract_mentions"]
