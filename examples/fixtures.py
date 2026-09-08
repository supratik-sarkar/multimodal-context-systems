"""Benchmark fixture set: corpus, graph relations, and labelled queries.

Relevance labels are hand-written against the corpus above. They are small
enough to audit by reading, which is the point: a benchmark whose ground truth
cannot be checked is not evidence of anything.
"""
from __future__ import annotations

from pathlib import Path

from gce import Document, EvidenceGraph, build_graph, load_corpus

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"


def documents() -> list[Document]:
    return load_corpus(CORPUS_DIR)


RELATIONS = [
    {"source": "journal#0", "target": "storage#0", "relation": "persists_to"},
    {"source": "journal#1", "target": "journal#0", "relation": "elaborates"},
    {"source": "tracing#1", "target": "journal#0", "relation": "observes"},
    {"source": "storage#1", "target": "storage#0", "relation": "elaborates"},
    {"source": "policy#1", "target": "policy#0", "relation": "elaborates"},
]

# (query, relevant object ids, kind)
QUERIES: list[tuple[str, set[str], str]] = [
    ("checkpoints appended rather than updated in place",
     {"storage#0"}, "lexical"),
    ("operation journal completed result reused",
     {"journal#1"}, "lexical"),
    ("delegation cannot exceed granted scope",
     {"policy#1"}, "lexical"),
    ("span replayed flag operation identifier",
     {"tracing#1"}, "lexical"),
    ("how does the journal relate to durable storage and what does tracing record",
     {"journal#0", "storage#0", "tracing#1"}, "multi_hop"),
    ("policy ordering and delegation minimisation",
     {"policy#0", "policy#1"}, "multi_hop"),
    ("retention of checkpoints over a long run", {"storage#1"}, "lexical"),
]


def graph(objects) -> EvidenceGraph:
    return build_graph(objects, RELATIONS)


def engine(**kw):
    from gce import ContextEngine
    docs = documents()
    objects = [o for d in docs for o in d.chunk(max_chars=200, overlap=0)]
    return ContextEngine(objects, graph=graph(objects), **kw)


__all__ = ["CORPUS_DIR", "QUERIES", "RELATIONS", "documents", "engine", "graph"]
