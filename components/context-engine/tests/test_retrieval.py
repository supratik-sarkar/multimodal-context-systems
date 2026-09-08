"""Model, lexical, dense, graph, fusion and metric behaviour."""
from __future__ import annotations

import pytest

from gce import (
    BM25Index,
    DenseIndex,
    Document,
    DocumentGraph,
    EdgeType,
    HashingEmbedder,
    chunk_document,
    decompose,
    extract_mentions,
    recall_at_k,
    reciprocal_rank_fusion,
    split_sentences,
    tokenize,
)
from gce.model import ScoredChunk


def doc(did: str, text: str, title: str = "") -> Document:
    return Document(did, text, title)


# -- chunking --------------------------------------------------------------
def test_chunk_offsets_index_the_source_exactly() -> None:
    d = doc("d", "First para here.\n\nSecond para here.\n\nThird para here.")
    for c in chunk_document(d, target_chars=20):
        assert d.text[c.start:c.end] == c.text


def test_chunking_an_empty_document_yields_nothing() -> None:
    assert chunk_document(doc("d", "   \n\n  ")) == []


def test_sentence_offsets_are_absolute() -> None:
    text = "Alpha runs first. Beta runs second."
    for s in split_sentences(text, offset=100):
        assert text[s.start - 100:s.end - 100] == s.text


# -- lexical ---------------------------------------------------------------
def test_bm25_ranks_the_matching_chunk_first() -> None:
    idx = BM25Index()
    idx.add(chunk_document(doc("a", "exponential backoff with full jitter")))
    idx.add(chunk_document(doc("b", "kubernetes ingress controller routing")))
    hits = idx.search("full jitter backoff")
    assert hits[0].chunk.doc_id == "a"
    assert "jitter" in hits[0].detail["matched_terms"]


def test_bm25_returns_nothing_for_an_absent_term() -> None:
    idx = BM25Index()
    idx.add(chunk_document(doc("a", "alpha beta gamma delta epsilon")))
    assert idx.search("zzzzquux") == []


def test_stopwords_are_dropped() -> None:
    assert "the" not in tokenize("the quick brown fox")
    assert "quick" in tokenize("the quick brown fox")


def test_empty_index_search_is_safe() -> None:
    assert BM25Index().search("anything") == []


# -- dense -----------------------------------------------------------------
def test_hashing_embedder_is_deterministic() -> None:
    e = HashingEmbedder()
    assert e.embed(["repeatable text"]) == e.embed(["repeatable text"])


def test_hashing_embedder_declares_itself_non_semantic() -> None:
    assert HashingEmbedder().is_semantic is False


def test_embeddings_are_unit_length() -> None:
    import math
    v = HashingEmbedder().embed(["some words here"])[0]
    assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, abs_tol=1e-9)


def test_dense_index_finds_the_lexically_overlapping_chunk() -> None:
    idx = DenseIndex(HashingEmbedder())
    idx.add(chunk_document(doc("a", "durable append only log with fsync")))
    idx.add(chunk_document(doc("b", "colourful parrots of the amazon basin")))
    hits = idx.search("append only log")
    assert hits and hits[0].chunk.doc_id == "a"


def test_dense_floor_suppresses_weak_matches() -> None:
    idx = DenseIndex(HashingEmbedder(), min_score=0.99)
    idx.add(chunk_document(doc("a", "entirely unrelated subject matter")))
    assert idx.search("quantum chromodynamics") == []


# -- graph -----------------------------------------------------------------
def test_bfs_finds_a_two_hop_path() -> None:
    g = DocumentGraph()
    g.add_edge("a", "b", EdgeType.DEPENDS_ON)
    g.add_edge("b", "c", EdgeType.DEPENDS_ON)
    path = g.shortest_path("a", "c")
    assert path is not None
    assert path.nodes == ["a", "b", "c"]
    assert path.hops == 2


def test_bfs_respects_the_hop_limit() -> None:
    g = DocumentGraph()
    g.add_edge("a", "b", EdgeType.DEPENDS_ON)
    g.add_edge("b", "c", EdgeType.DEPENDS_ON)
    assert "c" not in g.bfs("a", max_hops=1)


def test_traversal_can_filter_by_edge_type() -> None:
    g = DocumentGraph()
    g.add_edge("a", "b", EdgeType.DEPENDS_ON)
    g.add_edge("a", "c", EdgeType.RELATED_TO)
    reached = g.bfs("a", max_hops=1, types={EdgeType.DEPENDS_ON})
    assert "b" in reached and "c" not in reached


def test_unknown_start_node_returns_empty() -> None:
    assert DocumentGraph().bfs("nope") == {}


def test_duplicate_edges_are_not_stored_twice() -> None:
    g = DocumentGraph()
    g.add_edge("a", "b", EdgeType.MENTIONS)
    g.add_edge("a", "b", EdgeType.MENTIONS)
    assert len(g.edges) == 1


def test_mention_extraction_ignores_one_word_titles() -> None:
    docs = [doc("a", "this text mentions Store repeatedly", "Store"),
            doc("b", "unrelated body text", "Retry Policy")]
    g = extract_mentions(docs)
    assert not [e for e in g.edges if e.target == "a"]


def test_mention_extraction_links_multiword_titles() -> None:
    docs = [doc("a", "see the Retry Policy for details", "Alpha Doc"),
            doc("b", "body", "Retry Policy")]
    g = extract_mentions(docs)
    assert any(e.source == "a" and e.target == "b" for e in g.edges)


# -- fusion ----------------------------------------------------------------
def test_rrf_rewards_agreement_between_channels() -> None:
    chunks = chunk_document(doc("a", "alpha")) + chunk_document(doc("b", "beta"))
    a, b = chunks[0], chunks[1]
    fused = reciprocal_rank_fusion({
        "lexical": [ScoredChunk(b, 9.0), ScoredChunk(a, 1.0)],
        "dense": [ScoredChunk(a, 0.9), ScoredChunk(b, 0.1)],
        "graph": [ScoredChunk(a, 0.5)],
    })
    assert fused[0].chunk.chunk_id == a.chunk_id
    assert set(fused[0].channels) == {"lexical", "dense", "graph"}


def test_rrf_ignores_incomparable_score_scales() -> None:
    """A huge BM25 score must not outrank agreement, only its rank counts."""
    chunks = chunk_document(doc("a", "alpha")) + chunk_document(doc("b", "beta"))
    a, b = chunks[0], chunks[1]
    fused = reciprocal_rank_fusion({
        "lexical": [ScoredChunk(b, 10_000.0)],
        "dense": [ScoredChunk(a, 0.4)], "graph": [ScoredChunk(a, 0.4)],
    })
    assert fused[0].chunk.chunk_id == a.chunk_id


def test_rrf_on_no_channels_is_empty() -> None:
    assert reciprocal_rank_fusion({}) == []


# -- decomposition ---------------------------------------------------------
@pytest.mark.parametrize("query,strategy,n", [
    ("what is the retry policy", "single", 1),
    ("find the ledger and then find its store", "coordination", 2),
    ("which store backs the component that the policy depends on",
     "relative_clause", 2),
])
def test_decomposition_strategies(query, strategy, n) -> None:
    plan = decompose(query)
    assert plan.strategy == strategy
    assert len(plan.sub_queries) == n


def test_dependent_subquery_records_its_dependency() -> None:
    plan = decompose("which store backs the service that handles retries")
    assert plan.sub_queries[1].depends_on == 0
    assert plan.max_depth == 1


# -- metrics ---------------------------------------------------------------
def test_recall_at_k_counts_only_the_first_k() -> None:
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 1) == 0.5
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 3) == 1.0


def test_recall_with_no_relevant_documents_is_one() -> None:
    assert recall_at_k(["a"], set(), 5) == 1.0
