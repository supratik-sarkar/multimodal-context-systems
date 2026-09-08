"""End-to-end retrieval, multi-hop paths, evidence and the multimodal gate."""
from __future__ import annotations

import pytest

from gce import (
    DOCUMENTS,
    QUERIES,
    BackendUnavailable,
    Citation,
    Document,
    GraphContextEngine,
    MLXImageEmbedder,
    MultimodalIndex,
    Reranker,
    RetrievalConfig,
    StubImageEmbedder,
    attribution_rate,
    build_graph,
    ground_citations,
    summarise,
    unsupported_sentences,
)


@pytest.fixture
def engine() -> GraphContextEngine:
    e = GraphContextEngine()
    e.graph = build_graph()
    e.add_documents(DOCUMENTS)
    return e


# -- retrieval quality -----------------------------------------------------
def test_recall_at_5_on_the_labelled_set(engine) -> None:
    per_query = [(engine.retrieve(q.query).doc_ids, set(q.relevant_docs))
                 for q in QUERIES]
    summary = summarise(per_query)
    assert summary.recall_at_5 >= 0.8, summary.to_dict()


def test_every_single_hop_query_finds_a_relevant_document(engine) -> None:
    for q in (q for q in QUERIES if not q.multi_hop):
        got = engine.retrieve(q.query).doc_ids
        assert set(got) & set(q.relevant_docs), f"{q.query} -> {got}"


def test_channels_all_contribute(engine) -> None:
    result = engine.retrieve("what prevents a duplicated side effect")
    assert set(result.channel_counts) >= {"lexical", "dense"}
    assert sum(result.channel_counts.values()) > 0


def test_disabling_a_channel_changes_the_provenance(engine) -> None:
    engine.config = RetrievalConfig(use_dense=False)
    result = engine.retrieve("append only log")
    assert "dense" not in result.bundle.channels


# -- multi-hop -------------------------------------------------------------
@pytest.mark.parametrize("q", [q for q in QUERIES if q.multi_hop],
                         ids=lambda q: q.query[:34])
def test_multi_hop_queries_reach_the_terminal_document(engine, q) -> None:
    result = engine.retrieve(q.query)
    assert result.decomposition.is_multi_hop
    terminal = q.expected_path[-1]
    assert terminal in result.doc_ids, (
        f"expected {terminal}, got {result.doc_ids}"
    )


def test_multi_hop_records_the_traversal_path(engine) -> None:
    result = engine.retrieve(
        "which store backs the component that the retry policy depends on"
    )
    assert result.bundle.paths
    assert any(len(p["nodes"]) >= 2 for p in result.bundle.paths)


def test_graph_channel_reaches_a_document_no_keyword_matches(engine) -> None:
    """The point of the graph channel: a fact one hop from the query terms."""
    engine.config = RetrievalConfig(use_lexical=False, use_dense=False,
                                    use_graph=True)
    result = engine._retrieve_once("anything", seeds=["retry-policy"])[0]
    assert "idempotency-ledger" in {r.chunk.doc_id for r in result}


# -- evidence and citations ------------------------------------------------
def test_citations_resolve_to_the_exact_source_span(engine) -> None:
    answer = "An operation marked completed is never executed a second time."
    result = engine.retrieve("an operation marked completed is never executed again",
                             answer=answer)
    assert result.bundle.citations
    report = result.bundle.verify_all(engine.documents)
    assert report["all_verified"], report["unverifiable"]


def test_attribution_rate_is_one_for_a_fully_grounded_answer(engine) -> None:
    answer = "The Durable Store is an append only log."
    result = engine.retrieve("append only log", answer=answer)
    assert attribution_rate(answer, result.bundle.citations) == 1.0


def test_an_unsupported_sentence_gets_no_citation(engine) -> None:
    """The central guarantee: no source, no citation."""
    answer = "Quantum entanglement determines the retry ceiling on Tuesdays."
    result = engine.retrieve("retry policy", answer=answer)
    assert result.bundle.citations == []
    assert unsupported_sentences(answer, []) == [answer]


def test_a_tampered_citation_fails_verification(engine) -> None:
    bad = Citation("durable-store", "durable-store::c0", 0, 20,
                   "text that is not actually at those offsets")
    assert bad.verify(engine.documents) is False


def test_citation_to_an_unknown_document_fails_cleanly(engine) -> None:
    assert Citation("nope", "nope::c0", 0, 5, "x").verify(engine.documents) is False


def test_grounding_against_no_chunks_returns_nothing() -> None:
    assert ground_citations("Some claim here.", []) == []


# -- reranker --------------------------------------------------------------
def test_reranker_declares_it_is_not_model_based() -> None:
    assert Reranker().is_model_based is False


def test_reranking_preserves_the_result_set(engine) -> None:
    engine.reranker = Reranker()
    engine.config = RetrievalConfig(rerank=True)
    result = engine.retrieve("append only log fsync")
    assert result.results
    assert len(set(result.chunk_ids)) == len(result.chunk_ids)


# -- multimodal: interface tested, backend explicitly not validated ---------
def test_mlx_backend_is_unavailable_on_this_platform() -> None:
    emb = MLXImageEmbedder()
    if emb.available():
        pytest.skip("MLX present; this assertion is for non-Apple-Silicon CI")
    assert emb.is_semantic is True


def test_mlx_backend_raises_rather_than_falling_back() -> None:
    """No silent fallback. An unavailable backend must fail loudly."""
    emb = MLXImageEmbedder()
    if emb.available():
        pytest.skip("MLX present")
    with pytest.raises(BackendUnavailable, match="Apple Silicon"):
        emb.embed_text(["a query"])
    with pytest.raises(BackendUnavailable):
        emb.embed_images([])


def test_capability_report_marks_multimodal_as_requiring_the_mac() -> None:
    idx = MultimodalIndex(MLXImageEmbedder())
    report = idx.capability_report()
    if report["runtime_available"]:
        pytest.skip("MLX present")
    assert report["semantic_validation"] == "MAC_VALIDATION_REQUIRED"


def test_stub_image_embedder_exercises_the_interface(tmp_path) -> None:
    p = tmp_path / "a.png"
    p.write_bytes(b"\x89PNG fake bytes for interface testing")
    idx = MultimodalIndex(StubImageEmbedder())
    idx.add_images([p])
    assert len(idx.chunks) == 1
    assert idx.search("anything")[0].detail["is_semantic"] is False


def test_stub_embedder_never_claims_semantics() -> None:
    report = MultimodalIndex(StubImageEmbedder()).capability_report()
    assert report["is_semantic"] is False
    assert report["semantic_validation"] == "MAC_VALIDATION_REQUIRED"


# -- engine reporting ------------------------------------------------------
def test_capability_report_states_the_embedder_is_not_semantic(engine) -> None:
    report = engine.capability_report()
    assert report["embedder_is_semantic"] is False
    assert report["semantic_retrieval_validated"] is False


def test_retrieval_on_an_empty_engine_is_safe() -> None:
    assert GraphContextEngine().retrieve("anything").results == []


def test_ingesting_an_empty_document_adds_no_chunks() -> None:
    e = GraphContextEngine()
    assert e.add_documents([Document("empty", "")]) == 0


def test_retrieval_is_deterministic(engine) -> None:
    a = engine.retrieve("what happens when retries are exhausted")
    b = engine.retrieve("what happens when retries are exhausted")
    assert a.chunk_ids == b.chunk_ids


def test_answer_not_present_in_retrieved_evidence_is_not_cited() -> None:
    """Grounding is against what was actually retrieved, not the whole corpus.

    Two unconnected documents. The answer sentence is verbatim from the one
    the query does not surface, so it must go uncited: an evidence bundle may
    only support what it contains.
    """
    e = GraphContextEngine()
    e.add_documents([
        Document("alpha", "Exponential backoff samples the delay uniformly."),
        Document("beta", "Entries are retained for fourteen days."),
    ])
    answer = "Entries are retained for fourteen days."
    result = e.retrieve("exponential backoff delay", answer=answer)
    assert "beta" not in {c.doc_id for c in result.bundle.chunks}
    assert result.bundle.citations == []
