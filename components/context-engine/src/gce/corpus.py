"""Benchmark fixture corpus with a labelled query set.

Small, hand-built and fully labelled. Every query names the documents that
genuinely answer it, so Recall@k is measured against ground truth rather than
against the engine's own output.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .graph import DocumentGraph, EdgeType
from .model import Document


@dataclass(frozen=True, slots=True)
class LabelledQuery:
    query: str
    relevant_docs: frozenset[str]
    multi_hop: bool = False
    expected_path: tuple[str, ...] = field(default=())


DOCUMENTS: list[Document] = [
    Document("retry-policy", title="Retry Policy",
             text=(
                 "The retry policy governs how the dispatch service reacts to "
                 "transient failures. Requests are retried with exponential "
                 "backoff and full jitter.\n\n"
                 "Retries are capped at four attempts. Beyond that the request "
                 "is written to the dead letter queue for manual inspection. "
                 "The policy depends on the Idempotency Ledger to ensure a "
                 "retried request does not duplicate a completed side effect."
             )),
    Document("idempotency-ledger", title="Idempotency Ledger",
             text=(
                 "The Idempotency Ledger records an entry for every side "
                 "effecting operation, keyed by a deterministic operation "
                 "identifier derived from the request payload.\n\n"
                 "An operation marked completed is never executed a second "
                 "time. An operation marked started is retried, because its "
                 "outcome is genuinely unknown. The ledger is stored in the "
                 "Durable Store."
             )),
    Document("durable-store", title="Durable Store",
             text=(
                 "The Durable Store is an append only log. Entries are never "
                 "updated in place, so the history of an operation can always "
                 "be reconstructed.\n\n"
                 "Writes are fsynced before acknowledgement. The store is "
                 "replicated across three availability zones and is the "
                 "system's only stateful dependency."
             )),
    Document("dispatch-service", title="Dispatch Service",
             text=(
                 "The Dispatch Service accepts inbound work and routes it to "
                 "the appropriate handler. It is stateless and horizontally "
                 "scalable.\n\n"
                 "On failure it consults the Retry Policy. It emits a trace "
                 "span for every routing decision."
             )),
    Document("observability", title="Observability Guide",
             text=(
                 "Every service emits OpenTelemetry spans. Span names follow "
                 "a dotted convention indicating the subsystem.\n\n"
                 "Traces are sampled at ten percent in steady state and at "
                 "one hundred percent when an error status is recorded. The "
                 "Dispatch Service is the usual root span."
             )),
    Document("access-control", title="Access Control",
             text=(
                 "Principals hold scopes. A scope grants permission to invoke "
                 "a named capability.\n\n"
                 "Delegation cannot widen authority: a delegate never receives "
                 "a scope the delegator does not hold. Grants carry an expiry "
                 "and a single use nonce."
             )),
    Document("dead-letter-queue", title="Dead Letter Queue",
             text=(
                 "The dead letter queue holds requests that exhausted their "
                 "retries. Entries are retained for fourteen days.\n\n"
                 "Replaying an entry re-enters it at the dispatch stage. "
                 "Replay is a privileged operation and requires an explicit "
                 "scope."
             )),
    Document("backoff-maths", title="Backoff Maths",
             text=(
                 "Exponential backoff with full jitter samples the delay "
                 "uniformly between zero and the current ceiling.\n\n"
                 "Full jitter reduces synchronised retry storms far more "
                 "effectively than equal jitter, at the cost of higher "
                 "variance in individual request latency."
             )),
]

QUERIES: list[LabelledQuery] = [
    LabelledQuery("how are transient failures retried",
                  frozenset({"retry-policy"})),
    LabelledQuery("what prevents a duplicated side effect",
                  frozenset({"idempotency-ledger"})),
    LabelledQuery("where is the ledger stored",
                  frozenset({"idempotency-ledger", "durable-store"})),
    LabelledQuery("append only log fsync replication",
                  frozenset({"durable-store"})),
    LabelledQuery("what happens when retries are exhausted",
                  frozenset({"retry-policy", "dead-letter-queue"})),
    LabelledQuery("how does full jitter compare to equal jitter",
                  frozenset({"backoff-maths"})),
    LabelledQuery("can a delegate receive a wider scope",
                  frozenset({"access-control"})),
    LabelledQuery("what is the sampling rate for traces",
                  frozenset({"observability"})),
    LabelledQuery(
        "what stores the ledger that prevents duplicated side effects",
        frozenset({"idempotency-ledger", "durable-store"}),
        multi_hop=True, expected_path=("idempotency-ledger", "durable-store"),
    ),
    LabelledQuery(
        "which store backs the component that the retry policy depends on",
        frozenset({"retry-policy", "idempotency-ledger", "durable-store"}),
        multi_hop=True,
        expected_path=("retry-policy", "idempotency-ledger", "durable-store"),
    ),
]


def build_graph() -> DocumentGraph:
    """Explicit relations. Mention edges are extracted separately."""
    g = DocumentGraph()
    for doc in DOCUMENTS:
        g.add_node(doc.doc_id)
    g.add_edge("retry-policy", "idempotency-ledger", EdgeType.DEPENDS_ON, 1.0)
    g.add_edge("idempotency-ledger", "durable-store", EdgeType.DEPENDS_ON, 1.0)
    g.add_edge("dispatch-service", "retry-policy", EdgeType.DEPENDS_ON, 1.0)
    g.add_edge("retry-policy", "dead-letter-queue", EdgeType.RELATED_TO, 0.8)
    g.add_edge("retry-policy", "backoff-maths", EdgeType.RELATED_TO, 0.8)
    g.add_edge("dispatch-service", "observability", EdgeType.RELATED_TO, 0.6)
    g.add_edge("dead-letter-queue", "access-control", EdgeType.RELATED_TO, 0.6)
    return g


__all__ = ["DOCUMENTS", "QUERIES", "LabelledQuery", "build_graph"]
