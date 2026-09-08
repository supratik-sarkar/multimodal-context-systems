# What is not validated here

## Multimodal retrieval

`MLXImageEmbedder` has never been executed. It is written against the mlx-vlm
API and requires Apple Silicon; this package was developed on x86 Linux where
`import mlx` raises ImportError.

The interface, the capability probe and the failure behaviour *are* tested.
What is not tested is whether the backend produces correct joint text/image
embeddings, because that cannot be established without running it.

There is deliberately no fallback. `MLXImageEmbedder.embed_text()` raises
`BackendUnavailable` rather than substituting another embedder. A fallback
would produce vectors that look like a multimodal result and are not one, and
a caller would have no way to tell.

Gate: `scripts/validate_multimodal_mac.sh`, which refuses to run on non-arm64.

## Semantic retrieval quality

CI runs `HashingEmbedder`, which is deterministic and dependency-free but
carries no semantics. It will not match "car" to "automobile". Every benchmark
result records `embedder_is_semantic: false` so the numbers are read correctly.

What the fixture benchmark therefore establishes: the pipeline is wired
correctly, fusion behaves as specified, multi-hop decomposition reaches its
terminal document, and citations resolve to real source spans. What it does
not establish: retrieval quality with a real embedding model.

## Scale

Exact nearest-neighbour search over an in-memory list. Correct at fixture
scale; wrong at real scale, where an ANN index and its recall trade-off become
necessary. Eight documents is not an information-retrieval evaluation, and no
comparison to a published benchmark is offered.

## Entailment

Citation grounding is lexical overlap between an answer sentence and candidate
source sentences. It confirms a sentence came from somewhere in the retrieved
evidence. It does not confirm the source actually supports the claim — that is
entailment, a different problem requiring a different model.
