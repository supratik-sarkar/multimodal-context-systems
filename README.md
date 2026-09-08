# multimodal-context-systems

Multimodal and context systems: hybrid graph retrieval, evidence-grounded context assembly, and provenance-aware GenAI data pipelines.

This repository unifies contemporary 2026 multimodal context retrieval and data synthesis architectures with historical multimodal deep-learning foundations.

```
+-----------------------------------------------------------------------------------+
|                         multimodal-context-systems                                |
+-----------------------------------------------------------------------------------+
                                          |
        +---------------------------------+---------------------------------+
        |                                                                   |
        v                                                                   v
+------------------------------------+             +--------------------------------+
|    components/context-engine       |             |   components/data-foundry      |
|  Hybrid graph + vector retrieval   | <---------> |  Multimodal dataset synthesis  |
|  Dense search, BM25, RRF, MLX      |             |  Validation & schema governance|
+------------------------------------+             +--------------------------------+
```

---

## Architecture & Subsystems

1. **Graph Context Engine (`components/context-engine/`)**:
   High-throughput hybrid retrieval engine combining dense vector search with structural graph traversal. Features BM25 keyword matching, reciprocal rank fusion (RRF), explicit citation tracking, and Apple Silicon native MLX multimodal embedders (`mlx-community/clip-vit-base-patch32`).
2. **GenAI Data Foundry (`components/data-foundry/`)**:
   End-to-end synthetic dataset generator and curator. Implements multi-stage validation, lexical diversity scoring, schema conformance, and verifiable data lineage.

---

## Verification & Test Baseline

The current active systems are tested and verified on Python 3.12.13:

| Component | Directory | Baseline Tests | Status |
| :--- | :--- | :---: | :---: |
| Graph Context Engine | `components/context-engine` | 54 | Verified |
| GenAI Data Foundry | `components/data-foundry` | 25 | Verified |
| **Total Baseline** | **Active Components** | **79 tests** | **Passing** |

### Running the Full Verification Suite

The repository includes a standalone verification script that provisions ephemeral virtual environments under `mktemp -d`:

```bash
./scripts/verify_all.sh
```

The script executes:
1. Strict Python 3.12.13 version confirmation (fails closed on mismatch with exit code 2);
2. Isolated ephemeral venv provisioning;
3. Dependency installation for each subsystem;
4. Pytest test execution across all 79 tests;
5. Code quality, formatting, and typing gates (Ruff / mypy);
6. Automatic teardown of temporary environments upon completion.

---

## Apple Silicon MLX Hardware Validation

On Apple Silicon (M4 Pro / macOS 15 Darwin 24+), the context engine exercises the Apple MLX multimodal runtime via:
```bash
./components/context-engine/scripts/validate_multimodal_mac.sh
```
The embedder validates device architecture to ensure reproducible local acceleration.

---

## Historical Lineage & Provenance

This repository is anchored on the genuine 2025 `multimodal-stackgan` GitHub repository object. Historical research components are preserved under `legacy/`:
* `legacy/multimodal-stackgan/`: Multi-stage generative adversarial networks synthesizing imagery conditioned on text descriptions.
* `legacy/signature-similarity/`: Multimodal feature representation and invariant signature matching.

In September 2026, the repository was transitioned into the `multimodal-context-systems` umbrella, integrating the two modern systems with un-squashed Git commit histories.

See [HISTORY.md](HISTORY.md) for full lineage proofs, earliest commit timestamps, component tags, and commit links.

---

## License

Root orchestration and 2026 systems are licensed under Apache-2.0. Historical components retain their original notices. See [LICENSE](LICENSE) and [LICENSES.md](LICENSES.md).
