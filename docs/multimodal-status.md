# Multimodal status

## Current state

**Implemented and never executed.**

`MLXMultimodalEmbedder` is a complete adapter: capability probing, lazy model
loading, text embedding, image embedding through a processor, and explicit
failure when the runtime is absent. It is code, not a stub.

It has never run. This repository was developed on x86-64 Linux, where:

```
$ gce --embedder mlx capabilities
{
  "name": "mlx:mlx-community/clip-vit-base-patch32",
  "text": false,
  "image": false,
  "semantic": false,
  "reason": "MLX requires Apple Silicon; running on linux/x86_64"
}
```

## What that means for the numbers

There are no multimodal numbers in this repository. Not placeholders, not
estimates, not a chart built from a template. `test_no_multimodal_result_is_produced_off_apple_silicon`
fails the build if a result file on a non-Apple-Silicon platform contains
image-related keys.

Every benchmark figure currently committed comes from `DeterministicEmbedder`,
which declares `semantic: false`. Those figures show that the pipeline
retrieves the correct objects on the fixture corpus. They say nothing about
embedding quality, and the report records the embedder's declared capabilities
alongside the numbers so the distinction survives copying.

## Why the deterministic backend refuses images

It would be trivial to hash an image filename and return a vector. That would
produce a number, a chart, and a completely false impression. The backend
raises `UnsupportedModality` instead.

## What the Mac run will change

```bash
pip install -e ".[multimodal]"
./scripts/validate_multimodal_mac.sh
```

The script refuses to run off Apple Silicon and exits non-zero rather than
degrading. On success it writes `results/benchmark-mlx-text.json` and
`results/multimodal-query.json` — the first genuine multimodal artefacts this
repository will contain.

Until then this repository is **PARTIALLY_SANDBOX_VERIFIED**: every text,
graph, fusion, citation and metric path is tested and passing, and one runtime
is unexercised and says so.
