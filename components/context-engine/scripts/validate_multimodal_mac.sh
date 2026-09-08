#!/usr/bin/env bash
# Apple Silicon multimodal validation. Refuses to run anywhere else.
#
# This is the gate that moves 02-graph-context-engine from
# PARTIALLY_SANDBOX_VERIFIED to SANDBOX_VERIFIED. Until it passes, the
# repository makes no multimodal semantic claim.
set -euo pipefail

ARCH="$(uname -m)"
if [ "$ARCH" != "arm64" ]; then
  echo "REFUSING TO RUN: architecture is ${ARCH}, expected arm64."
  echo "MLX requires Apple Silicon. There is no fallback path, by design:"
  echo "a fallback would produce vectors that look like a multimodal result"
  echo "and are not one."
  exit 2
fi

echo "architecture: ${ARCH}"
python3 -c "import mlx.core; print('mlx: available')" || {
  echo "MLX not importable. Install with: pip install -e '.[mac]'"
  exit 3
}

echo
echo "--- capability probe ---"
python3 - <<'PY'
from gce import MLXImageEmbedder, MultimodalIndex
emb = MLXImageEmbedder()
print(f"embedder:   {emb.name}")
print(f"semantic:   {emb.is_semantic}")
print(f"available:  {emb.available()}")
idx = MultimodalIndex(emb)
for k, v in idx.capability_report().items():
    print(f"  {k}: {v}")
PY

echo
echo "--- text/image joint retrieval ---"
echo "Place sample images in examples/images/ and run:"
echo "  gce multimodal --images examples/images --query 'a diagram of a queue'"
echo
echo "Record the outcome in PORTFOLIO_VERIFICATION.yaml."
echo "If any step above failed, the repository stays PARTIALLY_SANDBOX_VERIFIED."
echo "Do not edit the status without a passing run."
