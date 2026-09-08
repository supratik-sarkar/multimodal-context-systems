"""Multimodal retrieval interface and its Apple Silicon backend.

The interface is complete and tested. The MLX backend below is written but has
NEVER been executed: MLX requires Apple Silicon, and this package was developed
on x86 Linux where ``import mlx`` fails with ImportError.

That distinction is enforced in code rather than left to a README sentence.
``MLXImageEmbedder.available()`` probes the real import, and every method
raises ``BackendUnavailable`` when it fails. There is no fallback path that
quietly produces vectors from something else -- a silent fallback would let a
caller believe multimodal retrieval was validated when it was not.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .model import Chunk, Modality, ScoredChunk


class BackendUnavailable(RuntimeError):
    """Raised when a backend's runtime is not present on this machine."""


@runtime_checkable
class ImageEmbedder(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    @property
    def is_semantic(self) -> bool: ...

    def available(self) -> bool: ...

    def embed_images(self, paths: list[Path]) -> list[list[float]]: ...

    def embed_text(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(slots=True)
class MLXImageEmbedder:
    """CLIP-style joint text/image embedding via MLX.

    NOT VALIDATED. Written against the mlx-vlm API and never run. It is
    included because the interface, the capability probe and the failure
    behaviour are testable without Apple Silicon -- and because shipping the
    interface without an implementation would be its own kind of pretence.

    Run ``scripts/validate_multimodal_mac.sh`` on Apple Silicon to exercise it.
    Until that passes, this repository claims no multimodal semantic result.
    """

    model_id: str = "mlx-community/clip-vit-base-patch32"
    dims: int = 512

    @property
    def name(self) -> str:
        return f"mlx:{self.model_id}"

    @property
    def dimensions(self) -> int:
        return self.dims

    @property
    def is_semantic(self) -> bool:
        return True

    def available(self) -> bool:
        try:
            import mlx.core  # noqa: F401
        except ImportError:
            return False
        return True

    def _require(self) -> None:
        if not self.available():
            raise BackendUnavailable(
                "MLX is not available on this machine. This backend requires "
                "Apple Silicon; it has never been executed in this "
                "repository's CI. Run scripts/validate_multimodal_mac.sh on a "
                "Mac to validate it."
            )

    def _load_model(self) -> tuple[Any, Any]:
        self._require()
        from huggingface_hub import snapshot_download
        from transformers import AutoProcessor

        from .clip_model import CLIPModel

        model_path = snapshot_download(self.model_id)
        model = CLIPModel.from_pretrained(model_path)
        processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return model, processor

    def embed_images(self, paths: list[Path]) -> list[list[float]]:
        self._require()
        import mlx.core as mx
        from PIL import Image

        model, processor = self._load_model()
        images = [Image.open(p).convert("RGB") for p in paths]
        inputs = processor(images=images, return_tensors="np")
        pixel_values = mx.array(inputs["pixel_values"]).transpose(0, 2, 3, 1)
        features = model.get_image_features(pixel_values)
        return [_normalise(list(map(float, row))) for row in features]

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        self._require()
        import mlx.core as mx

        model, processor = self._load_model()
        inputs = processor(text=texts, return_tensors="np", padding=True)
        input_ids = mx.array(inputs["input_ids"])
        features = model.get_text_features(input_ids)
        return [_normalise(list(map(float, row))) for row in features]


@dataclass(slots=True)
class StubImageEmbedder:
    """Deterministic image embedder for interface tests.

    Hashes file bytes into a vector. It carries no visual semantics whatsoever
    and ``is_semantic`` is False, so no result produced with it can be mistaken
    for a multimodal retrieval measurement.
    """

    dims: int = 64

    @property
    def name(self) -> str:
        return f"stub-image-{self.dims}d"

    @property
    def dimensions(self) -> int:
        return self.dims

    @property
    def is_semantic(self) -> bool:
        return False

    def available(self) -> bool:
        return True

    def _vec(self, payload: bytes) -> list[float]:
        vec = [0.0] * self.dims
        digest = hashlib.blake2b(payload, digest_size=self.dims).digest()
        for i, byte in enumerate(digest):
            vec[i % self.dims] += (byte - 127.5) / 127.5
        return _normalise(vec)

    def embed_images(self, paths: list[Path]) -> list[list[float]]:
        return [self._vec(Path(p).read_bytes()) for p in paths]

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t.encode()) for t in texts]


def _normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


@dataclass(slots=True)
class MultimodalIndex:
    """Joint text/image retrieval over one shared vector space."""

    embedder: ImageEmbedder
    chunks: list[Chunk] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)

    def add_images(self, paths: list[Path], doc_id: str = "images") -> None:
        if not paths:
            return
        vectors = self.embedder.embed_images(paths)
        for path, vec in zip(paths, vectors, strict=True):
            self.chunks.append(Chunk(
                chunk_id=f"{doc_id}::{Path(path).name}", doc_id=doc_id,
                text=Path(path).name, start=0, end=0,
                modality=Modality.IMAGE,
                metadata={"path": str(path), "embedder": self.embedder.name},
            ))
            self.vectors.append(vec)

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        if not self.chunks:
            return []
        q = self.embedder.embed_text([query])[0]
        scored = [
            ScoredChunk(c, sum(x * y for x, y in zip(q, v, strict=True)),
                        "multimodal",
                        {"embedder": self.embedder.name,
                         "is_semantic": self.embedder.is_semantic})
            for c, v in zip(self.chunks, self.vectors, strict=True)
        ]
        scored.sort(key=lambda s: (-s.score, s.chunk.chunk_id))
        return scored[:top_k]

    def capability_report(self) -> dict[str, Any]:
        return {
            "embedder": self.embedder.name,
            "dimensions": self.embedder.dimensions,
            "is_semantic": self.embedder.is_semantic,
            "runtime_available": self.embedder.available(),
            "indexed_items": len(self.chunks),
            "semantic_validation": (
                "VALIDATED" if self.embedder.is_semantic
                and self.embedder.available()
                else "MAC_VALIDATION_REQUIRED"
            ),
        }


__all__ = ["BackendUnavailable", "ImageEmbedder", "MLXImageEmbedder",
           "MultimodalIndex", "StubImageEmbedder"]
