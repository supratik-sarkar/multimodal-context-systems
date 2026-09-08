"""Embedder interfaces and backends.

Three implementations with sharply different honesty properties:

``DeterministicEmbedder``  a hashed bag-of-words projection. Reproducible, needs
                           no model, and is what CI uses. It carries real
                           lexical signal and **no semantic signal** -- it will
                           not relate "car" to "automobile" and does not claim to.
``SentenceTransformerEmbedder``  a real text model, optional dependency.
``MLXMultimodalEmbedder``  a real Apple-Silicon image/text model. The adapter is
                           written; it cannot run off Apple Silicon and reports
                           that rather than degrading silently.

Capability detection is explicit so callers can branch on what is genuinely
available instead of discovering it through an exception.
"""
from __future__ import annotations

import hashlib
import math
import platform
import sys
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .lexical import tokenize
from .objects import CanonicalObject, Modality


@dataclass(frozen=True, slots=True)
class EmbedderCapabilities:
    text: bool
    image: bool
    semantic: bool
    name: str
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "text": self.text, "image": self.image,
                "semantic": self.semantic, "reason": self.reason}


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def capabilities(self) -> EmbedderCapabilities: ...
    def embed_text(self, texts: list[str]) -> list[list[float]]: ...
    def embed_object(self, obj: CanonicalObject) -> list[float]: ...


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if not na or not nb else num / (na * nb)


class DeterministicEmbedder:
    """Hashed bag-of-words projection. Reproducible, offline, non-semantic."""

    def __init__(self, dim: int = 256) -> None:
        # Wider than strictly needed for a small corpus: a narrow projection
        # produces hash collisions, which show up as non-zero similarity
        # between unrelated texts. That is a property of the trick, not of the
        # documents, and it should not leak into retrieval results.
        self.dim = dim

    def capabilities(self) -> EmbedderCapabilities:
        return EmbedderCapabilities(
            text=True, image=False, semantic=False, name="deterministic",
            reason="hashed bag-of-words; lexical signal only, no semantic "
                   "similarity. Suitable for CI determinism, not for quality "
                   "measurement.",
        )

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in tokenize(text):
            h = int(hashlib.sha256(tok.encode()).hexdigest()[:8], 16)
            vec[h % self.dim] += 1.0
            vec[(h >> 8) % self.dim] += 0.5
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_object(self, obj: CanonicalObject) -> list[float]:
        if obj.modality is Modality.IMAGE:
            # No pixels are read. Deriving a vector from a filename would be a
            # lexical trick dressed as image understanding.
            raise UnsupportedModality(
                "deterministic embedder cannot embed images; use an "
                "image-capable backend on Apple Silicon"
            )
        return self._one(obj.text)


class UnsupportedModality(RuntimeError):
    """Raised when a backend is asked for a modality it cannot honestly serve."""


class BackendUnavailable(RuntimeError):
    """Raised when a backend's runtime is absent on this platform."""


class SentenceTransformerEmbedder:
    """Real text embeddings. Optional dependency, loaded lazily."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
                 ) -> None:
        self.model_name = model_name
        self._model = None
        self.dim = 384

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise BackendUnavailable(
                    f"sentence-transformers is not installed: {exc}"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
            self.dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def capabilities(self) -> EmbedderCapabilities:
        try:
            import sentence_transformers  # noqa: F401
            available = True
            reason = ""
        except ImportError as exc:
            available = False
            reason = f"sentence-transformers not installed: {exc}"
        return EmbedderCapabilities(text=available, image=False,
                                    semantic=available,
                                    name=f"sentence-transformers:{self.model_name}",
                                    reason=reason)

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._load().encode(texts)]

    def embed_object(self, obj: CanonicalObject) -> list[float]:
        if obj.modality is Modality.IMAGE:
            raise UnsupportedModality("text-only backend")
        return self.embed_text([obj.text])[0]


def apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


class MLXMultimodalEmbedder:
    """Image and text embeddings via MLX on Apple Silicon.

    The adapter is complete. It refuses to run anywhere else rather than
    falling back to something that would produce numbers without meaning.
    """

    def __init__(self, model_name: str = "mlx-community/clip-vit-base-patch32"
                 ) -> None:
        self.model_name = model_name
        self.dim = 512
        self._model = None
        self._processor = None

    def _probe(self) -> tuple[bool, str]:
        if not apple_silicon():
            return False, (
                f"MLX requires Apple Silicon; running on {sys.platform}/"
                f"{platform.machine()}"
            )
        try:
            import mlx.core  # noqa: F401
        except ImportError as exc:
            return False, f"mlx is not installed: {exc}"
        try:
            from mlx_vlm import load  # noqa: F401
        except ImportError as exc:
            return False, f"mlx-vlm is not installed: {exc}"
        return True, ""

    def capabilities(self) -> EmbedderCapabilities:
        ok, reason = self._probe()
        return EmbedderCapabilities(text=ok, image=ok, semantic=ok,
                                    name=f"mlx:{self.model_name}", reason=reason)

    def _load(self):
        ok, reason = self._probe()
        if not ok:
            raise BackendUnavailable(reason)
        if self._model is None:
            from mlx_vlm import load
            self._model, self._processor = load(self.model_name)
        return self._model, self._processor

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        model, processor = self._load()
        inputs = processor(text=texts, return_tensors="np", padding=True)
        out = model.get_text_features(**inputs)
        return [list(map(float, v)) for v in out]

    def embed_object(self, obj: CanonicalObject) -> list[float]:
        model, processor = self._load()
        if obj.modality is Modality.IMAGE:
            from PIL import Image
            if not obj.image_path:
                raise UnsupportedModality(f"{obj.object_id} has no image_path")
            image = Image.open(obj.image_path).convert("RGB")
            inputs = processor(images=[image], return_tensors="np")
            out = model.get_image_features(**inputs)
            return list(map(float, out[0]))
        return self.embed_text([obj.text])[0]


def build_embedder(kind: str = "deterministic", **kw) -> Embedder:
    match kind:
        case "deterministic":
            return DeterministicEmbedder(**kw)
        case "sentence-transformers":
            return SentenceTransformerEmbedder(**kw)
        case "mlx":
            return MLXMultimodalEmbedder(**kw)
        case _:
            raise ValueError(f"unknown embedder kind: {kind}")


__all__ = ["BackendUnavailable", "DeterministicEmbedder", "Embedder",
           "EmbedderCapabilities", "MLXMultimodalEmbedder",
           "SentenceTransformerEmbedder", "UnsupportedModality",
           "apple_silicon", "build_embedder", "cosine"]
