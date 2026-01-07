"""
Vision embedding and similarity search utilities using CLIP + FAISS.

The :class:`VisionManager` provides lightweight image embedding, persistent FAISS
index handling and similarity search suitable for product or template lookup
within the AutOCR pipeline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, TypedDict
from collections import Counter

import numpy as np
from PIL import Image

# Imports moved to methods/lazy to avoid DLL conflicts with PaddleOCR
faiss = None
torch = None
open_clip = None


class VisionSearchResult(TypedDict, total=False):
    """Similarity search result entry."""

    path: str
    score: float


@dataclass
class VisionManagerConfig:
    """Configuration for :class:`VisionManager`."""

    enabled: bool = True
    index_path: str = os.path.join("data", "vision_index.faiss")
    embeddings_dir: str = os.path.join("data", "vision_embeddings")
    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    use_gpu: bool = False


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".jfif", ".avif"}


class VisionManager:
    """
    Manage CLIP embedding extraction and FAISS similarity search.

    Parameters
    ----------
    config:
        Defines model, storage locations and GPU usage.
    logger:
        Optional logger for diagnostics.
    """

    def __init__(
        self,
        config: VisionManagerConfig | None = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or VisionManagerConfig()
        self.logger = logger or logging.getLogger(__name__)
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = "cpu"
        self._index = None
        self._metadata: List[dict] = []
        self._load_runtime()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #


    def ensure_loaded(self) -> None:
        """Ensure model and index are loaded."""
        self._ensure_model()
        self._ensure_index()

    def embed_image(self, path: str) -> np.ndarray:
        """
        Compute a CLIP embedding for ``path``.

        Returns
        -------
        np.ndarray
            1D vector (L2 normalised) representing the image.
        """
        self._ensure_model()
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                tensor = self._preprocess(image).unsqueeze(0).to(self._device)  # type: ignore[attr-defined]
            with torch.no_grad():  # type: ignore[union-attr]
                embedding = self._model.encode_image(tensor)  # type: ignore[union-attr]
            embedding = embedding.detach().cpu().numpy().astype("float32")  # type: ignore[union-attr]
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            return embedding.squeeze(0)
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            return embedding.squeeze(0)
        except Exception as e:
            self.logger.error(f"Failed to embed image: {e}")
            return np.zeros(512, dtype="float32")

    def classify_image(self, path: str, candidates: List[str]) -> List[Tuple[str, float]]:
        """
        Perform zero-shot classification using CLIP.
        Returns list of (label, score) tuples sorted by score.
        """
        if not candidates:
            return []
            
        self._ensure_model()
        try:
            # 1. Embed Image
            with Image.open(path) as image:
                image = image.convert("RGB")
                image_tensor = self._preprocess(image).unsqueeze(0).to(self._device)
            
            with torch.no_grad():
                image_features = self._model.encode_image(image_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            # 2. Embed Text Candidates
            text_tokens = self._tokenizer(candidates).to(self._device)
            with torch.no_grad():
                text_features = self._model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)

            # 3. Compute Similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(len(candidates))

            results = []
            for value, index in zip(values, indices):
                score = value.item()
                if score > 0.05: # Return mostly everything, filter later or let softmax handle it
                    results.append((candidates[index], score))
            
            
            return results

        except Exception as e:
            self.logger.error(f"Classification failed: {e}")
            return []

    def analyze_colors(self, image_path: str, num_colors: int = 5) -> list:
        """Extract dominant colors from an image."""
        try:
            image = Image.open(image_path).convert('RGB')
            # Resize for speed
            image = image.resize((150, 150))
            pixels = np.array(image).reshape(-1, 3)
            
            # Simple quantization (bucket by 32)
            quantized = pixels // 32 * 32
            pixels_tuple = [tuple(p) for p in quantized]
            
            counts = Counter(pixels_tuple)
            common = counts.most_common(num_colors)
            
            hex_colors = []
            for color, count in common:
                hex_c = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
                hex_colors.append({"hex": hex_c, "rgb": color, "count": count})
                
            return hex_colors
        except Exception as e:
            self.logger.error(f"Color analysis failed: {e}")
            return []

    def embed_text(self, text: str) -> np.ndarray:
        """
        Compute a CLIP embedding for a text query.
        """
        self._ensure_model()
        try:
            tokens = self._tokenizer(text).to(self._device)
            with torch.no_grad():
                embedding = self._model.encode_text(tokens)
            embedding = embedding.detach().cpu().numpy().astype("float32")
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            return embedding.squeeze(0)
        except Exception as e:
            self.logger.error(f"Error embedding text '{text}': {e}")
            return np.zeros(512, dtype="float32")

    def build_index(self, images_dir: str) -> None:
        """
        Build (or rebuild) the FAISS index from images under ``images_dir``.
        """
        if not self.config.enabled:
            self.logger.info("Vision search disabled; skipping index build.")
            return
        try:
            global faiss, torch, open_clip
            import faiss, torch, open_clip
        except ImportError:
            self.logger.warning(
                "Vision dependencies missing (faiss/torch/open_clip); cannot build index."
            )
            return

        image_paths = list(iter_image_files(images_dir))
        if not image_paths:
            self.logger.info("No images found in %s; clearing index.", images_dir)
            self._index = None
            self._metadata = []
            self._remove_existing_index()
            return

        embeddings = []
        metadata = []
        os.makedirs(self.config.embeddings_dir, exist_ok=True)

        for path in image_paths:
            try:
                embedding = self._load_or_compute_embedding(path)
            except Exception as exc:  # pragma: no cover - runtime dependency issues
                self.logger.error("Failed to embed %s: %s", path, exc)
                continue
            embeddings.append(embedding)
            metadata.append({"path": os.path.abspath(path)})

        if not embeddings:
            self.logger.warning("No embeddings produced; index not updated.")
            return

        matrix = np.stack(embeddings).astype("float32")
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)

        self._index = index
        self._metadata = metadata
        self._save_index(index, metadata)

    def search_similar(self, image_path: str, k: int = 10) -> List[VisionSearchResult]:
        """
        Search for images similar to ``image_path``.
        """
        if not self.config.enabled:
            return []
        # Lazy check imports
        try:
            global faiss, torch, open_clip
            if faiss is None: import faiss
            if torch is None: import torch
            if open_clip is None: import open_clip
        except ImportError:
            return []

        self._ensure_index()
        if self._index is None or not self._metadata:
            return []

        try:
            query = self.embed_image(image_path).astype("float32")
            return self.search_by_vector(query, k)
        except Exception as e:
            self.logger.error(f"Error searching similar to image {image_path}: {e}")
            return []

    def search_by_text(self, text: str, k: int = 10) -> List[VisionSearchResult]:
        """Search images semantically similar to text query."""
        if not self.config.enabled:
            return []
        
        self._ensure_index()
        if self._index is None or not self._metadata:
            self.logger.warning("Attempted search with empty index")
            return []

        query_vec = self.embed_text(text)
        return self.search_by_vector(query_vec, k)

    def search_by_vector(self, vector: np.ndarray, k: int = 10) -> List[VisionSearchResult]:
        if self._index is None:
            return []
            
        vector = vector.reshape(1, -1).astype("float32")
        distances, indices = self._index.search(vector, min(k, len(self._metadata)))

        results: List[VisionSearchResult] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            results.append(
                VisionSearchResult(
                    path=self._metadata[idx]["path"],
                    score=float(score),
                )
            )
        return results

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _load_runtime(self) -> None:
        try:
            global torch, open_clip
            import torch
            import open_clip
        except ImportError:
             self.logger.warning("Vision libs missing.")
             self.config.enabled = False
             return

        if self.config.use_gpu and torch.cuda.is_available():  # type: ignore[union-attr]
            self._device = "cuda"
        else:
            self._device = "cpu"

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if not self.config.enabled:
            raise RuntimeError("Vision features are disabled in configuration.")
        
        # Ensure imports
        global open_clip, torch
        import open_clip
        import torch

        model_name = self.config.model_name
        pretrained = self.config.pretrained

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self._device,
        )
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self._model.eval()  # type: ignore[union-attr]

    def _ensure_index(self) -> None:
        if self._index is not None and self._metadata:
            return
        if not os.path.exists(self.config.index_path):
            return
        if faiss is None:
            return
        try:
            self._index = faiss.read_index(self.config.index_path)
            metadata_path = self._metadata_path()
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as fh:
                    self._metadata = json.load(fh)
        except Exception as exc:  # pragma: no cover - disk corruption
            self.logger.error("Failed to load FAISS index: %s", exc)
            self._index = None
            self._metadata = []

    def _save_index(self, index, metadata: List[dict]) -> None:
        if faiss is None:
            return
        os.makedirs(os.path.dirname(self.config.index_path) or ".", exist_ok=True)
        faiss.write_index(index, self.config.index_path)
        with open(self._metadata_path(), "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, ensure_ascii=False, indent=2)

    def _remove_existing_index(self) -> None:
        try:
            if os.path.exists(self.config.index_path):
                os.remove(self.config.index_path)
            metadata_path = self._metadata_path()
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
        except OSError:
            self.logger.debug("Failed to remove existing vision index.", exc_info=True)

    def _load_or_compute_embedding(self, path: str) -> np.ndarray:
        embedding_path = self._embedding_path(path)
        if os.path.exists(embedding_path):
            try:
                if os.path.getmtime(embedding_path) >= os.path.getmtime(path):
                    cached = np.load(embedding_path, allow_pickle=False)
                    return cached.astype("float32")
            except Exception:
                self.logger.debug("Cached embedding invalid for %s; recomputing.", path, exc_info=True)
        embedding = self.embed_image(path).astype("float32")
        try:
            np.save(embedding_path, embedding)
        except Exception:
            self.logger.debug("Failed to cache embedding for %s.", path, exc_info=True)
        return embedding

    def _embedding_path(self, image_path: str) -> str:
        digest = hashlib.sha1(os.path.abspath(image_path).encode("utf-8")).hexdigest()
        return os.path.join(self.config.embeddings_dir, f"{digest}.npy")

    def _metadata_path(self) -> str:
        return os.path.join(
            os.path.dirname(self.config.index_path) or ".",
            "vision_index_metadata.json",
        )


def iter_image_files(root: str) -> Iterable[str]:
    for folder, _, files in os.walk(root):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in SUPPORTED_IMAGE_EXTS:
                yield os.path.join(folder, name)


__all__ = [
    "VisionManager",
    "VisionManagerConfig",
    "VisionSearchResult",
]
