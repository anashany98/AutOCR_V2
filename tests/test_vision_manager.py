"""VisionManager roundtrip tests."""

import logging
from pathlib import Path

import numpy as np
import pytest

from modules.vision_manager import VisionManager, VisionManagerConfig

try:  # pragma: no cover - optional dependency
    import faiss  # noqa: F401
except ImportError:  # pragma: no cover
    faiss = None


@pytest.mark.skipif(faiss is None, reason="faiss is required for this test")
def test_build_and_search_roundtrip(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Create dummy images
    from PIL import Image

    for idx in range(3):
        img = Image.new("RGB", (32, 32), (idx * 40, idx * 40, idx * 40))
        img.save(images_dir / f"img_{idx}.png")

    manager = VisionManager.__new__(VisionManager)
    manager.config = VisionManagerConfig(
        enabled=True,
        index_path=str(tmp_path / "index.faiss"),
        embeddings_dir=str(tmp_path / "embeddings"),
        model_name="test",
        use_gpu=False,
    )
    manager.logger = logging.getLogger("vision-test")
    manager._index = None
    manager._metadata = []

    def fake_embed(path: str) -> np.ndarray:
        seed = sum(ord(ch) for ch in Path(path).name)
        rng = np.random.default_rng(seed)
        vec = rng.random(16, dtype=np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    manager.embed_image = fake_embed  # type: ignore
    manager._ensure_model = lambda: None  # type: ignore
    manager._device = "cpu"

    manager.build_index(str(images_dir))
    assert manager._index is not None
    assert manager._metadata

    results = manager.search_similar(str(images_dir / "img_0.png"), k=2)
    assert results
    top_paths = [Path(result["path"]).name for result in results]
    assert "img_0.png" in top_paths
