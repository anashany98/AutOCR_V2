"""
Performance Benchmark Suite for AutoOCR Document AI Pipeline.

Measures throughput, latency, and resource usage across all pipeline stages.

Usage::

    # Full benchmark
    python scripts/benchmark.py --pages 50

    # OCR-only benchmark
    python scripts/benchmark.py --stage ocr --pages 20

    # Save results to JSON
    python scripts/benchmark.py --pages 50 --output results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Benchmark Metrics
# ============================================================================

@dataclass
class StageMetrics:
    """Timing and resource metrics for a single pipeline stage."""
    stage: str
    total_time_s: float = 0.0
    avg_time_per_page_s: float = 0.0
    min_time_s: float = float("inf")
    max_time_s: float = 0.0
    pages_processed: int = 0
    items_produced: int = 0
    errors: int = 0
    throughput_pages_per_min: float = 0.0

    def finalize(self):
        if self.pages_processed > 0:
            self.avg_time_per_page_s = self.total_time_s / self.pages_processed
            self.throughput_pages_per_min = (self.pages_processed / self.total_time_s) * 60 if self.total_time_s > 0 else 0


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""
    total_pages: int = 0
    total_time_s: float = 0.0
    stages: Dict[str, StageMetrics] = field(default_factory=dict)
    system_info: Dict[str, Any] = field(default_factory=dict)
    throughput_e2e_pages_per_min: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_pages": self.total_pages,
            "total_time_s": round(self.total_time_s, 2),
            "throughput_e2e_pages_per_min": round(self.throughput_e2e_pages_per_min, 1),
            "stages": {
                k: {
                    "total_time_s": round(v.total_time_s, 2),
                    "avg_per_page_s": round(v.avg_time_per_page_s, 3),
                    "min_s": round(v.min_time_s, 3) if v.min_time_s != float("inf") else 0,
                    "max_s": round(v.max_time_s, 3),
                    "pages": v.pages_processed,
                    "items_produced": v.items_produced,
                    "throughput_ppm": round(v.throughput_pages_per_min, 1),
                    "errors": v.errors,
                }
                for k, v in self.stages.items()
            },
            "system_info": self.system_info,
        }


# ============================================================================
# System Info
# ============================================================================

def get_system_info() -> Dict[str, Any]:
    """Collect system information for context."""
    import platform
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }

    # GPU info
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 1)
            info["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        info["cuda_available"] = False

    # RAM
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / (1024**3), 1)
        info["ram_available_gb"] = round(mem.available / (1024**3), 1)
    except ImportError:
        pass

    return info


# ============================================================================
# Synthetic Test Data
# ============================================================================

def generate_test_pages(num_pages: int) -> List[Image.Image]:
    """Generate synthetic document-like images for benchmarking."""
    pages = []
    for i in range(num_pages):
        # Create A4-like image at 200 DPI (1654 x 2339 px)
        img = Image.new("RGB", (1654, 2339), color=(255, 255, 255))
        # Draw some synthetic text-like patterns
        arr = np.array(img)
        for y in range(100, 2200, 40):
            # Simulate text lines with random widths
            line_width = np.random.randint(800, 1500)
            arr[y:y+12, 100:100+line_width] = np.random.randint(0, 50, (12, line_width, 3))
        pages.append(Image.fromarray(arr))
    return pages


# ============================================================================
# Benchmark Runner
# ============================================================================

def benchmark_ocr(pages: List[Image.Image]) -> StageMetrics:
    """Benchmark OCR stage."""
    metrics = StageMetrics(stage="ocr")

    try:
        from modules.ocr_manager import OCRManager
        ocr = OCRManager()
    except Exception as e:
        logger.error("Failed to init OCR manager: %s", e)
        metrics.errors = len(pages)
        return metrics

    for page in pages:
        gc.collect()
        t0 = time.perf_counter()
        try:
            text, conf = ocr._run_primary_engine(page)
            elapsed = time.perf_counter() - t0
            metrics.total_time_s += elapsed
            metrics.min_time_s = min(metrics.min_time_s, elapsed)
            metrics.max_time_s = max(metrics.max_time_s, elapsed)
            metrics.pages_processed += 1
            if text.strip():
                metrics.items_produced += 1
        except Exception as e:
            logger.warning("OCR error: %s", e)
            metrics.errors += 1

    metrics.finalize()
    return metrics


def benchmark_chunking(num_blocks: int) -> StageMetrics:
    """Benchmark chunking stage with synthetic blocks."""
    from pipeline.chunking_step import ChunkingStep
    metrics = StageMetrics(stage="chunking")
    chunker = ChunkingStep(max_tokens=256, overlap_sentences=2)

    # Generate synthetic blocks
    blocks = [
        {
            "text": f"Este es un bloque de texto ficticio número {i}. " * 50,
            "block_type": "text",
            "page_number": (i // 10) + 1,
            "block_id": f"block_{i}",
        }
        for i in range(num_blocks)
    ]

    t0 = time.perf_counter()
    try:
        chunks = chunker.process(
            document_id="benchmark-doc",
            tenant_id="benchmark-tenant",
            hotel_id=None,
            blocks=blocks,
            db=None,
        )
        metrics.total_time_s = time.perf_counter() - t0
        metrics.pages_processed = num_blocks
        metrics.items_produced = len(chunks)
    except Exception as e:
        logger.error("Chunking error: %s", e)
        metrics.errors = 1

    metrics.finalize()
    return metrics


def benchmark_embedding(num_texts: int) -> StageMetrics:
    """Benchmark embedding generation."""
    metrics = StageMetrics(stage="embedding")

    try:
        from pipeline.embedding_step import EmbeddingStep
        emb = EmbeddingStep(db=None)
        # Force model load
        _ = emb.model
    except Exception as e:
        logger.error("Failed to init embedding model: %s", e)
        metrics.errors = num_texts
        return metrics

    texts = [f"Texto de ejemplo para benchmarking del modelo de embeddings número {i}." for i in range(num_texts)]

    t0 = time.perf_counter()
    try:
        embeddings = emb._encode_batched(texts)
        metrics.total_time_s = time.perf_counter() - t0
        metrics.pages_processed = num_texts
        metrics.items_produced = len(embeddings)
    except Exception as e:
        logger.error("Embedding error: %s", e)
        metrics.errors = num_texts

    metrics.finalize()
    return metrics


def run_benchmark(num_pages: int = 20, stage: str = "all") -> BenchmarkResult:
    """Run the complete benchmark suite."""
    result = BenchmarkResult()
    result.system_info = get_system_info()
    result.total_pages = num_pages

    logger.info("=" * 60)
    logger.info("AutoOCR Document AI Pipeline — Benchmark")
    logger.info("=" * 60)
    logger.info("Pages: %d | Stage: %s", num_pages, stage)
    logger.info("System: %s", json.dumps(result.system_info, indent=2))
    logger.info("")

    t_start = time.perf_counter()

    if stage in ("all", "ocr"):
        logger.info("📝 Benchmarking OCR (%d pages)...", num_pages)
        pages = generate_test_pages(num_pages)
        result.stages["ocr"] = benchmark_ocr(pages)
        logger.info("   ✅ OCR: %.1f pág/min (avg %.2fs/pág)",
                     result.stages["ocr"].throughput_pages_per_min,
                     result.stages["ocr"].avg_time_per_page_s)

    if stage in ("all", "chunking"):
        logger.info("📦 Benchmarking Chunking (%d blocks)...", num_pages * 10)
        result.stages["chunking"] = benchmark_chunking(num_pages * 10)
        logger.info("   ✅ Chunking: %d chunks from %d blocks in %.2fs",
                     result.stages["chunking"].items_produced,
                     result.stages["chunking"].pages_processed,
                     result.stages["chunking"].total_time_s)

    if stage in ("all", "embedding"):
        logger.info("🔢 Benchmarking Embeddings (%d texts)...", num_pages * 5)
        result.stages["embedding"] = benchmark_embedding(num_pages * 5)
        logger.info("   ✅ Embeddings: %d vectors in %.2fs (%.0f vec/s)",
                     result.stages["embedding"].items_produced,
                     result.stages["embedding"].total_time_s,
                     result.stages["embedding"].items_produced / max(result.stages["embedding"].total_time_s, 0.001))

    result.total_time_s = time.perf_counter() - t_start
    if result.total_time_s > 0:
        result.throughput_e2e_pages_per_min = (num_pages / result.total_time_s) * 60

    logger.info("")
    logger.info("=" * 60)
    logger.info("Total benchmark time: %.1fs", result.total_time_s)
    logger.info("End-to-end throughput: %.1f pages/min", result.throughput_e2e_pages_per_min)
    logger.info("=" * 60)

    return result


# ============================================================================
# Cost Estimation
# ============================================================================

def estimate_monthly_costs(
    docs_per_day: int = 100,
    avg_pages_per_doc: int = 5,
    ocr_time_per_page_s: float = 2.0,
    enable_vl: bool = False,
    enable_rag: bool = True,
) -> Dict[str, Any]:
    """
    Estimate monthly infrastructure costs.

    Assumes:
    - Docker on cloud VM (GPU instance)
    - PostgreSQL managed service
    - Redis managed service
    """
    total_pages_month = docs_per_day * avg_pages_per_doc * 30
    gpu_hours_month = (total_pages_month * ocr_time_per_page_s) / 3600

    # VL adds ~3x processing time per figure asset (~20% of pages have figures)
    if enable_vl:
        vl_pages = int(total_pages_month * 0.2)
        gpu_hours_month += (vl_pages * ocr_time_per_page_s * 3) / 3600

    # Embedding time (much faster than OCR)
    if enable_rag:
        chunks_month = total_pages_month * 3  # ~3 chunks per page
        embedding_hours = (chunks_month * 0.01) / 3600  # ~10ms per chunk
        gpu_hours_month += embedding_hours

    scenarios = {
        "small": {
            "label": "Pequeño (1 hotel, 50 docs/día)",
            "docs_per_day": 50,
            "gpu_instance": "T4 (g4dn.xlarge)",
            "gpu_cost_h": 0.526,  # AWS g4dn.xlarge on-demand
            "pg_cost_month": 50,  # RDS db.t3.medium
            "redis_cost_month": 15,  # ElastiCache t3.micro
            "storage_gb_month": 50,
            "storage_cost_gb": 0.023,
        },
        "medium": {
            "label": "Medio (5 hoteles, 200 docs/día)",
            "docs_per_day": 200,
            "gpu_instance": "T4 (g4dn.xlarge)",
            "gpu_cost_h": 0.526,
            "pg_cost_month": 120,  # RDS db.r5.large
            "redis_cost_month": 30,
            "storage_gb_month": 200,
            "storage_cost_gb": 0.023,
        },
        "large": {
            "label": "Grande (20+ hoteles, 500 docs/día)",
            "docs_per_day": 500,
            "gpu_instance": "A10G (g5.xlarge)",
            "gpu_cost_h": 1.006,  # AWS g5.xlarge
            "pg_cost_month": 300,  # RDS db.r5.xlarge
            "redis_cost_month": 50,
            "storage_gb_month": 1000,
            "storage_cost_gb": 0.023,
        },
    }

    results = {}
    for tier, s in scenarios.items():
        tier_pages = s["docs_per_day"] * avg_pages_per_doc * 30
        tier_gpu_h = (tier_pages * ocr_time_per_page_s) / 3600

        # Assume 12h/day active processing average
        gpu_monthly = s["gpu_cost_h"] * 12 * 30
        storage_monthly = s["storage_gb_month"] * s["storage_cost_gb"]

        total = gpu_monthly + s["pg_cost_month"] + s["redis_cost_month"] + storage_monthly

        results[tier] = {
            "label": s["label"],
            "docs_month": s["docs_per_day"] * 30,
            "pages_month": tier_pages,
            "gpu_instance": s["gpu_instance"],
            "gpu_hours_needed": round(tier_gpu_h, 1),
            "costs": {
                "gpu_compute": round(gpu_monthly, 2),
                "postgresql": s["pg_cost_month"],
                "redis": s["redis_cost_month"],
                "storage": round(storage_monthly, 2),
                "total_monthly_usd": round(total, 2),
            },
            "cost_per_doc": round(total / (s["docs_per_day"] * 30), 3),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="AutoOCR Pipeline Benchmark")
    parser.add_argument("--pages", type=int, default=20, help="Number of test pages")
    parser.add_argument("--stage", choices=["all", "ocr", "chunking", "embedding"], default="all")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--costs", action="store_true", help="Show cost estimation")
    args = parser.parse_args()

    if args.costs:
        costs = estimate_monthly_costs()
        print("\n" + "=" * 60)
        print("💰 Estimación de Costes Mensuales (AWS)")
        print("=" * 60)
        for tier, data in costs.items():
            print(f"\n📊 {data['label']}")
            print(f"   Documentos/mes: {data['docs_month']:,}")
            print(f"   Páginas/mes: {data['pages_month']:,}")
            print(f"   GPU: {data['gpu_instance']}")
            print(f"   Horas GPU necesarias: {data['gpu_hours_needed']}h")
            print(f"   Coste GPU:    ${data['costs']['gpu_compute']:.2f}")
            print(f"   PostgreSQL:   ${data['costs']['postgresql']:.2f}")
            print(f"   Redis:        ${data['costs']['redis']:.2f}")
            print(f"   Storage:      ${data['costs']['storage']:.2f}")
            print(f"   ─────────────────────────")
            print(f"   TOTAL/mes:    ${data['costs']['total_monthly_usd']:.2f}")
            print(f"   Coste/doc:    ${data['cost_per_doc']:.3f}")
        return

    result = run_benchmark(num_pages=args.pages, stage=args.stage)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
