"""
AutOCR Document AI Pipeline — Package initializer.

Each step in the pipeline is a self-contained module that can be invoked
independently or chained together via the :class:`PipelineOrchestrator`.
"""

from __future__ import annotations

from pipeline.ingestion import IngestionStep
from pipeline.ocr_step import OCRStep
from pipeline.layout_step import LayoutStep
from pipeline.visual_step import VisualStep
from pipeline.chunking_step import ChunkingStep
from pipeline.embedding_step import EmbeddingStep
from pipeline.job_manager import JobManager
from pipeline.orchestrator import PipelineOrchestrator

__all__ = [
    "PipelineOrchestrator",
    "IngestionStep",
    "OCRStep",
    "LayoutStep",
    "VisualStep",
    "ChunkingStep",
    "EmbeddingStep",
    "JobManager",
]
