"""Tests for FusionManager heuristics."""

from modules.fusion_manager import FusionConfig, FusionManager


def test_primary_wins_with_high_confidence():
    manager = FusionManager(FusionConfig(min_confidence_primary=0.6))
    text, confidence = manager.fuse("primary", 0.8, "secondary", 0.9, None)
    assert text == "primary"
    assert confidence == 0.8


def test_secondary_used_when_confidence_higher():
    manager = FusionManager(FusionConfig(min_confidence_primary=0.6, confidence_margin=0.05))
    text, confidence = manager.fuse("primary", 0.4, "secondary", 0.7, None)
    assert text == "secondary"
    assert confidence == 0.7


def test_heuristic_prefers_numeric_text():
    manager = FusionManager(FusionConfig(min_confidence_primary=0.6))
    text, confidence = manager.fuse("subtotal", 0.55, "Factura 2024-01", 0.56, None)
    assert text == "Factura 2024-01"
    assert confidence == 0.56
