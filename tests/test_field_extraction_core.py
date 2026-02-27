from __future__ import annotations

from modules.normalizer import DataNormalizer
from modules.smart_extractor import FieldExtractor


def test_field_extractor_extracts_core_fields():
    text = """
    FACTURA
    MUEBLES ACME S.L.
    Fecha: 02/01/2026
    Base imponible: 100,00 EUR
    IVA: 21,00 EUR
    TOTAL A PAGAR: 121,00 EUR
    """.strip()

    extractor = FieldExtractor(config={})
    extractor.llm_client = None
    fields = extractor.extract_fields(text)

    assert fields.get("date", {}).get("value") == "2026-01-02"
    assert float(fields.get("total", {}).get("value")) == 121.0
    assert fields.get("vendor", {}).get("value")
    assert fields.get("supplier", {}).get("value")
    assert fields.get("supplier", {}).get("value") == fields.get("vendor", {}).get("value")


def test_field_extractor_prioritizes_total_keyword_over_larger_non_total_amount():
    text = """
    FACTURA DEMO
    PROVEEDOR DEMO
    Subtotal: 999,99 EUR
    TOTAL: 120,00 EUR
    """.strip()

    extractor = FieldExtractor(config={})
    extractor.llm_client = None
    fields = extractor.extract_fields(text)

    assert float(fields.get("total", {}).get("value")) == 120.0


def test_data_normalizer_preserves_metadata_and_supplier_alias():
    fields = {
        "date": {"value": "2026-01-02", "confidence": 0.8},
        "total": {"value": "1.234,56", "confidence": 0.7},
        "vendor": {"value": "MUEBLES ACME S.L.", "confidence": 0.9},
        "validation_status": "valid",
    }

    normalizer = DataNormalizer(config={})
    normalized = normalizer.normalize(fields)

    assert normalized["validation_status"] == "valid"
    assert normalized["date"]["value"] == "2026-01-02"
    assert normalized["total"]["value"] == 1234.56
    assert normalized["vendor"]["value"] == "Muebles Acme"
    assert normalized["supplier"]["value"] == "Muebles Acme"


def test_data_normalizer_syncs_supplier_to_vendor_when_only_supplier_exists():
    fields = {
        "supplier": {"value": "PROVEEDOR X S.L.", "confidence": 0.6},
    }

    normalizer = DataNormalizer(config={})
    normalized = normalizer.normalize(fields)

    assert "vendor" in normalized
    assert normalized["vendor"]["value"] == normalized["supplier"]["value"]
