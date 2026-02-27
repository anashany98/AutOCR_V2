from __future__ import annotations

import json

from modules.field_quality import evaluate_field_quality
from scripts import field_quality_baseline as baseline_script


def test_evaluate_field_quality_perfect_dataset():
    dataset = [
        {
            "id": "a1",
            "text": "FACTURA\nACME S.L.\nFecha: 2026-01-02\nTOTAL 121,00 EUR",
            "expected": {"date": "2026-01-02", "total": 121.0, "supplier": "Acme"},
        },
        {
            "id": "a2",
            "text": "INVOICE\nGLOBAL LIGHTING LTD\nDate 15/03/2026\nGrand Total: $653.40",
            "expected": {"date": "2026-03-15", "total": 653.40, "supplier": "Global Lighting"},
        },
    ]

    res = evaluate_field_quality(dataset)
    assert res["samples"] == 2
    assert res["overall"]["all_fields_exact"] == 2
    assert res["overall"]["all_fields_exact_rate"] == 1.0
    assert res["field_metrics"]["date"]["f1"] == 1.0
    assert res["field_metrics"]["total"]["f1"] == 1.0
    assert res["field_metrics"]["supplier"]["f1"] == 1.0
    assert res["failures"] == []


def test_evaluate_field_quality_records_failures():
    dataset = [
        {
            "id": "bad-1",
            "text": "Factura\nProveedor X\nFecha: 2026-01-01\nTOTAL 10,00 EUR",
            "expected": {"date": "2026-01-02", "total": 11.0, "supplier": "Proveedor Y"},
        }
    ]

    res = evaluate_field_quality(dataset, max_failures=10)
    assert res["samples"] == 1
    assert res["overall"]["all_fields_exact"] == 0
    assert res["overall"]["all_fields_exact_rate"] == 0.0
    assert len(res["failures"]) == 1
    assert res["failures"][0]["id"] == "bad-1"


def _stub_results(
    *,
    date_f1: float,
    total_f1: float,
    supplier_f1: float,
    exact_rate: float,
) -> dict:
    return {
        "samples": 3,
        "overall": {
            "all_fields_exact": int(round(3 * exact_rate)),
            "all_fields_exact_rate": exact_rate,
        },
        "field_metrics": {
            "date": {"precision": date_f1, "recall": date_f1, "f1": date_f1},
            "total": {"precision": total_f1, "recall": total_f1, "f1": total_f1},
            "supplier": {"precision": supplier_f1, "recall": supplier_f1, "f1": supplier_f1},
        },
        "failures": [],
    }


def test_field_quality_baseline_cli_no_gate_does_not_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(baseline_script, "load_reference_dataset", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        baseline_script,
        "evaluate_field_quality",
        lambda *_args, **_kwargs: _stub_results(date_f1=0.40, total_f1=0.35, supplier_f1=0.30, exact_rate=0.25),
    )

    output = tmp_path / "report.json"
    exit_code = baseline_script.main(["--dataset", "ignored.jsonl", "--output", str(output)])

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["config"]["gate_enabled"] is False


def test_field_quality_baseline_cli_gate_defaults_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(baseline_script, "load_reference_dataset", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        baseline_script,
        "evaluate_field_quality",
        lambda *_args, **_kwargs: _stub_results(date_f1=0.97, total_f1=0.99, supplier_f1=0.99, exact_rate=0.96),
    )

    output = tmp_path / "report.json"
    exit_code = baseline_script.main(["--dataset", "ignored.jsonl", "--output", str(output), "--gate"])

    assert exit_code == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["config"]["gate_enabled"] is True


def test_field_quality_baseline_cli_explicit_thresholds_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(baseline_script, "load_reference_dataset", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        baseline_script,
        "evaluate_field_quality",
        lambda *_args, **_kwargs: _stub_results(date_f1=0.80, total_f1=0.81, supplier_f1=0.82, exact_rate=0.79),
    )

    output = tmp_path / "report.json"
    exit_code = baseline_script.main(
        [
            "--dataset",
            "ignored.jsonl",
            "--output",
            str(output),
            "--min-date-f1",
            "0.75",
            "--min-total-f1",
            "0.75",
            "--min-supplier-f1",
            "0.75",
            "--min-all-fields-exact-rate",
            "0.75",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["config"]["gate_enabled"] is True
