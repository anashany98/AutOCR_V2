"""
Field quality baseline runner.

Usage:
  python scripts/field_quality_baseline.py --dataset data/benchmarks/field_extraction_reference.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.field_quality import evaluate_field_quality, load_reference_dataset  # noqa: E402


DEFAULT_GATE_THRESHOLDS = {
    "date_f1": 0.98,
    "total_f1": 0.98,
    "supplier_f1": 0.98,
    "all_fields_exact_rate": 0.95,
}


def _resolve_gate_thresholds(args: argparse.Namespace) -> dict[str, float | None]:
    thresholds: dict[str, float | None] = {
        "date_f1": args.min_date_f1,
        "total_f1": args.min_total_f1,
        "supplier_f1": args.min_supplier_f1,
        "all_fields_exact_rate": args.min_all_fields_exact_rate,
    }
    if args.gate:
        for key, default_value in DEFAULT_GATE_THRESHOLDS.items():
            if thresholds[key] is None:
                thresholds[key] = default_value
    return thresholds


def _gate_enabled(thresholds: dict[str, float | None]) -> bool:
    return any(value is not None for value in thresholds.values())


def _evaluate_gate_failures(results: dict[str, object], thresholds: dict[str, float | None]) -> list[str]:
    metrics = {
        "date_f1": float(((results.get("field_metrics", {}) or {}).get("date", {}) or {}).get("f1", 0.0)),
        "total_f1": float(((results.get("field_metrics", {}) or {}).get("total", {}) or {}).get("f1", 0.0)),
        "supplier_f1": float(((results.get("field_metrics", {}) or {}).get("supplier", {}) or {}).get("f1", 0.0)),
        "all_fields_exact_rate": float(((results.get("overall", {}) or {}).get("all_fields_exact_rate", 0.0))),
    }
    labels = {
        "date_f1": "date.f1",
        "total_f1": "total.f1",
        "supplier_f1": "supplier.f1",
        "all_fields_exact_rate": "overall.all_fields_exact_rate",
    }

    failures: list[str] = []
    for key, threshold in thresholds.items():
        if threshold is None:
            continue
        actual = metrics[key]
        if actual < threshold:
            failures.append(f"{labels[key]}={actual:.4f} is below min={threshold:.4f}")
    return failures


def _format_threshold(value: float | None) -> str:
    return "off" if value is None else f"{value:.2%}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate extraction quality baseline on a reference dataset")
    parser.add_argument(
        "--dataset",
        default=str(PROJECT_ROOT / "data" / "benchmarks" / "field_extraction_reference.jsonl"),
        help="Path to reference dataset (.jsonl or .json)",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "reports" / "field_quality_baseline.json"),
        help="Path to JSON report output",
    )
    parser.add_argument("--abs-tol-total", type=float, default=0.01, help="Absolute tolerance for total amount match")
    parser.add_argument("--rel-tol-total", type=float, default=0.001, help="Relative tolerance for total amount match")
    parser.add_argument("--max-failures", type=int, default=25, help="Max failing samples to include in report")
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Enable quality gate with default thresholds (can be overridden by --min-* args)",
    )
    parser.add_argument("--min-date-f1", type=float, default=None, help="Minimum accepted F1 score for date extraction")
    parser.add_argument("--min-total-f1", type=float, default=None, help="Minimum accepted F1 score for total extraction")
    parser.add_argument(
        "--min-supplier-f1",
        type=float,
        default=None,
        help="Minimum accepted F1 score for supplier extraction",
    )
    parser.add_argument(
        "--min-all-fields-exact-rate",
        type=float,
        default=None,
        help="Minimum accepted exact-match rate across all core fields",
    )

    args = parser.parse_args(argv)
    thresholds = _resolve_gate_thresholds(args)
    for key, value in thresholds.items():
        if value is None:
            continue
        if not 0.0 <= value <= 1.0:
            parser.error(f"{key} must be between 0 and 1 (received: {value})")
    gate_enabled = _gate_enabled(thresholds)

    dataset = load_reference_dataset(args.dataset)
    results = evaluate_field_quality(
        dataset,
        abs_tol_total=float(args.abs_tol_total),
        rel_tol_total=float(args.rel_tol_total),
        max_failures=int(args.max_failures),
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(Path(args.dataset).resolve()),
        "config": {
            "abs_tol_total": float(args.abs_tol_total),
            "rel_tol_total": float(args.rel_tol_total),
            "max_failures": int(args.max_failures),
            "gate_enabled": gate_enabled,
            "gate_thresholds": thresholds,
        },
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fm = results["field_metrics"]
    print(f"Samples: {results['samples']}")
    print(
        "Exact all-fields rate: "
        f"{results['overall']['all_fields_exact_rate']:.2%} "
        f"({results['overall']['all_fields_exact']}/{results['samples']})"
    )
    print(
        "Date  -> P:{:.2%} R:{:.2%} F1:{:.2%}".format(
            fm["date"]["precision"], fm["date"]["recall"], fm["date"]["f1"]
        )
    )
    print(
        "Total -> P:{:.2%} R:{:.2%} F1:{:.2%}".format(
            fm["total"]["precision"], fm["total"]["recall"], fm["total"]["f1"]
        )
    )
    print(
        "Supplier -> P:{:.2%} R:{:.2%} F1:{:.2%}".format(
            fm["supplier"]["precision"], fm["supplier"]["recall"], fm["supplier"]["f1"]
        )
    )
    print(f"Report saved to: {output_path.resolve()}")

    if gate_enabled:
        failures = _evaluate_gate_failures(results, thresholds)
        print("Quality gate: enabled")
        print(
            "Thresholds -> "
            f"date.f1>={_format_threshold(thresholds['date_f1'])}, "
            f"total.f1>={_format_threshold(thresholds['total_f1'])}, "
            f"supplier.f1>={_format_threshold(thresholds['supplier_f1'])}, "
            f"overall.all_fields_exact_rate>={_format_threshold(thresholds['all_fields_exact_rate'])}"
        )
        if failures:
            print("Quality gate: FAILED")
            for item in failures:
                print(f"- {item}")
            return 1
        print("Quality gate: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
