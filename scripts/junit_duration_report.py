"""
Build per-file and per-test duration reports from a pytest JUnit XML file.

Usage:
  python scripts/junit_duration_report.py --junitxml tmp/junit.xml \
      --output-json tmp/durations.json --output-md tmp/durations.md --top 20
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def _classname_to_file(classname: str) -> str:
    parts = classname.split(".")
    if len(parts) >= 2 and parts[0] == "tests":
        return f"tests/{parts[1]}.py"
    return classname or "unknown"


def build_report(junitxml_path: Path, top: int = 20) -> dict:
    root = ET.parse(junitxml_path).getroot()
    per_file = defaultdict(float)
    per_test = []

    for testcase in root.findall(".//testcase"):
        classname = testcase.get("classname", "")
        test_name = testcase.get("name", "")
        seconds = float(testcase.get("time", "0") or 0)
        file_path = _classname_to_file(classname)
        nodeid = f"{classname}::{test_name}" if classname else test_name

        per_file[file_path] += seconds
        per_test.append({"nodeid": nodeid, "file": file_path, "seconds": seconds})

    files_sorted = sorted(
        ({"file": path, "seconds": secs} for path, secs in per_file.items()),
        key=lambda row: row["seconds"],
        reverse=True,
    )
    tests_sorted = sorted(per_test, key=lambda row: row["seconds"], reverse=True)

    return {
        "source": str(junitxml_path),
        "total_seconds": round(sum(per_file.values()), 6),
        "top": int(top),
        "files": [{"file": row["file"], "seconds": round(row["seconds"], 6)} for row in files_sorted],
        "tests_top": [
            {"nodeid": row["nodeid"], "file": row["file"], "seconds": round(row["seconds"], 6)}
            for row in tests_sorted[:top]
        ],
    }


def to_markdown(report: dict) -> str:
    lines = [
        "# Test Duration Report",
        "",
        f"- Source: `{report['source']}`",
        f"- Total test time (sum of testcase times): `{report['total_seconds']:.3f}s`",
        "",
        "## By File",
        "",
        "| File | Seconds |",
        "|---|---:|",
    ]
    for row in report["files"]:
        lines.append(f"| `{row['file']}` | {row['seconds']:.3f} |")

    lines.extend(
        [
            "",
            f"## Top {report['top']} Slow Tests",
            "",
            "| Test | File | Seconds |",
            "|---|---|---:|",
        ]
    )
    for row in report["tests_top"]:
        lines.append(f"| `{row['nodeid']}` | `{row['file']}` | {row['seconds']:.3f} |")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate duration summaries from pytest JUnit XML")
    parser.add_argument("--junitxml", required=True, help="Path to JUnit XML file")
    parser.add_argument("--output-json", required=True, help="Output path for JSON report")
    parser.add_argument("--output-md", required=True, help="Output path for Markdown report")
    parser.add_argument("--top", type=int, default=20, help="How many slow tests to include")
    args = parser.parse_args()

    junit_path = Path(args.junitxml)
    if not junit_path.exists():
        raise FileNotFoundError(f"JUnit XML file not found: {junit_path}")

    report = build_report(junit_path, top=max(1, int(args.top)))

    json_path = Path(args.output_json)
    md_path = Path(args.output_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")

    print(f"Wrote JSON report: {json_path}")
    print(f"Wrote Markdown report: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
