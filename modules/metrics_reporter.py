"""
Metrics reporter module.

Once a batch of documents has been processed the AutOCR system can
generate both CSV and PDF summary reports.  Each record in the report
captures the file name, final status (``OK`` or ``FAILED``), processing
duration and assigned document type.  Aggregated metrics (total OK and
FAILED counts, average time and reliability percentage) are also
included at the bottom of the report.

CSV reports are always generated when enabled in configuration; PDF
generation is optional and requires the ``reportlab`` package.
"""

from __future__ import annotations

import datetime
import os
from typing import Iterable, List, Mapping

import pandas as pd

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
except ImportError:
    canvas = None  # type: ignore


def _write_pdf(
    records: List[Mapping[str, object]],
    metrics: Mapping[str, object],
    pdf_path: str,
) -> None:
    """
    Write a simple PDF summary using reportlab.  The table will be split
    across pages if it exceeds the available vertical space.
    """
    if canvas is None:
        raise RuntimeError("reportlab is not installed; cannot generate PDF reports")
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    margin = 20 * mm
    x = margin
    y = height - margin

    # Header
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, "AutOCR Batch Summary")
    c.setFont("Helvetica", 10)
    y -= 10
    c.drawString(x, y, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 20

    # Column headers
    headers = ["Filename", "Status", "Duration (s)", "Type"]
    col_widths = [70 * mm, 30 * mm, 30 * mm, 40 * mm]
    c.setFont("Helvetica-Bold", 9)
    for i, header in enumerate(headers):
        c.drawString(x + sum(col_widths[:i]), y, header)
    y -= 12

    c.setFont("Helvetica", 9)
    # Rows
    for record in records:
        row = [
            str(record.get("filename", ""))[:40],
            str(record.get("status", "")),
            f"{record.get('duration', 0):.2f}",
            str(record.get("type", ""))[:30],
        ]
        if y < margin + 40:
            c.showPage()
            y = height - margin
            # repeat header
            c.setFont("Helvetica-Bold", 9)
            for i, header in enumerate(headers):
                c.drawString(x + sum(col_widths[:i]), y, header)
            y -= 12
            c.setFont("Helvetica", 9)
        for i, cell in enumerate(row):
            c.drawString(x + sum(col_widths[:i]), y, cell)
        y -= 12

    # Aggregated metrics
    y -= 10
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x, y, "Summary")
    y -= 12
    c.setFont("Helvetica", 9)
    summary_lines = [
        f"Total OK documents: {metrics.get('ok_docs', 0)}",
        f"Total failed documents: {metrics.get('failed_docs', 0)}",
        f"Average processing time (s): {metrics.get('avg_time', 0):.2f}",
        f"Reliability (%): {metrics.get('reliability_pct', 0):.2f}",
    ]
    for line in summary_lines:
        if y < margin + 20:
            c.showPage()
            y = height - margin
            c.setFont("Helvetica-Bold", 10)
            c.drawString(x, y, "Summary")
            y -= 12
            c.setFont("Helvetica", 9)
        c.drawString(x, y, line)
        y -= 12

    c.save()


def generate_summary_report(
    records: Iterable[Mapping[str, object]],
    report_folder: str,
    prefix: Optional[str] = None,
    include_pdf: bool = True,
    metrics: Optional[Mapping[str, object]] = None,
) -> None:
    """
    Generate CSV (and optionally PDF) summary reports for the processed batch.

    Parameters
    ----------
    records:
        Iterable of dictionaries containing at least the keys ``filename``,
        ``status``, ``duration`` and ``type``.
    report_folder:
        Folder where the reports should be saved.  Will be created if missing.
    prefix:
        Optional prefix for the file names.  If omitted the current timestamp
        will be used.
    include_pdf:
        When True and reportlab is installed a PDF will be generated in
        addition to the CSV.
    metrics:
        Aggregated statistics used for the summary section of the PDF.  When
        omitted default values of zero are used.
    """
    os.makedirs(report_folder, exist_ok=True)
    ts = prefix or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(report_folder, f"{ts}_summary.csv")
    records_list = list(records)
    df = pd.DataFrame(records_list)
    df.to_csv(csv_path, index=False)

    # Prepare metrics for PDF
    metrics_data = metrics or {
        "ok_docs": 0,
        "failed_docs": 0,
        "avg_time": 0.0,
        "reliability_pct": 0.0,
    }

    if include_pdf and canvas is not None:
        pdf_path = os.path.join(report_folder, f"{ts}_summary.pdf")
        _write_pdf(records_list, metrics_data, pdf_path)