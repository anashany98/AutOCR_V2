"""Generate demo samples and run the AutOCR pipeline."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageDraw
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

from postbatch_processor import main as postbatch_main

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
SAMPLES_DIR = PROJECT_ROOT / "samples"
INPUT_DIR = SAMPLES_DIR / "demo_input"
PROCESSED_DIR = SAMPLES_DIR / "demo_processed"
FAILED_DIR = SAMPLES_DIR / "demo_failed"
REPORTS_DIR = SAMPLES_DIR / "demo_reports"
TEMP_CONFIG = SAMPLES_DIR / "demo_config.yaml"


def generate_pdf(path: Path) -> None:
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    data = [["Producto", "Cantidad", "Precio"], ["A", "2", "10.00"], ["B", "5", "4.50"], ["Total", "", "32.50"]]
    table = Table(data)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    doc.build([table])


def generate_invoice_image(path: Path) -> None:
    image = Image.new("RGB", (600, 400), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 20, 580, 380], outline="black", width=3)
    draw.text((40, 40), "Factura 2024-001", fill="black")
    draw.text((40, 90), "Cliente: Ejemplo S.A.", fill="black")
    draw.text((40, 140), "Importe: 123,45 EUR", fill="black")
    draw.text((40, 190), "Fecha: 2024-01-15", fill="black")
    draw.text((40, 260), "Gracias por su compra", fill="black")
    image.save(path)


def prepare_directories() -> None:
    for folder in (INPUT_DIR, PROCESSED_DIR, FAILED_DIR, REPORTS_DIR):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)


def write_temp_config() -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle) or {}
    base_config.setdefault("postbatch", {})
    base_config["postbatch"].update(
        {
            "input_folder": str(INPUT_DIR),
            "processed_folder": str(PROCESSED_DIR),
            "failed_folder": str(FAILED_DIR),
            "reports_folder": str(REPORTS_DIR),
            "delete_original": True,
            "inactivity_trigger_minutes": 0,
            "max_workers": 1,
        }
    )
    with open(TEMP_CONFIG, "w", encoding="utf-8") as handle:
        yaml.safe_dump(base_config, handle, sort_keys=False, allow_unicode=True)


def run_demo() -> int:
    prepare_directories()
    generate_pdf(INPUT_DIR / "sample_table.pdf")
    generate_invoice_image(INPUT_DIR / "invoice.png")
    write_temp_config()
    print("Running demo pipeline...", flush=True)
    result = postbatch_main(["--config", str(TEMP_CONFIG), "--immediate"])
    if result == 0:
        print("Demo completed. Processed files saved in:")
        print(f"  {PROCESSED_DIR}")
    else:
        print("Demo finished with errors. Check:")
        print(f"  {REPORTS_DIR}")
    return result


if __name__ == "__main__":
    sys.exit(run_demo())
