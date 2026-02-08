#!/usr/bin/env python3
"""
Daily Ingest Script for AutoOCR.

This script is designed to be run as a cron job (e.g., nightly) or manually.
It scans a specified directory for document files and enqueues them for processing
if they haven't been processed yet.

Usage:
    python scripts/daily_ingest.py --input /path/to/docs --recursive
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path ensuring modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "daily_ingest.log"),
    ],
)
logger = logging.getLogger("daily_ingest")

def main():
    parser = argparse.ArgumentParser(description="AutoOCR Daily Ingest")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input directory to scan")
    parser.add_argument("--recursive", "-r", action="store_true", help="Scan recursively")
    parser.add_argument("--extensions", "-e", nargs="+", default=["pdf", "jpg", "jpeg", "png", "tif", "tiff"], help="Extensions to process")
    parser.add_argument("--batch-size", "-b", type=int, default=100, help="Max files to enqueue per run") 
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Import task after path setup
    try:
        from modules.tasks import process_document_task
    except ImportError as e:
        logger.error(f"Failed to import AutoOCR modules: {e}")
        sys.exit(1)

    logger.info(f"Starting ingest scan on {input_dir} (Recursive: {args.recursive})")
    
    extensions = {f".{ext.lower().lstrip('.')}" for ext in args.extensions}
    count = 0
    enqueued = 0
    
    iterator = input_dir.rglob("*") if args.recursive else input_dir.glob("*")
    
    for file_path in iterator:
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            count += 1
            
            # TODO: Add logic to check if already processed
            # For now, we assume this script moves files or we rely on deduplication in the pipeline
            # A simple check logic could be looking for a .processed marker or DB check
            
            # Simple Marker Check
            marker_file = file_path.with_suffix(file_path.suffix + ".processed")
            if marker_file.exists():
                continue
                
            try:
                logger.info(f"Enqueueing: {file_path.name}")
                
                options = {
                    "delete_original": False,  # Safety first in batch
                    "ocr_enabled": True,
                    "classification_enabled": True,
                    "input_root": str(input_dir),
                    "batch_ingest": True
                }
                
                process_document_task(str(file_path), options)
                
                # Mark as enqueued/processed to prevent re-queueing immediately
                # In production, maybe move to a 'processing' folder instead
                # marker_file.touch() 
                
                enqueued += 1
                if enqueued >= args.batch_size:
                    logger.info(f"Batch limit reached ({args.batch_size}). Stopping.")
                    break
                    
            except Exception as e:
                logger.error(f"Failed to enqueue {file_path}: {e}")

    logger.info(f"Scan complete. Found {count} files. Enqueued {enqueued} new tasks.")

if __name__ == "__main__":
    main()
