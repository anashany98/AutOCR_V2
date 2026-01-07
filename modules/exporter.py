import csv
import os
import logging
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def export_to_csv(documents: List[Dict[str, Any]], output_path: str) -> str:
    """
    Exports a list of document metadata to a CSV file.
    
    Args:
        documents: List of document dictionaries (from DB).
        output_path: Path where to save the CSV.
        
    Returns:
        Absolute path to the generated CSV.
    """
    if not documents:
        return ""

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Determine headers from the first document's keys
    # But we want to be specific about what we export
    headers = [
        "id", "filename", "type", "status", "workflow_state", 
        "confidence", "datetime", "path"
    ]

    try:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            for doc in documents:
                writer.writerow(doc)
        
        return os.path.abspath(output_path)
    except Exception as e:
        logger.error(f"Failed to export to CSV: {e}")
        return ""
