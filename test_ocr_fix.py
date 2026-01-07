import logging
import os
from pathlib import Path
import yaml
from modules.db_manager import DBManager
from postbatch_processor import PipelineComponents, initialise_pipeline, process_single_file
from modules.classifier import DocumentClassifier

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OCR-FIX")

def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    project_root = str(Path(".").absolute())
    db = DBManager(config)
    pipeline = initialise_pipeline(config, project_root, logger)
    classifier = DocumentClassifier() if config.get("postbatch", {}).get("classification_enabled", True) else None
    
    # Target file
    # Based on previous list_dir, it's now in 'errors' folder
    target_filename = "2026-01-05_Contract_NoProject_NoVendor_10001-15000_11329S.pdf"
    target_path = os.path.join(project_root, "errors", target_filename)
    
    if not os.path.exists(target_path):
        print(f"File not found at {target_path}")
        return

    print(f"Re-processing {target_filename}...")
    
    # We need to manually call process_single_file
    # Note: process_single_file inserts into DB. Since it's already there, compute_hash might trigger duplicate detection.
    # To force re-processing, we might need to delete the DB record first or modify process_single_file.
    # Actually, process_single_file in this codebase checks hash.
    
    # Let's delete the existing DB record to force a fresh process
    print("Clearing existing DB records for this file to force re-processing...")
    with db.get_connection() as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT id FROM documents WHERE filename = {db.placeholder}", (target_filename,))
        row = cur.fetchone()
        if row:
            doc_id = row[0]
            cur.execute(f"DELETE FROM ocr_texts WHERE id_doc = {db.placeholder}", (doc_id,))
            cur.execute(f"DELETE FROM documents WHERE id = {db.placeholder}", (doc_id,))
            conn.commit()
    
    result = process_single_file(
        target_path,
        pipeline,
        classifier,
        db,
        os.path.join(project_root, "processed"),
        os.path.join(project_root, "errors"),
        delete_original=False, # Don't delete our test file
        ocr_enabled=True,
        classification_enabled=True,
        logger=logger,
        input_root=os.path.join(project_root, "processed")
    )
    # Print result status
    print(f"Result: {result}")
    
    if result.get("status") == "FAILED":
        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(f"SELECT error_message FROM documents WHERE id = {db.placeholder}", (result['doc_id'],))
            row = cur.fetchone()
            if row:
                print(f"--- FAILURE REASON ---")
                print(row[0])

    # Check new text
    if result.get('doc_id'):
        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(f"SELECT text FROM ocr_texts WHERE id_doc = {db.placeholder}", (result['doc_id'],))
            row = cur.fetchone()
            new_text = row[0] if row else "NO TEXT FOUND"
            print("--- NEW OCR TEXT (Sample) ---")
            print(new_text[:500] if new_text else "EMPTY TEXT")
    else:
        print("NO DOC_ID RETURNED")

if __name__ == "__main__":
    main()
