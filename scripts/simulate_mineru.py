import os
import sys
import json
import datetime
from modules.db_manager import DBManager
# Ensure project root is in path
sys.path.append(os.getcwd())

def main():
    print("Simulating MinerU Data Insertion...")
    
    # Init DB
    config = {"database": {"engine": "sqlite"}} # Default to sqlite for local verify
    db = DBManager(config)
    
    # 1. Create Document
    doc_id = db.insert_document(
        filename="mineru_test_doc.pdf",
        path="tests/mineru_test_doc.pdf",
        md5_hash="dummy_hash_mineru",
        timestamp=datetime.datetime.now(),
        duration=1.5,
        status="OK",
        doc_type="Report",
        tags=["MinerU_Test"],
        workflow_state="new"
    )
    print(f"Created Document ID: {doc_id}")
    
    # 2. Prepare MinerU Data
    mineru_tables = [
        """<table border="1">
            <thead>
                <tr><th>Header A</th><th>Header B</th></tr>
            </thead>
            <tbody>
                <tr><td>Value 1</td><td>Value 2</td></tr>
                <tr><td>Date</td><td>2023-01-01</td></tr>
            </tbody>
        </table>""",
        """<table class="table table-striped">
            <tr><td>Another</td><td>Table</td></tr>
        </table>"""
    ]
    
    mineru_formulas = [
        "E = mc^2",
        "\\frac{1}{2}mv^2"
    ]
    
    structured_data = {
        "mineru": {
            "tables": mineru_tables,
            "formulas": mineru_formulas,
            "metadata": {"version": "1.0"}
        }
    }
    
    # 3. Insert OCR Text with Structured Data
    db.insert_ocr_text(
        id_doc=doc_id,
        text="# MinerU Test Document\n\nThis is a test content.",
        markdown_text="# MinerU Test Document\n\nThis is a test content.",
        language="eng",
        confidence=0.99,
        structured_data=structured_data
    )
    
    print("Inserted OCR Text with MinerU Structured Data.")
    print(f"Please open http://127.0.0.1:5000/document/{doc_id} to verify.")

if __name__ == "__main__":
    main()
