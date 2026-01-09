import sys
import os
import logging
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IntegrationTest")

def test_integration():
    print("\n--- Testing Pipeline Integration ---")
    
    # 1. Mock dependencies
    # We don't want to run the heavy OCR engine, just test the flow
    with patch("postbatch_processor.load_document_pages") as mock_load, \
         patch("postbatch_processor.extract_content") as mock_extract, \
         patch("postbatch_processor.is_visual_document") as mock_is_visual:
        
        # Setup mocks
        mock_load.return_value = [] # No pages needed for logic part if we skip OCR blocks
        # Mock extract_content to return low confidence to trigger router
        mock_extract.return_value = ("Sample text", "eng", 0.35, False) 
        mock_is_visual.return_value = True

        # Import target function
        from postbatch_processor import process_single_file, PipelineComponents

        # Create mock objects
        pipeline = PipelineComponents(
            ocr_manager=MagicMock(),
            layout_manager=None,
            table_manager=None,
            fusion_manager=None,
            vision_manager=MagicMock(), # Needed for tags
            recheck_threshold=0.6,
            output_formats=["json"],
            save_markdown_in_db=False
        )
        
        # Mock vision manager classify to avoid real DL
        pipeline.vision_manager.classify_image.return_value = [("factura", 0.8)]
        
        db = MagicMock()
        db.check_duplicate.return_value = None
        
        # 2. Run process_single_file
        # We use a dummy file path
        result = process_single_file(
            file_path="dummy_document.jpg",
            pipeline=pipeline,
            classifier=None,
            db=db,
            processed_folder="out",
            failed_folder="fail",
            delete_original=False,
            ocr_enabled=True,
            classification_enabled=False,
            logger=logger,
            input_root="in"
        )
        
        # 3. Verify Output
        print("\n--- Verification Results ---")
        print(f"Result Status: {result.get('status')}")
        
        # We modified the pipeline to add 'interpretation_needed' to summary_payload
        # But process_single_file implementation in integration_test mocks might not return the full internal payload 
        # unless we check how process_single_file constructs its return value.
        # Upon reviewing postbatch_processor.py: 
        # It returns a dict associated with 'status', 'path', etc. 
        # The 'summary_payload' is used for saving JSON/Markdown, but the function return value is smaller.
        
        # HOWEVER, we added `tags.append("Requires_Advanced_Review")` in the code.
        # We can check if `tags` or equivalent made it out.
        # Actually, `process_single_file` does NOT return the tags list in its return dict (lines 607-614 of postbatch_processor).
        # It only returns {filename, status, duration, type, path, doc_id}.
        
        # To verify, we should rely on the Side Effect we logged.
        # Or better, we can verify that the `postbatch_processor.AdvancedInterpretationRouter.evaluate_document` was called.
        # Since we are running the REAL `Simple` router (not mocked in step 2 logic), we can't mock the class method easily 
        # inside the function unless we patch `postbatch_processor.AdvancedInterpretationRouter`.
        
        # Let's patch the Router in the test to verify it was called!
        
        # Rerunning with patch to verify interaction
    
    with patch("modules.interpretation_manager.AdvancedInterpretationRouter.evaluate_document") as mock_eval:
        # Setup mock return to force activation
        mock_eval.return_value = {
            "activar_interpretacion_avanzada": True,
            "accion": "invocar_modulo_interpretacion",
            "motivo": "Test Trigger",
            "confianza_decision": 0.8,
            "tipo_interpretacion": "test"
        }
        
        # We need to re-import or re-run the function logic. 
        # Since we already imported process_single_file, the patch should work if applied to where it is defined/imported.
        # `postbatch_processor.AdvancedInterpretationRouter`
        
        from postbatch_processor import process_single_file, PipelineComponents
        
        # Re-setup mocks (condensed)
        pipeline = PipelineComponents(MagicMock(), None, None, None, MagicMock(), 0.6, [], False)
        # Force low confidence to enter the router block
        with patch("postbatch_processor.extract_content", return_value=("", "eng", 0.1, False)), \
             patch("postbatch_processor.is_visual_document", return_value=True), \
             patch("postbatch_processor.load_document_pages", return_value=[]), \
             patch("postbatch_processor.compute_hash", return_value="dummy_hash"), \
             patch("postbatch_processor.DBManager") as mock_db_cls:
            
            mock_db = mock_db_cls.return_value
            mock_db.check_duplicate.return_value = None
            
            print("🚀 Executing Pipeline with Low Confidence (0.1)...")
            process_single_file("test_doc.jpg", pipeline, None, mock_db, "out", "fail", False, True, False, logger, "in")
            
            if mock_eval.called:
                print("✅ TEST PASSED: AdvancedInterpretationRouter was invoked!")
                print(f"   Call arguments: {mock_eval.call_args[0][0]['metricas_ocr']}")
            else:
                print("❌ TEST FAILED: Router was NOT invoked.")

if __name__ == "__main__":
    test_integration()
