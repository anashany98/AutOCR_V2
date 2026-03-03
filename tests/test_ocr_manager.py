"""
Unit tests for the OCR manager module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path


class TestOCRManager:
    """Test cases for OCRManager."""

    def test_ocr_manager_initialization(self):
        """Test that OCR manager initializes correctly."""
        from modules.ocr_manager import OCRManager
        
        with patch('modules.ocr_manager.get_ocr_engine'):
            manager = OCRManager()
            assert manager is not None

    @patch('modules.ocr_manager.get_ocr_engine')
    def test_process_pdf_document(self, mock_engine):
        """Test processing a PDF document."""
        from modules.ocr_manager import OCRManager
        
        # Setup mock
        mock_ocr = MagicMock()
        mock_ocr.process.return_value = {
            'text': 'Sample extracted text',
            'confidence': 0.95,
            'pages': 1
        }
        mock_engine.return_value = mock_ocr
        
        manager = OCRManager()
        
        # Create a mock PDF file
        test_pdf_path = Path("/tmp/test.pdf")
        
        with patch('builtins.open', mock_open(read_data=b'PDF data')):
            with patch('pathlib.Path.exists', return_value=True):
                result = manager.process_document(test_pdf_path)
                
        assert result is not None

    @patch('modules.ocr_manager.get_ocr_engine')
    def test_process_image_document(self, mock_engine):
        """Test processing an image document."""
        from modules.ocr_manager import OCRManager
        
        mock_ocr = MagicMock()
        mock_ocr.process.return_value = {
            'text': 'Image text',
            'confidence': 0.92,
            'pages': 1
        }
        mock_engine.return_value = mock_ocr
        
        manager = OCRManager()
        
        test_image_path = Path("/tmp/test.jpg")
        
        with patch('builtins.open', mock_open(read_data=b'IMAGE data')):
            with patch('pathlib.Path.exists', return_value=True):
                result = manager.process_document(test_image_path)
                
        assert result is not None

    def test_unsupported_file_type(self):
        """Test handling of unsupported file types."""
        from modules.ocr_manager import OCRManager
        
        with patch('modules.ocr_manager.get_ocr_engine'):
            manager = OCRManager()
            
            test_path = Path("/tmp/test.xyz")
            
            with pytest.raises(ValueError) as exc_info:
                manager.process_document(test_path)
                
            assert "Unsupported file type" in str(exc_info.value)

    def test_nonexistent_file(self):
        """Test handling of non-existent files."""
        from modules.ocr_manager import OCRManager
        
        with patch('modules.ocr_manager.get_ocr_engine'):
            manager = OCRManager()
            
            test_path = Path("/tmp/nonexistent.pdf")
            
            with patch('pathlib.Path.exists', return_value=False):
                with pytest.raises(FileNotFoundError):
                    manager.process_document(test_path)


class TestOCREngines:
    """Test cases for OCR engines."""

    @patch('modules.ocr_manager.PaddleOCR')
    def test_paddle_ocr_engine(self, mock_paddle):
        """Test PaddleOCR engine."""
        from modules.ocr_manager import PaddleOCREngine
        
        mock_paddle_instance = MagicMock()
        mock_paddle.return_value = mock_paddle_instance
        
        engine = PaddleOCREngine()
        
        # Test that engine can process
        assert engine is not None

    @patch('modules.ocr_manager.FlorenceOCR')
    def test_florence_ocr_engine(self, mock_florence):
        """Test Florence OCR engine."""
        from modules.ocr_manager import FlorenceOCREngine
        
        mock_florence_instance = MagicMock()
        mock_florence.return_value = mock_florence_instance
        
        engine = FlorenceOCREngine()
        
        assert engine is not None


class TestOCRPerformance:
    """Performance-related tests for OCR."""

    @patch('modules.ocr_manager.get_ocr_engine')
    def test_batch_processing(self, mock_engine):
        """Test batch processing of multiple documents."""
        from modules.ocr_manager import OCRManager
        
        mock_ocr = MagicMock()
        mock_ocr.process.return_value = {
            'text': 'Sample text',
            'confidence': 0.9,
            'pages': 1
        }
        mock_engine.return_value = mock_ocr
        
        manager = OCRManager()
        
        test_files = [
            Path("/tmp/doc1.pdf"),
            Path("/tmp/doc2.pdf"),
            Path("/tmp/doc3.pdf"),
        ]
        
        with patch('pathlib.Path.exists', return_value=True):
            results = manager.process_batch(test_files)
            
        assert len(results) == len(test_files)

    @patch('modules.ocr_manager.get_ocr_engine')
    def test_caching_results(self, mock_engine):
        """Test that results are cached."""
        from modules.ocr_manager import OCRManager
        
        mock_ocr = MagicMock()
        mock_ocr.process.return_value = {
            'text': 'Cached text',
            'confidence': 0.9,
            'pages': 1
        }
        mock_engine.return_value = mock_ocr
        
        manager = OCRManager(enable_cache=True)
        
        test_path = Path("/tmp/test.pdf")
        
        # Process same file twice
        with patch('pathlib.Path.exists', return_value=True):
            result1 = manager.process_document(test_path)
            result2 = manager.process_document(test_path)
            
        # OCR should only be called once due to caching
        assert mock_ocr.process.call_count == 1
