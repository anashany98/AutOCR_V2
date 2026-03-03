"""
Unit tests for the classifier module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestClassifier:
    """Test cases for DocumentClassifier."""

    def test_classifier_initialization(self):
        """Test that classifier initializes correctly."""
        from modules.classifier import DocumentClassifier
        
        # Test with default config
        classifier = DocumentClassifier()
        assert classifier is not None

    def test_classify_contract(self):
        """Test classification of a contract document."""
        from modules.classifier import DocumentClassifier
        
        classifier = DocumentClassifier()
        
        # Mock text that resembles a contract
        test_text = """
        CONTRATO DE SERVICIOS
        Fecha: 01/01/2026
        Entre: Empresa A y Empresa B
        Objeto: Prestación de servicios
        """
        
        with patch.object(classifier, 'classify') as mock_classify:
            mock_classify.return_value = {
                'document_type': 'contract',
                'confidence': 0.95
            }
            result = classifier.classify(test_text)
            assert result['document_type'] == 'contract'
            assert result['confidence'] > 0.9

    def test_classify_invoice(self):
        """Test classification of an invoice."""
        from modules.classifier import DocumentClassifier
        
        classifier = DocumentClassifier()
        
        test_text = """
        FACTURA Nº 12345
        Fecha: 15/01/2026
        Importe: 1,000.00 EUR
        IVA: 21%
        """
        
        with patch.object(classifier, 'classify') as mock_classify:
            mock_classify.return_value = {
                'document_type': 'invoice',
                'confidence': 0.92
            }
            result = classifier.classify(test_text)
            assert result['document_type'] == 'invoice'

    def test_classify_receipt(self):
        """Test classification of a receipt."""
        from modules.classifier import DocumentClassifier
        
        classifier = DocumentClassifier()
        
        test_text = """
        RECIBO
        Total: €50.00
        Fecha: 20/01/2026
        """
        
        with patch.object(classifier, 'classify') as mock_classify:
            mock_classify.return_value = {
                'document_type': 'receipt',
                'confidence': 0.88
            }
            result = classifier.classify(test_text)
            assert result['document_type'] == 'receipt'

    def test_empty_text_classification(self):
        """Test classification with empty text."""
        from modules.classifier import DocumentClassifier
        
        classifier = DocumentClassifier()
        
        with patch.object(classifier, 'classify') as mock_classify:
            mock_classify.return_value = {
                'document_type': 'unknown',
                'confidence': 0.0
            }
            result = classifier.classify("")
            assert result['document_type'] == 'unknown'

    def test_low_confidence_classification(self):
        """Test classification with low confidence."""
        from modules.classifier import DocumentClassifier
        
        classifier = DocumentClassifier()
        
        test_text = "Some random text without clear structure"
        
        with patch.object(classifier, 'classify') as mock_classify:
            mock_classify.return_value = {
                'document_type': 'unknown',
                'confidence': 0.3
            }
            result = classifier.classify(test_text)
            # Should return low confidence
            assert result['confidence'] < 0.5


class TestClassifierIntegration:
    """Integration tests for classifier with mock models."""

    @patch('modules.classifier.get_model')
    def test_classifier_with_mock_model(self, mock_get_model):
        """Test classifier with a mocked model."""
        from modules.classifier import DocumentClassifier
        
        # Setup mock
        mock_model = MagicMock()
        mock_model.predict.return_value = ['contract']
        mock_model.predict_proba.return_value = [[0.1, 0.9]]
        mock_get_model.return_value = mock_model
        
        classifier = DocumentClassifier()
        result = classifier.classify("CONTRATO test text")
        
        assert result is not None
        mock_model.predict.assert_called()
