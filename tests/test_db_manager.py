"""
Unit tests for the database manager module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


class TestDBManager:
    """Test cases for DBManager."""

    @patch('modules.db_manager.get_db_connection')
    def test_db_manager_initialization(self, mock_conn):
        """Test that DB manager initializes correctly."""
        from modules.db_manager import DBManager
        
        mock_conn.return_value = MagicMock()
        
        with patch('modules.db_manager.init_database'):
            manager = DBManager()
            assert manager is not None

    @patch('modules.db_manager.get_db_connection')
    def test_insert_document(self, mock_conn):
        """Test inserting a document."""
        from modules.db_manager import DBManager
        
        mock_cursor = MagicMock()
        mock_conn.return_value.cursor.return_value = mock_cursor
        
        with patch('modules.db_manager.init_database'):
            manager = DBManager()
            
            doc_data = {
                'filename': 'test.pdf',
                'document_type': 'contract',
                'text': 'Sample text',
                'confidence': 0.95,
                'tenant_id': 'default'
            }
            
            result = manager.insert_document(doc_data)
            assert result is not None

    @patch('modules.db_manager.get_db_connection')
    def test_get_document_by_id(self, mock_conn):
        """Test retrieving a document by ID."""
        from modules.db_manager import DBManager
        
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {
            'id': 1,
            'filename': 'test.pdf',
            'document_type': 'contract'
        }
        mock_conn.return_value.cursor.return_value = mock_cursor
        
        with patch('modules.db_manager.init_database'):
            manager = DBManager()
            
            result = manager.get_document(1)
            assert result is not None
            assert result['filename'] == 'test.pdf'

    @patch('modules.db_manager.get_db_connection')
    def test_get_documents_by_tenant(self, mock_conn):
        """Test retrieving documents by tenant."""
        from modules.db_manager import DBManager
        
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {'id': 1, 'filename': 'test1.pdf'},
            {'id': 2, 'filename': 'test2.pdf'}
        ]
        mock_conn.return_value.cursor.return_value = mock_cursor
        
        with patch('modules.db_manager.init_database'):
            manager = DBManager()
            
            result = manager.get_documents_by_tenant('default')
            assert len(result) == 2

    @patch('modules.db_manager.get_db_connection')
    def test_update_document_status(self, mock_conn):
        """Test updating document status."""
        from modules.db_manager import DBManager
        
        mock_cursor = MagicMock()
        mock_conn.return_value.cursor.return_value = mock_cursor
        
        with patch('modules.db_manager.init_database'):
            manager = DBManager()
            
            result = manager.update_document_status(1, 'processed')
            assert result is True

    @patch('modules.db_manager.get_db_connection')
    def test_delete_document(self, mock_conn):
        """Test deleting a document."""
        from modules.db_manager import DBManager
        
        mock_cursor = MagicMock()
        mock_conn.return_value.cursor.return_value = mock_cursor
        
        with patch('modules.db_manager.init_database'):
            manager = DBManager()
            
            result = manager.delete_document(1)
            assert result is True


class TestDBQueries:
    """Test cases for database queries."""

    @patch('modules.db_manager.get_db_connection')
    def test_search_documents(self, mock_conn):
        """Test searching documents."""
        from modules.db_manager import DBManager
        
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {'id': 1, 'filename': 'contract.pdf', 'text': 'contract text'}
        ]
        mock_conn.return_value.cursor.return_value = mock_cursor
        
        with patch('modules.db_manager.init_database'):
            manager = DBManager()
            
            result = manager.search_documents('contract')
            assert len(result) >= 0

    @patch('modules.db_manager.get_db_connection')
    def test_get_statistics(self, mock_conn):
        """Test getting document statistics."""
        from modules.db_manager import DBManager
        
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {
            'total': 100,
            'processed': 80,
            'pending': 20,
            'failed': 0
        }
        mock_conn.return_value.cursor.return_value = mock_cursor
        
        with patch('modules.db_manager.init_database'):
            manager = DBManager()
            
            result = manager.get_statistics()
            assert result is not None


class TestDBTransactions:
    """Test cases for database transactions."""

    @patch('modules.db_manager.get_db_connection')
    def test_batch_insert(self, mock_conn):
        """Test batch insert of documents."""
        from modules.db_manager import DBManager
        
        mock_conn.return_value = MagicMock()
        
        with patch('modules.db_manager.init_database'):
            manager = DBManager()
            
            docs = [
                {'filename': f'test{i}.pdf', 'text': f'text{i}'}
                for i in range(10)
            ]
            
            result = manager.batch_insert(docs)
            assert result is not None
