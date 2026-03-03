"""
Test configuration and fixtures for AutoOCR.
"""
import os
import sys
import pytest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Return the test data directory."""
    return project_root / "tests" / "data"


@pytest.fixture
def sample_pdf_path(test_data_dir):
    """Return path to a sample PDF for testing."""
    sample_dir = test_data_dir / "samples"
    if not sample_dir.exists():
        pytest.skip(f"Test data directory not found: {sample_dir}")
    
    pdf_files = list(sample_dir.glob("*.pdf"))
    if not pdf_files:
        pytest.skip("No PDF files found in test data samples directory")
    
    return pdf_files[0]


@pytest.fixture
def sample_image_path(test_data_dir):
    """Return path to a sample image for testing."""
    sample_dir = test_data_dir / "samples"
    if not sample_dir.exists():
        pytest.skip(f"Test data directory not found: {sample_dir}")
    
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.tiff", "*.tif")
    for ext in image_extensions:
        images = list(sample_dir.glob(ext))
        if images:
            return images[0]
    
    pytest.skip("No image files found in test data samples directory")


@pytest.fixture
def mock_config():
    """Return a mock configuration for testing."""
    return {
        "postbatch": {
            "input_folder": "input",
            "processed_folder": "processed",
            "failed_folder": "errors",
            "file_types": [".pdf", ".jpg", ".png"],
            "ocr_enabled": True,
            "classification_enabled": True,
            "auto_verify": True,
        },
        "database": {
            "type": "sqlite",
            "path": ":memory:"
        }
    }


@pytest.fixture
def app_context():
    """Create an application context for testing."""
    from web_app.app import create_app
    app = create_app(testing=True)
    with app.app_context():
        yield app


@pytest.fixture
def client(app_context):
    """Create a test client with proper cleanup."""
    with app_context.test_client() as client:
        yield client
        # Cleanup happens automatically with context manager
