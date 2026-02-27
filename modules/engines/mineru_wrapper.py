"""
MinerU OCR Engine Wrapper for AutOCR.

This module provides a wrapper around MinerU for extracting structured
content (Markdown, tables, formulas) from complex PDF documents.
MinerU is used as a secondary engine when PaddleOCR-VL is the primary.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# Lazy imports to avoid loading MinerU if not needed
_mineru_available = None


def _check_mineru_available() -> bool:
    """Check if MinerU is installed and available."""
    global _mineru_available
    if _mineru_available is None:
        try:
            from mineru import MinerU
            _mineru_available = True
        except ImportError:
            _mineru_available = False
    return _mineru_available


class MinerUEngine:
    """
    MinerU-based document processing engine.
    
    Specializes in extracting structured content from complex PDFs:
    - Tables → HTML format
    - Formulas → LaTeX format
    - Multi-column layouts → Proper reading order
    - Headers/Footers → Removed for clean output
    
    Parameters
    ----------
    config : dict
        Configuration dictionary with MinerU settings.
    logger : logging.Logger, optional
        Logger for diagnostic messages.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.enabled = False
        self._mineru = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """
        Initialize the MinerU engine.
        
        Returns
        -------
        bool
            True if initialization succeeded, False otherwise.
        """
        if self._initialized:
            return self.enabled
        
        if not _check_mineru_available():
            self.logger.warning(
                "MinerU is not installed. Install with: pip install 'mineru[all]'"
            )
            self.enabled = False
            self._initialized = True
            return False
        
        try:
            from mineru import MinerU
            
            # Get configuration options
            use_gpu = self.config.get("use_gpu", True)
            backend = "cuda" if use_gpu else "pipeline"
            
            self.logger.info(f"Initializing MinerU engine (backend={backend})...")
            
            # Initialize MinerU with configuration
            self._mineru = MinerU(
                backend=backend,
                # Additional config can be added here
            )
            
            self.enabled = True
            self._initialized = True
            self.logger.info("MinerU engine initialized successfully.")
            return True
            
        except Exception as exc:
            self.logger.error(f"Failed to initialize MinerU: {exc}")
            self.enabled = False
            self._initialized = True
            return False
    
    def process(self, file_path: str, output_format: str = "markdown") -> Dict[str, Any]:
        """
        Process a document and extract structured content.
        
        Parameters
        ----------
        file_path : str
            Path to the PDF or image file.
        output_format : str
            Output format: 'markdown' or 'json'
        
        Returns
        -------
        dict
            Extracted content with keys:
            - 'text': Full extracted text or markdown
            - 'tables': List of tables in HTML format
            - 'formulas': List of formulas in LaTeX format
            - 'metadata': Document metadata
        """
        if not self.enabled:
            if not self.initialize():
                return {"error": "MinerU not available", "text": ""}
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir)
                
                # Run MinerU processing
                self._mineru.run(
                    input_path=file_path,
                    output_path=str(output_path),
                )
                
                # Read the output
                result = self._read_output(output_path, output_format)
                return result
                
        except Exception as exc:
            self.logger.error(f"MinerU processing failed: {exc}")
            return {"error": str(exc), "text": ""}
    
    def extract_tables(self, file_path: str) -> List[str]:
        """
        Extract tables from a document.
        
        Parameters
        ----------
        file_path : str
            Path to the document.
        
        Returns
        -------
        list[str]
            List of tables in HTML format.
        """
        result = self.process(file_path)
        return result.get("tables", [])
    
    def extract_formulas(self, file_path: str) -> List[str]:
        """
        Extract mathematical formulas from a document.
        
        Parameters
        ----------
        file_path : str
            Path to the document.
        
        Returns
        -------
        list[str]
            List of formulas in LaTeX format.
        """
        result = self.process(file_path)
        return result.get("formulas", [])
    
    def _read_output(self, output_path: Path, output_format: str) -> Dict[str, Any]:
        """Read and parse MinerU output files."""
        result = {
            "text": "",
            "tables": [],
            "formulas": [],
            "metadata": {},
        }
        
        # Look for markdown output
        md_files = list(output_path.glob("**/*.md"))
        if md_files:
            result["text"] = md_files[0].read_text(encoding="utf-8")
        
        # Look for JSON output with structured data
        json_files = list(output_path.glob("**/*.json"))
        if json_files:
            import json
            try:
                data = json.loads(json_files[0].read_text(encoding="utf-8"))
                result["metadata"] = data.get("metadata", {})
                
                # Extract tables and formulas from structured output
                for item in data.get("content", []):
                    if item.get("type") == "table":
                        result["tables"].append(item.get("html", ""))
                    elif item.get("type") == "formula":
                        result["formulas"].append(item.get("latex", ""))
            except Exception:
                pass
        
        return result
    
    def is_complex_document(self, file_path: str) -> bool:
        """
        Heuristically determine if a document would benefit from MinerU processing.
        
        Parameters
        ----------
        file_path : str
            Path to the document.
        
        Returns
        -------
        bool
            True if the document appears complex (has tables, formulas, multi-column).
        """
        ext = Path(file_path).suffix.lower()
        
        # Only process PDFs with MinerU
        if ext != ".pdf":
            return False
        
        # 1. Check file size (heuristics: complex manuals often > 2MB)
        size_mb: float = 0.0
        try:
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if size_mb > 50: # Too large might timeout on CPU
                self.logger.info(f"PDF too large for MinerU ({size_mb:.2f}MB), skipping class check.")
                return False
        except Exception:
            pass

        # 2. Analyze content with PyMuPDF (fitz) if available
        try:
            import fitz  # PyMuPDF
            with fitz.open(file_path) as doc:
                # Check first 3 pages
                pages_to_check = min(3, len(doc))
                
                has_vector_graphics = False
                has_tables_keywords = False
                
                for i in range(pages_to_check):
                    page = doc[i]
                    # Check for vector drawings (often tables/diagrams)
                    drawings = page.get_drawings()
                    if len(drawings) > 20: # Arbitrary threshold for "complex layout"
                        has_vector_graphics = True
                    
                    # Check for keywords
                    text = page.get_text().lower()
                    if any(x in text for x in ["table", "tabla", "cuadro", "figura", "figure", "equation", "formula"]):
                        has_tables_keywords = True

                if has_vector_graphics or has_tables_keywords:
                    self.logger.info(f"MinerU Heuristic: Complex PDF detected (Graphics={has_vector_graphics}, Keywords={has_tables_keywords})")
                    return True

        except ImportError:
            self.logger.debug("PyMuPDF not installed, falling back to simple heuristics.")
            # Fallback: All PDFs > 1MB are "complex"
            if size_mb > 1.0:
                 return True

        except Exception as e:
            self.logger.warning(f"Error in MinerU heuristics: {e}")
            
        return False


# Singleton instance
_mineru_engine: Optional[MinerUEngine] = None


def get_mineru_engine(config: Optional[Dict[str, Any]] = None) -> MinerUEngine:
    """
    Get or create the MinerU engine singleton.
    
    Parameters
    ----------
    config : dict, optional
        Configuration for the engine.
    
    Returns
    -------
    MinerUEngine
        The MinerU engine instance.
    """
    global _mineru_engine
    
    if _mineru_engine is None:
        _mineru_engine = MinerUEngine(config or {})
    
    return _mineru_engine


__all__ = ["MinerUEngine", "get_mineru_engine"]
