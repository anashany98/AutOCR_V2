"""
Simple rule‑based document classifier.

The classifier inspects the extracted text of a document and assigns a high
level type along with optional tags based on keyword matching.  This is a
lightweight approach suitable for well defined document classes such as
invoices, contracts, receipts and reports.  If no keywords are found the
document is classified as ``Unknown``.

To improve classification you can extend ``KEYWORDS`` with additional
categories and synonyms.  The keys of ``KEYWORDS`` are the human readable
document types, while the values are lists of keywords that trigger that
type.  Case is ignored during matching.
"""

from __future__ import annotations

import re
import pickle
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Mapping of document types to lists of keywords.  Feel free to extend this
# dictionary with additional categories and translations.
KEYWORDS = {
    "Invoice": ["invoice", "factura", "bill", "recibo"],
    "Contract": ["contract", "contrato", "agreement", "acuerdo"],
    "Receipt": ["receipt", "recibo", "ticket", "comprobante"],
    "Estimate": ["presupuesto", "estimate", "cotización", "quote"],
    "Report": ["report", "informe", "memo", "informe"],
    "Letter": ["dear", "estimado", "letter", "carta"],
    "Technical Plan": ["plano", "drawing", "scheme", "esquema", "blueprint", "diagrama", "technical drawing"],
}


class DocumentClassifier:
    """Perform keyword‑based determination or ML classification."""

    def __init__(self, keywords: Optional[dict] = None, model_path: Optional[str] = None) -> None:
        self.keywords = keywords or KEYWORDS
        self.model = None
        
        if model_path:
            p = Path(model_path)
            if p.exists():
                try:
                    with open(p, "rb") as f:
                        self.model = pickle.load(f)
                    logger.info(f"Loaded AI Classifier from {p}")
                except Exception as e:
                    logger.error(f"Failed to load AI Classifier: {e}")

    def classify(self, text: str) -> Tuple[str, List[str]]:
        """
        Assign a document type and tags based on keyword occurrences.

        Parameters
        ----------
        text:
            Free text extracted from OCR.

        Returns
        -------
        Tuple[str, List[str]]
            A tuple containing the document type (one of the keys in
            ``self.keywords`` or ``'Unknown'``) and a list of matching tags.
        """
        if not text:
            return "Unknown", []

        # 1. Try AI Model
        if self.model:
            try:
                prediction = self.model.predict([text])[0]
                # We can add confidence check here if using predict_proba
                return prediction, ["AI"]
            except Exception as e:
                logger.error(f"AI Prediction failed: {e}")
        
        # 2. Fallback to Keywords
        lower = text.lower()
        found_type: Optional[str] = None
        found_tags: List[str] = []

        for doc_type, triggers in self.keywords.items():
            for kw in triggers:
                if kw.lower() in lower:
                    found_type = doc_type
                    found_tags.append(kw)
                    # Break after first match to avoid assigning multiple types
                    break
            if found_type:
                break

        if not found_type:
            found_type = "Unknown"

        return found_type, found_tags