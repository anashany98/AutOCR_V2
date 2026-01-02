"""
Smart field extraction module for invoice/document processing.

Intelligently extracts structured fields (Date, Total, Vendor) from OCR text
with confidence scoring and position-aware analysis.
"""

import re
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from .vendor_matcher import VendorMatcher

logger = logging.getLogger(__name__)


class FieldExtractor:
    """Intelligent extraction of structured fields from OCR text."""
    
    # Date patterns (ordered by preference)
    DATE_PATTERNS = [
        # ISO format: 2026-01-02
        (r'\b(\d{4})-(\d{2})-(\d{2})\b', '%Y-%m-%d'),
        # European: 02/01/2026 or 02-01-2026
        (r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b', '%d/%m/%Y'),
        # US format: 01/02/2026
        (r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b', '%m/%d/%Y'),
        # Spanish: 2 de enero de 2026
        (r'\b(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})\b', 'es_long'),
    ]
    
    # Amount patterns
    AMOUNT_PATTERNS = [
        # European: 1.234,56€ or €1.234,56
        (r'[€$£]\s?(\d{1,3}(?:\.\d{3})*,\d{2})', 'eu'),
        (r'(\d{1,3}(?:\.\d{3})*,\d{2})\s?[€$£]', 'eu'),
        # US: $1,234.56 or 1,234.56$
        (r'[€$£]\s?(\d{1,3}(?:,\d{3})*\.\d{2})', 'us'),
        (r'(\d{1,3}(?:,\d{3})*\.\d{2})\s?[€$£]', 'us'),
        # Simple: 123.45 or 123,45
        (r'\b(\d+[.,]\d{2})\b', 'simple'),
    ]
    
    # Currency symbols
    CURRENCY_MAP = {
        '€': 'EUR',
        '$': 'USD',
        '£': 'GBP',
    }
    
    # Spanish month names
    SPANISH_MONTHS = {
        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
        'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
        'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
    }
    
    # Context keywords for field detection
    DATE_KEYWORDS = ['fecha', 'date', 'emis', 'issued']
    TOTAL_KEYWORDS = ['total', 'importe', 'amount', 'suma', 'pagar']
    VENDOR_KEYWORDS = ['proveedor', 'vendor', 'empresa', 'company', 'razón social']
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize field extractor with optional configuration."""
        self.config = config or {}
        self.vendor_matcher = VendorMatcher(self.config)
        
    def extract_fields(self, text: str, blocks: List[Dict] = None) -> Dict[str, Dict]:
        """
        Extract structured fields from OCR text.
        
        Args:
            text: Full OCR text
            blocks: Optional list of text blocks with position info
            
        Returns:
            Dictionary of extracted fields with confidence scores
        """
        fields = {}
        
        # Extract date
        date_result = self._extract_date(text, blocks)
        if date_result:
            fields['date'] = date_result
            
        # Extract total amount
        total_result = self._extract_total(text, blocks)
        if total_result:
            fields['total'] = total_result
            
        # Extract vendor
        vendor_result = self._extract_vendor(text, blocks)
        if vendor_result:
            fields['vendor'] = vendor_result
            
        return fields
    
    def _extract_date(self, text: str, blocks: List[Dict] = None) -> Optional[Dict]:
        """Extract date field with confidence scoring."""
        best_match = None
        best_confidence = 0.0
        
        lines = text.split('\n')
        for line_idx, line in enumerate(lines):
            for pattern, fmt in self.DATE_PATTERNS:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    # Try to parse the date
                    date_value = self._parse_date(match.group(), fmt)
                    if not date_value:
                        continue
                        
                    # Calculate confidence
                    confidence = self._calculate_date_confidence(
                        line, match, line_idx, len(lines)
                    )
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = {
                            'value': date_value.strftime('%Y-%m-%d'),
                            'confidence': round(confidence, 2),
                            'raw': match.group().strip(),
                            'format': fmt
                        }
        
        return best_match
    
    def _parse_date(self, date_str: str, fmt: str) -> Optional[datetime]:
        """Parse date string to datetime object."""
        try:
            if fmt == 'es_long':
                # Parse Spanish long format
                match = re.search(r'(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})', date_str, re.IGNORECASE)
                if match:
                    day = int(match.group(1))
                    month_name = match.group(2).lower()
                    year = int(match.group(3))
                    month = self.SPANISH_MONTHS.get(month_name)
                    if month:
                        return datetime(year, month, day)
            else:
                return datetime.strptime(date_str, fmt)
        except (ValueError, AttributeError):
            pass
        return None
    
    def _calculate_date_confidence(self, line: str, match: re.Match, 
                                   line_idx: int, total_lines: int) -> float:
        """Calculate confidence score for date extraction."""
        confidence = 0.5  # Base confidence for valid date format
        
        # Bonus for context keywords nearby
        line_lower = line.lower()
        if any(kw in line_lower for kw in self.DATE_KEYWORDS):
            confidence += 0.3
        
        # Bonus for being in top 30% of document
        if line_idx < total_lines * 0.3:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _extract_total(self, text: str, blocks: List[Dict] = None) -> Optional[Dict]:
        """Extract total amount with currency detection."""
        best_match = None
        best_confidence = 0.0
        
        lines = text.split('\n')
        for line_idx, line in enumerate(lines):
            for pattern, num_format in self.AMOUNT_PATTERNS:
                matches = re.finditer(pattern, line)
                for match in matches:
                    # Parse amount
                    amount_str = match.group(1) if match.groups() else match.group()
                    amount_value = self._parse_amount(amount_str, num_format)
                    if amount_value is None:
                        continue
                    
                    # Detect currency
                    currency = self._detect_currency(match.group())
                    
                    # Calculate confidence
                    confidence = self._calculate_amount_confidence(
                        line, match, line_idx, len(lines)
                    )
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = {
                            'value': round(amount_value, 2),
                            'confidence': round(confidence, 2),
                            'raw': match.group().strip(),
                            'currency': currency
                        }
        
        return best_match
    
    def _parse_amount(self, amount_str: str, num_format: str) -> Optional[float]:
        """Parse amount string to float."""
        try:
            if num_format == 'eu':
                # Remove thousand separators and replace comma with dot
                amount_str = amount_str.replace('.', '').replace(',', '.')
            elif num_format == 'us':
                # Remove thousand separators
                amount_str = amount_str.replace(',', '')
            elif num_format == 'simple':
                # Replace comma with dot if present
                amount_str = amount_str.replace(',', '.')
            
            return float(amount_str)
        except ValueError:
            return None
    
    def _detect_currency(self, text: str) -> str:
        """Detect currency from text."""
        for symbol, code in self.CURRENCY_MAP.items():
            if symbol in text:
                return code
        return 'EUR'  # Default to EUR
    
    def _calculate_amount_confidence(self, line: str, match: re.Match,
                                     line_idx: int, total_lines: int) -> float:
        """Calculate confidence score for amount extraction."""
        confidence = 0.4  # Base confidence for valid amount format
        
        # Strong bonus for "total" keyword
        line_lower = line.lower()
        if any(kw in line_lower for kw in self.TOTAL_KEYWORDS):
            confidence += 0.4
        
        # Moderate bonus for being in bottom 50% (totals usually at end)
        if line_idx > total_lines * 0.5:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _extract_vendor(self, text: str, blocks: List[Dict] = None) -> Optional[Dict]:
        """Extract vendor/company name."""
        lines = text.split('\n')
        
        # Look for vendor in top 20% of document
        top_lines = lines[:max(1, int(len(lines) * 0.2))]
        
        for line_idx, line in enumerate(top_lines):
            line = line.strip()
            if not line or len(line) < 3:
                continue
            
            # Skip lines with only numbers or common words
            if re.match(r'^[\d\s\-./]+$', line):
                continue
            
            # Check for context keywords
            line_lower = line.lower()
            has_keyword = any(kw in line_lower for kw in self.VENDOR_KEYWORDS)
            
            # First non-empty line in caps or with keyword context is likely vendor
            if line.isupper() or has_keyword or line_idx == 0:
                confidence = 0.6
                if line.isupper():
                    confidence += 0.2
                if has_keyword:
                    confidence += 0.2
                
                normalized = self.vendor_matcher.normalize(line)
                return {
                    'value': normalized,
                    'confidence': round(min(confidence, 1.0), 2),
                    'raw': line
                }
        
        # Fallback: return first substantial line
        for line in top_lines:
            line = line.strip()
            if len(line) > 5 and not re.match(r'^[\d\s\-./]+$', line):
                normalized = self.vendor_matcher.normalize(line)
                return {
                    'value': normalized,
                    'confidence': 0.4,
                    'raw': line
                }
        
        return None


__all__ = ['FieldExtractor']
