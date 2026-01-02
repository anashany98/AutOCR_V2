"""
Data normalization module for extracted fields.

Standardizes dates, currencies, and vendor names to canonical formats.
"""

import re
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class DataNormalizer:
    """Normalize extracted data to standard formats."""
    
    # Legal entity suffixes to remove
    LEGAL_SUFFIXES = [
        r'\bS\.?L\.?', r'\bS\.?A\.?', r'\bLtd\.?', r'\bInc\.?',
        r'\bCorp\.?', r'\bGmbH', r'\bLLC', r'\bLimited'
    ]
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize normalizer with optional configuration."""
        self.config = config or {}
        
    def normalize(self,fields: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Normalize all extracted fields.
        
        Args:
            fields: Dictionary of extracted fields
            
        Returns:
            Normalized fields dictionary
        """
        normalized = {}
        
        if 'date' in fields:
            normalized['date'] = self._normalize_date(fields['date'])
            
        if 'total' in fields:
            normalized['total'] = self._normalize_amount(fields['total'])
            
        if 'vendor' in fields:
            normalized['vendor'] = self._normalize_vendor(fields['vendor'])
            
        return normalized
    
    def _normalize_date(self, date_field: Dict) -> Dict:
        """Ensure date is in ISO 8601 format (YYYY-MM-DD)."""
        # Date should already be normalized by FieldExtractor
        # This is a safety check
        value = date_field.get('value', '')
        
        # Verify it's ISO format
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', value):
            logger.warning(f"Date not in ISO format: {value}")
            
        return date_field
    
    def _normalize_amount(self, amount_field: Dict) -> Dict:
        """Ensure amount is numeric with currency code."""
        # Amount should already be normalized by FieldExtractor
        # Ensure value is float
        if isinstance(amount_field.get('value'), (int, float)):
            amount_field['value'] = round(float(amount_field['value']), 2)
        
        # Ensure currency code exists
        if 'currency' not in amount_field:
            amount_field['currency'] = 'EUR'  # Default
            
        return amount_field
    
    def _normalize_vendor(self, vendor_field: Dict) -> Dict:
        """Normalize vendor name by removing legal suffixes and standardizing format."""
        value = vendor_field.get('value', '')
        
        # Remove legal suffixes
        for suffix in self.LEGAL_SUFFIXES:
            value = re.sub(suffix, '', value, flags=re.IGNORECASE)
        
        # Normalize whitespace
        value = ' '.join(value.split())
        
        # Titlecase if all uppercase
        if value.isupper():
            value = value.title()
        
        vendor_field['value'] = value.strip()
        vendor_field['normalized'] = True
        
        return vendor_field


__all__ = ['DataNormalizer']
