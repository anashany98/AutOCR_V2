"""
Anomaly detection module for invoice/document validation.

Detects unusual patterns in extracted data to flag potential errors or fraud.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta
from statistics import mean, stdev

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detect anomalies in extracted document fields."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize anomaly detector with optional configuration."""
        self.config = config or {}
        
    def detect(self, fields: Dict[str, Dict], document_history: List[Dict] = None) -> List[str]:
        """
        Detect anomalies in extracted fields.
        
        Args:
            fields: Extracted and normalized fields
            document_history: Optional list of previous documents for statistical analysis
            
        Returns:
            List of anomaly codes
        """
        anomalies = []
        
        # Check for missing critical fields
        if not fields.get('date'):
            anomalies.append('missing_date')
        if not fields.get('total'):
            anomalies.append('missing_total')
        if not fields.get('vendor'):
            anomalies.append('missing_vendor')
        
        # Date-based anomalies
        if 'date' in fields:
            date_anomalies = self._detect_date_anomalies(fields['date'])
            anomalies.extend(date_anomalies)
        
        # Amount-based anomalies
        if 'total' in fields and document_history:
            amount_anomalies = self._detect_amount_anomalies(
                fields['total'], document_history
            )
            anomalies.extend(amount_anomalies)
        
        return anomalies
    
    def _detect_date_anomalies(self, date_field: Dict) -> List[str]:
        """Detect date-related anomalies."""
        anomalies = []
        
        try:
            date_value = datetime.strptime(date_field['value'], '%Y-%m-%d')
            today = datetime.now()
            
            # Future date (more than 7 days ahead)
            if date_value > today + timedelta(days=7):
                anomalies.append('future_date')
            
            # Weekend date (Saturday=5, Sunday=6)
            if date_value.weekday() >= 5:
                anomalies.append('weekend_date')
            
            # Very old date (more than 5 years ago)
            if date_value < today - timedelta(days=365*5):
                anomalies.append('very_old_date')
                
        except ValueError:
            anomalies.append('invalid_date_format')
        
        return anomalies
    
    def _detect_amount_anomalies(self, total_field: Dict, 
                                 document_history: List[Dict]) -> List[str]:
        """Detect amount-related anomalies using statistical analysis."""
        anomalies = []
        
        try:
            current_amount = float(total_field['value'])
            
            # Extract historical amounts
            historical_amounts = []
            for doc in document_history:
                if 'total' in doc.get('fields', {}):
                    try:
                        amt = float(doc['fields']['total']['value'])
                        historical_amounts.append(amt)
                    except (ValueError, KeyError):
                        continue
            
            # Need at least 5 historical documents for statistics
            if len(historical_amounts) < 5:
                return anomalies
            
            # Calculate statistics
            avg = mean(historical_amounts)
            std = stdev(historical_amounts) if len(historical_amounts) > 1 else 0
            
            # Unusual amount (more than 3 standard deviations from mean)
            if std > 0 and abs(current_amount - avg) > 3 * std:
                anomalies.append('unusual_amount')
            
            # Very large amount (more than 10x average)
            if current_amount > avg * 10:
                anomalies.append('very_large_amount')
            
            # Round number check (might be estimated)
            if current_amount == round(current_amount, -2):  # e.g., 1000.00
                if current_amount >= 1000:
                    anomalies.append('suspicious_round_number')
                    
        except (ValueError, KeyError, ZeroDivisionError):
            anomalies.append('invalid_amount')
        
        return anomalies


__all__ = ['AnomalyDetector']
