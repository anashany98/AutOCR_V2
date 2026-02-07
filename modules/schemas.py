from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime
import re

class BaseDocument(BaseModel):
    """Base schema for all document types."""
    doc_type: str = Field(..., description="Type of document (e.g., Invoice, Receipt, ID)")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    vlm_validated: bool = False

class InvoiceSchema(BaseDocument):
    """Schema for Invoices."""
    vendor_name: str = Field(..., description="Name of the seller/company")
    cif_nif: Optional[str] = Field(None, description="Tax identification number (CIF/NIF in Spain)")
    date: Optional[str] = Field(None, description="Date of the invoice (YYYY-MM-DD)")
    invoice_number: Optional[str] = Field(None, description="Unique identifier for the invoice")
    
    base_amount: float = Field(default=0.0, description="Taxable base amount")
    vat_percent: float = Field(default=21.0, description="VAT percentage")
    vat_amount: float = Field(default=0.0, description="Calculated VAT amount")
    total_amount: float = Field(..., description="Final total amount including taxes")
    
    currency: str = Field(default="EUR")

    @field_validator('total_amount')
    @classmethod
    def check_math_consistency(cls, v: float, info):
        values = info.data
        base = values.get('base_amount', 0.0)
        vat = values.get('vat_amount', 0.0)
        
        # If both base and vat are provided, check if they sum up to total
        if base > 0 and vat > 0:
            expected = round(base + vat, 2)
            if abs(v - expected) > 0.1: # Allow for small rounding errors
                # We don't raise error, just log it or mark it
                pass
        return v

class ReceiptSchema(BaseDocument):
    """Simplified schema for tickets/receipts."""
    vendor_name: str
    date: Optional[str]
    total_amount: float
    currency: str = "EUR"

def get_schema_for_type(doc_type: str):
    """Factory to get the correct schema class."""
    doc_type = doc_type.lower()
    if 'invoice' in doc_type or 'factura' in doc_type:
        return InvoiceSchema
    if 'receipt' in doc_type or 'ticket' in doc_type:
        return ReceiptSchema
    return BaseDocument
