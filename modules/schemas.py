from typing import List, Optional, Any
from pydantic import BaseModel, Field, field_validator
from datetime import date as date_type, datetime

class DocumentUpdateSchema(BaseModel):
    """Schema for validating document update requests via API."""
    
    filename: Optional[str] = Field(None, min_length=1, strip_whitespace=True)
    date: Optional[str] = Field(None, description="ISO 8601 date string (YYYY-MM-DD)")
    total: Optional[float] = Field(None, ge=0)
    supplier: Optional[str] = Field(None, strip_whitespace=True)
    type: Optional[str] = Field(None, strip_whitespace=True)
    tags: List[str] = Field(default_factory=list)
    
    # Optional JSON fields if we want to update blocks/tables directly
    notes: Optional[str] = Field(None)

    @field_validator('date')
    @classmethod
    def validate_date_format(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            return None
        try:
            # Try parsing to ensure it's a valid date, but return string for DB
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")
