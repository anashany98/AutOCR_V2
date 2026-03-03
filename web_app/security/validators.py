"""
Input validation and security utilities for AutoOCR.
"""
import os
import re
import mimetypes
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from werkzeug.datastructures import FileStorage


# Allowed file extensions and MIME types
ALLOWED_EXTENSIONS = {
    'pdf': ['application/pdf'],
    'tif': ['image/tiff', 'image/x-tiff'],
    'tiff': ['image/tiff', 'image/x-tiff'],
    'jpg': ['image/jpeg'],
    'jpeg': ['image/jpeg'],
    'png': ['image/png'],
    'bmp': ['image/bmp'],
    'gif': ['image/gif'],
    'docx': ['application/vnd.openxmlformats-officedocument.wordprocessingml.document'],
    'xlsx': ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'],
    'xlsm': ['application/vnd.ms-excel.sheet.macroEnabled.12'],
    'csv': ['text/csv', 'application/csv'],
    'txt': ['text/plain'],
    'json': ['application/json'],
    'eml': ['message/rfc822', 'application/eml'],
    'webp': ['image/webp'],
    'jfif': ['image/jpeg'],
    'avif': ['image/avif'],
}

# Maximum file sizes (in bytes)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
MAX_BATCH_SIZE = 100 * 1024 * 1024  # 100 MB total for batch

# Dangerous file extensions (should never be allowed)
DANGEROUS_EXTENSIONS = {
    'exe', 'bat', 'cmd', 'com', 'msi', 'dll',
    'sh', 'bash', 'zsh', 'fish',
    'rb', 'php', 'pl', 'cgi',
    'html', 'htm', 'xml',
    'sql', 'db', 'sqlite',
    'jar', 'war', 'ear',
    'ps1', 'psm1', 'vbs'
}

# Banned MIME types
BANNED_MIME_TYPES = {
    'application/x-msdownload',
    'application/x-executable',
    'application/x-sh',
    'application/x-shellscript',
    'text/x-python',
    'text/x-java',
}


class ValidationError(Exception):
    """Custom validation error."""
    pass


class FileValidator:
    """Validator for file uploads."""
    
    def __init__(self, allowed_extensions: Optional[set] = None):
        self.allowed_extensions = allowed_extensions or set(ALLOWED_EXTENSIONS.keys())
    
    def validate_file(self, file: FileStorage) -> Tuple[bool, Optional[str]]:
        """
        Validate a file upload.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if file exists
        if not file:
            return False, "No file provided"
        
        # Check filename
        filename = file.filename
        if not filename:
            return False, "No filename provided"
        
        # Check for path traversal and null bytes
        if '..' in filename or '/' in filename or '\\' in filename or '\x00' in filename:
            return False, "Invalid filename"
        
        # Get file extension
        ext = self._get_extension(filename)
        if not ext:
            return False, "No file extension"
        
        ext = ext.lower()
        
        # Check for dangerous extensions
        if ext in DANGEROUS_EXTENSIONS:
            return False, f"File type not allowed: {ext}"
        
        # Check allowed extensions
        if ext not in self.allowed_extensions:
            return False, f"Extension not allowed: {ext}"
        
        # Check file size (seek to end to get size, then reset)
        file.seek(0, 2)  # Seek to end
        size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if size > MAX_FILE_SIZE:
            return False, f"File too large: {size} bytes (max: {MAX_FILE_SIZE})"
        
        if size == 0:
            return False, "File is empty"
        
        # Check content type
        content_type = file.content_type
        if content_type and content_type != 'application/octet-stream':
            allowed_types = ALLOWED_EXTENSIONS.get(ext, [])
            if allowed_types and content_type not in allowed_types:
                return False, f"Content type mismatch: {content_type}"
        
        # Check MIME type ban list
        if content_type in BANNED_MIME_TYPES:
            return False, f"MIME type not allowed: {content_type}"
        
        return True, None
    
    def validate_file_size(self, file_size: int, max_size: int = MAX_FILE_SIZE) -> Tuple[bool, Optional[str]]:
        """Validate file size."""
        if file_size > max_size:
            return False, f"File too large: {file_size} bytes (max: {max_size})"
        return True, None
    
    def validate_batch(self, files: List[FileStorage]) -> Tuple[bool, Optional[str]]:
        """Validate a batch of files."""
        if not files:
            return False, "No files provided"
        
        total_size = 0
        for file in files:
            is_valid, error = self.validate_file(file)
            if not is_valid:
                return False, f"Invalid file {file.filename}: {error}"
            
            # Get file size
            file.seek(0, 2)  # Seek to end
            size = file.tell()
            file.seek(0)  # Reset to beginning
            total_size += size
            
            if total_size > MAX_BATCH_SIZE:
                return False, f"Batch too large: {total_size} bytes (max: {MAX_BATCH_SIZE})"
        
        return True, None
    
    def _get_extension(self, filename: str) -> Optional[str]:
        """Get file extension from filename."""
        if '.' in filename:
            return filename.rsplit('.', 1)[1].lower()
        return None


class InputValidator:
    """Validator for general input parameters."""
    
    # Patterns for validation
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    ALPHANUMERIC_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
    UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')
    
    def validate_email(self, email: str) -> Tuple[bool, Optional[str]]:
        """Validate email address."""
        if not email:
            return False, "Email is required"
        
        if not self.EMAIL_PATTERN.match(email):
            return False, "Invalid email format"
        
        return True, None
    
    def validate_string(self, value: str, min_length: int = 0, 
                       max_length: int = 1000, pattern: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """Validate string input."""
        if not isinstance(value, str):
            return False, "Value must be a string"
        
        if len(value) < min_length:
            return False, f"String too short (min: {min_length})"
        
        if len(value) > max_length:
            return False, f"String too long (max: {max_length})"
        
        if pattern and not re.match(pattern, value):
            return False, f"String doesn't match required pattern"
        
        return True, None
    
    def validate_integer(self, value: Any, min_value: Optional[int] = None,
                        max_value: Optional[int] = None) -> Tuple[bool, Optional[str]]:
        """Validate integer input."""
        try:
            int_value = int(value)
        except (ValueError, TypeError):
            return False, "Value must be an integer"
        
        if min_value is not None and int_value < min_value:
            return False, f"Value too small (min: {min_value})"
        
        if max_value is not None and int_value > max_value:
            return False, f"Value too large (max: {max_value})"
        
        return True, None
    
    def validate_uuid(self, uuid_str: str) -> Tuple[bool, Optional[str]]:
        """Validate UUID string."""
        if not uuid_str:
            return False, "UUID is required"
        
        if not self.UUID_PATTERN.match(uuid_str):
            return False, "Invalid UUID format"
        
        return True, None
    
    def validate_tenant_id(self, tenant_id: str) -> Tuple[bool, Optional[str]]:
        """Validate tenant ID."""
        if not tenant_id:
            return False, "Tenant ID is required"
        
        if not self.ALPHANUMERIC_PATTERN.match(tenant_id):
            return False, "Invalid tenant ID format"
        
        return True, None
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize a filename for safe storage."""
        # Remove dangerous characters
        filename = re.sub(r'[^\w\s.-]', '', filename)
        
        # Replace spaces with underscores
        filename = filename.replace(' ', '_')
        
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            name = name[:255 - len(ext) - 1]
            filename = f"{name}.{ext}" if ext else name
        
        return filename


# Global instances
_file_validator = None
_input_validator = None

def get_file_validator() -> FileValidator:
    """Get the global file validator."""
    global _file_validator
    if _file_validator is None:
        _file_validator = FileValidator()
    return _file_validator

def get_input_validator() -> InputValidator:
    """Get the global input validator."""
    global _input_validator
    if _input_validator is None:
        _input_validator = InputValidator()
    return _input_validator
