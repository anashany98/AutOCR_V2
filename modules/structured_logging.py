"""
Structured JSON logging for AutoOCR.

Provides JSON-formatted logging for better log analysis and monitoring.
Includes request correlation for distributed tracing.
"""
import json
import logging
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, Optional
from logging.handlers import RotatingFileHandler
from pathlib import Path
import contextvars

# Request correlation context
_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar('request_id', default='')
_trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar('trace_id', default='')


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if self.include_extra:
            extra_fields = {
                k: v for k, v in record.__dict__.items()
                if k not in (
                    'name', 'msg', 'args', 'created', 'filename',
                    'funcName', 'levelname', 'levelno', 'lineno',
                    'module', 'msecs', 'pathname', 'process',
                    'processName', 'relativeCreated', 'thread',
                    'threadName', 'exc_info', 'exc_text', 'stack_info'
                )
            }
            if extra_fields:
                log_data['extra'] = extra_fields
        
        return json.dumps(log_data)


class StructuredLogger:
    """Structured logger with JSON output."""
    
    def __init__(self, name: str, log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler with JSON formatter
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(JSONFormatter())
            self.logger.addHandler(file_handler)
    
    def log(self, level: int, message: str, **kwargs):
        """Log a message with extra fields."""
        extra = kwargs if kwargs else None
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.log(logging.CRITICAL, message, **kwargs)


class AuditLogger:
    """Specialized audit logger for security events."""
    
    def __init__(self, log_file: Optional[str] = None):
        self.logger = logging.getLogger('audit')
        self.logger.setLevel(logging.INFO)
        
        # Use JSON formatter
        formatter = JSONFormatter()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,
                backupCount=10
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def log_event(self, event_type: str, user: Optional[str] = None, 
                  resource: Optional[str] = None, action: Optional[str] = None,
                  result: str = 'success', **kwargs):
        """Log an audit event."""
        event_data = {
            'event_type': event_type,
            'user': user or 'anonymous',
            'resource': resource or 'N/A',
            'action': action or 'N/A',
            'result': result
        }
        event_data.update(kwargs)
        
        self.logger.info(f"Audit: {event_type}", extra=event_data)
    
    def login(self, user: str, result: str = 'success', **kwargs):
        """Log login attempt."""
        self.log_event('LOGIN', user=user, action='login', result=result, **kwargs)
    
    def logout(self, user: str, **kwargs):
        """Log logout."""
        self.log_event('LOGOUT', user=user, action='logout', **kwargs)
    
    def upload(self, user: str, filename: str, size: int, **kwargs):
        """Log file upload."""
        self.log_event('UPLOAD', user=user, resource=filename, 
                      action='upload', size=size, **kwargs)
    
    def download(self, user: str, filename: str, **kwargs):
        """Log file download."""
        self.log_event('DOWNLOAD', user=user, resource=filename,
                      action='download', **kwargs)
    
    def delete(self, user: str, resource: str, **kwargs):
        """Log delete action."""
        self.log_event('DELETE', user=user, resource=resource,
                      action='delete', **kwargs)
    
    def permission_change(self, user: str, target_user: str, 
                         permission: str, **kwargs):
        """Log permission change."""
        self.log_event('PERMISSION_CHANGE', user=user, 
                      resource=target_user, action='permission_change',
                      permission=permission, **kwargs)


# Global logger instances
_structured_logger = None
_audit_logger = None

def get_structured_logger(name: str = 'autocr', 
                          log_file: Optional[str] = None) -> StructuredLogger:
    """Get or create a structured logger."""
    global _structured_logger
    if _structured_logger is None:
        _structured_logger = StructuredLogger(name, log_file)
    return _structured_logger

def get_audit_logger(log_file: Optional[str] = None) -> AuditLogger:
    """Get or create the audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(log_file)
    return _audit_logger


# Request correlation functions
def get_request_id() -> str:
    """Get current request ID or generate new one."""
    req_id = _request_id_var.get()
    if not req_id:
        req_id = str(uuid.uuid4())
        _request_id_var.set(req_id)
    return req_id

def get_trace_id() -> str:
    """Get current trace ID or generate new one."""
    trace_id = _trace_id_var.get()
    if not trace_id:
        trace_id = str(uuid.uuid4())
        _trace_id_var.set(trace_id)
    return trace_id

def set_request_id(request_id: str) -> None:
    """Set request ID for current context."""
    _request_id_var.set(request_id)

def set_trace_id(trace_id: str) -> None:
    """Set trace ID for current context."""
    _trace_id_var.set(trace_id)


class CorrelationLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds correlation IDs to log records."""
    
    def process(self, msg, kwargs):
        # Add correlation IDs to extra
        extra = kwargs.get('extra', {})
        extra['request_id'] = get_request_id()
        extra['trace_id'] = get_trace_id()
        kwargs['extra'] = extra
        return msg, kwargs


def get_correlation_logger(name: str) -> CorrelationLoggerAdapter:
    """Get a logger with correlation ID support."""
    logger = logging.getLogger(name)
    return CorrelationLoggerAdapter(logger, {})
