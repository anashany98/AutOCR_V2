"""
Webhooks module for AutoOCR.

Provides webhook functionality for event notifications.
"""
import json
import logging
import time
import hmac
import hashlib
import requests
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from threading import Thread
from queue import Queue
from queue import Empty

# Configure logger
logger = logging.getLogger(__name__)


class WebhookEventType(str, Enum):
    """Types of webhook events."""
    DOCUMENT_UPLOADED = "document.uploaded"
    DOCUMENT_PROCESSED = "document.processed"
    DOCUMENT_FAILED = "document.failed"
    DOCUMENT_DELETED = "document.deleted"
    CLASSIFICATION_COMPLETE = "classification.complete"
    OCR_COMPLETE = "ocr.complete"
    EXTRACTION_COMPLETE = "extraction.complete"
    USER_LOGIN = "user.login"
    USER_LOGOUT = "user.logout"
    TENANT_CREATED = "tenant.created"
    TENANT_DELETED = "tenant.deleted"


@dataclass
class WebhookEvent:
    """Represents a webhook event."""
    event_type: WebhookEventType
    payload: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + 'Z')
    event_id: str = field(default_factory=lambda: f"{int(time.time())}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}")
    retry_count: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp,
            'payload': self.payload,
            'retry_count': self.retry_count
        }


@dataclass 
class WebhookConfig:
    """Configuration for a webhook."""
    url: str
    secret: str
    events: List[WebhookEventType]
    enabled: bool = True
    timeout: int = 30
    retry_limit: int = 3
    retry_delay: int = 60  # seconds


class WebhookSender:
    """Handles sending webhook events."""
    
    def __init__(self, config: WebhookConfig):
        self.config = config
    
    def _generate_signature(self, payload: str) -> str:
        """Generate HMAC signature for payload."""
        return hmac.new(
            self.config.secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def send(self, event: WebhookEvent) -> bool:
        """Send a webhook event."""
        if not self.config.enabled:
            return False
        
        payload = json.dumps(event.to_dict())
        signature = self._generate_signature(payload)
        
        headers = {
            'Content-Type': 'application/json',
            'X-Webhook-Signature': signature,
            'X-Webhook-Event': event.event_type.value,
            'X-Webhook-Event-Id': event.event_id,
            'User-Agent': 'AutoOCR-Webhook/1.0'
        }
        
        try:
            response = requests.post(
                self.config.url,
                data=payload,
                headers=headers,
                timeout=self.config.timeout
            )
            
            return 200 <= response.status_code < 300
            
        except requests.RequestException as e:
            logger.error(f"Webhook delivery failed: {e}")
            return False


class WebhookQueue:
    """Queue for managing webhook events."""
    
    def __init__(self, max_size: int = 1000):
        self.queue = Queue(maxsize=max_size)
        self.webhooks: Dict[str, WebhookSender] = {}
        self.running = False
        self.worker_thread: Optional[Thread] = None
    
    def register_webhook(self, name: str, config: WebhookConfig):
        """Register a webhook configuration."""
        self.webhooks[name] = WebhookSender(config)
    
    def unregister_webhook(self, name: str):
        """Unregister a webhook."""
        if name in self.webhooks:
            del self.webhooks[name]
    
    def enqueue(self, event: WebhookEvent):
        """Add an event to the queue."""
        try:
            self.queue.put(event, block=False)
        except Exception as e:
            logger.error(f"Failed to enqueue event: {e}")
    
    def _process_event(self, event: WebhookEvent):
        """Process a single event."""
        # Find matching webhooks
        for name, sender in self.webhooks.items():
            # Check if webhook is interested in this event type
            if event.event_type in sender.config.events:
                success = sender.send(event)
                
                if not success and event.retry_count < sender.config.retry_limit:
                    # Schedule retry asynchronously instead of blocking
                    event.retry_count += 1
                    retry_delay = sender.config.retry_delay
                    
                    def retry_later(e=event, delay=retry_delay, sender_name=name):
                        import threading
                        timer = threading.Timer(delay, self._retry_event, args=(e, sender_name))
                        timer.daemon = True
                        timer.start()
                    
                    retry_later()
    
    def _retry_event(self, event: WebhookEvent, sender_name: str):
        """Retry a failed event."""
        if sender_name not in self.webhooks:
            return
        
        sender = self.webhooks[sender_name]
        if event.event_type in sender.config.events:
            sender.send(event)
    
    def _worker(self):
        """Worker thread for processing events."""
        while self.running:
            try:
                event = self.queue.get(timeout=1)
                self._process_event(event)
                self.queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing webhook event: {e}")
    
    def start(self):
        """Start the webhook queue worker."""
        if not self.running:
            self.running = True
            self.worker_thread = Thread(target=self._worker, daemon=True)
            self.worker_thread.start()
    
    def stop(self):
        """Stop the webhook queue worker."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
    
    def get_status(self) -> Dict:
        """Get queue status."""
        return {
            'queue_size': self.queue.qsize(),
            'webhooks_count': len(self.webhooks),
            'running': self.running
        }


# Global webhook queue
_webhook_queue: Optional[WebhookQueue] = None

def get_webhook_queue() -> WebhookQueue:
    """Get the global webhook queue."""
    global _webhook_queue
    if _webhook_queue is None:
        _webhook_queue = WebhookQueue()
    return _webhook_queue


def trigger_webhook(event_type: WebhookEventType, payload: Dict[str, Any]):
    """Trigger a webhook event."""
    event = WebhookEvent(
        event_type=event_type,
        payload=payload
    )
    get_webhook_queue().enqueue(event)


# Convenience functions for common events
def webhook_document_processed(document_id: int, filename: str, 
                                document_type: str, confidence: float):
    """Trigger document processed webhook."""
    trigger_webhook(
        WebhookEventType.DOCUMENT_PROCESSED,
        {
            'document_id': document_id,
            'filename': filename,
            'document_type': document_type,
            'confidence': confidence
        }
    )

def webhook_document_failed(document_id: int, filename: str, error: str):
    """Trigger document failed webhook."""
    trigger_webhook(
        WebhookEventType.DOCUMENT_FAILED,
        {
            'document_id': document_id,
            'filename': filename,
            'error': error
        }
    )

def webhook_user_login(user_id: str, username: str, ip_address: str):
    """Trigger user login webhook."""
    trigger_webhook(
        WebhookEventType.USER_LOGIN,
        {
            'user_id': user_id,
            'username': username,
            'ip_address': ip_address
        }
    )
