"""
Metrics module for AutoOCR.

Provides Prometheus-style metrics for monitoring.
"""
import time
import bisect
import threading
from typing import Dict, Optional, Callable
from collections import defaultdict
from functools import wraps


class Counter:
    """Simple counter metric."""
    
    def __init__(self, name: str, description: str = "", labels: Optional[Dict] = None):
        self.name = name
        self.description = description
        self.labels = labels or {}
        self._value = 0
        self._lock = threading.Lock()
    
    def inc(self, amount: int = 1):
        """Increment counter."""
        with self._lock:
            self._value += amount
    
    def get_value(self) -> int:
        """Get current value."""
        with self._lock:
            return self._value
    
    def __str__(self):
        return f"Counter({self.name}={self.get_value()})"


class Gauge:
    """Simple gauge metric."""
    
    def __init__(self, name: str, description: str = "", labels: Optional[Dict] = None):
        self.name = name
        self.description = description
        self.labels = labels or {}
        self._value = 0.0
        self._lock = threading.Lock()
    
    def inc(self, amount: float = 1.0):
        """Increment gauge."""
        with self._lock:
            self._value += amount
    
    def dec(self, amount: float = 1.0):
        """Decrement gauge."""
        with self._lock:
            self._value -= amount
    
    def set(self, value: float):
        """Set gauge value."""
        with self._lock:
            self._value = value
    
    def get_value(self) -> float:
        """Get current value."""
        with self._lock:
            return self._value
    
    def __str__(self):
        return f"Gauge({self.name}={self.get_value()})"


class Histogram:
    """Simple histogram metric."""
    
    def __init__(self, name: str, description: str = "", 
                 buckets: Optional[list] = None, labels: Optional[Dict] = None):
        self.name = name
        self.description = description
        self.labels = labels or {}
        self.buckets = buckets or [0.1, 0.5, 1.0, 5.0, 10.0, float('inf')]
        self._values = []
        self._lock = threading.Lock()
    
    def observe(self, value: float):
        """Observe a value."""
        with self._lock:
            self._values.append(value)
    
    def get_count(self) -> int:
        """Get total observation count."""
        with self._lock:
            return len(self._values)
    
    def get_sum(self) -> float:
        """Get sum of all observations."""
        with self._lock:
            return sum(self._values)
    
    def get_buckets(self) -> Dict[float, int]:
        """Get bucket counts using binary search for efficiency."""
        with self._lock:
            if not self._values:
                return {bucket: 0 for bucket in self.buckets}
            
            # Sort values once for binary search
            sorted_values = sorted(self._values)
            result = {}
            
            for bucket in self.buckets:
                # Use bisect to find count of values <= bucket (O(log n))
                count = bisect.bisect_right(sorted_values, bucket)
                result[bucket] = count
            
            return result
    
    def __str__(self):
        return f"Histogram({self.name}, count={self.get_count()})"


class MetricsRegistry:
    """Central metrics registry."""
    
    def __init__(self):
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._lock = threading.Lock()
    
    def counter(self, name: str, description: str = "", 
                labels: Optional[Dict] = None) -> Counter:
        """Get or create a counter."""
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(name, description, labels)
            return self._counters[name]
    
    def gauge(self, name: str, description: str = "",
              labels: Optional[Dict] = None) -> Gauge:
        """Get or create a gauge."""
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(name, description, labels)
            return self._gauges[name]
    
    def histogram(self, name: str, description: str = "",
                  buckets: Optional[list] = None,
                  labels: Optional[Dict] = None) -> Histogram:
        """Get or create a histogram."""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(name, description, buckets, labels)
            return self._histograms[name]
    
    def get_all_metrics(self) -> Dict:
        """Get all metrics in Prometheus format."""
        with self._lock:
            result = {
                'counters': {},
                'gauges': {},
                'histograms': {}
            }
            
            for name, counter in self._counters.items():
                result['counters'][name] = {
                    'value': counter.get_value(),
                    'labels': counter.labels
                }
            
            for name, gauge in self._gauges.items():
                result['gauges'][name] = {
                    'value': gauge.get_value(),
                    'labels': gauge.labels
                }
            
            for name, histogram in self._histograms.items():
                result['histograms'][name] = {
                    'count': histogram.get_count(),
                    'sum': histogram.get_sum(),
                    'buckets': histogram.get_buckets(),
                    'labels': histogram.labels
                }
            
            return result
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        for name, counter in self._counters.items():
            lines.append(f"# TYPE {name} counter")
            if counter.description:
                lines.append(f"# HELP {name} {counter.description}")
            lines.append(f"{name} {counter.get_value()}")
        
        for name, gauge in self._gauges.items():
            lines.append(f"# TYPE {name} gauge")
            if gauge.description:
                lines.append(f"# HELP {name} {gauge.description}")
            lines.append(f"{name} {gauge.get_value()}")
        
        for name, histogram in self._histograms.items():
            lines.append(f"# TYPE {name} histogram")
            if histogram.description:
                lines.append(f"# HELP {name} {histogram.description}")
            
            buckets = histogram.get_buckets()
            for bucket, count in buckets.items():
                bucket_label = f'{name}_bucket{{le="{bucket}"}}'
                lines.append(f"{bucket_label} {count}")
            
            lines.append(f"{name}_count {histogram.get_count()}")
            lines.append(f"{name}_sum {histogram.get_sum()}")
        
        return '\n'.join(lines)


# Global registry
_registry = None

def get_metrics_registry() -> MetricsRegistry:
    """Get the global metrics registry."""
    global _registry
    if _registry is None:
        _registry = MetricsRegistry()
    return _registry


# Predefined metrics
def get_document_processing_counter() -> Counter:
    """Get document processing counter."""
    return get_metrics_registry().counter(
        'autocr_documents_processed_total',
        'Total number of documents processed',
        {'type': 'counter'}
    )

def get_ocr_duration_histogram() -> Histogram:
    """Get OCR processing duration histogram."""
    return get_metrics_registry().histogram(
        'autocr_ocr_duration_seconds',
        'OCR processing duration in seconds',
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, float('inf')]
    )

def get_active_jobs_gauge() -> Gauge:
    """Get active jobs gauge."""
    return get_metrics_registry().gauge(
        'autocr_active_jobs',
        'Number of active processing jobs'
    )

def get_error_counter() -> Counter:
    """Get error counter."""
    return get_metrics_registry().counter(
        'autocr_errors_total',
        'Total number of errors',
        {'type': 'counter'}
    )


def track_duration(histogram_name: str):
    """Decorator to track function duration."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            registry = get_metrics_registry()
            histogram = registry.histogram(
                f'autocr_{histogram_name}_duration_seconds',
                f'Duration of {func.__name__}'
            )
            
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start
                histogram.observe(duration)
        
        return wrapper
    return decorator
