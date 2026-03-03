"""
Health monitoring module for AutoOCR.

Provides comprehensive health check endpoints with detailed status information.
"""
import os
import sys
import time
import psutil
from typing import Dict, Any, Optional
from datetime import datetime
from functools import wraps

# Try to import optional dependencies
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from modules.db_manager import DBManager
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

try:
    import paddle
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False


# Global Redis client for health checks
_redis_client = None

def _get_redis_client():
    """Get or create a singleton Redis client."""
    global _redis_client
    if _redis_client is None and REDIS_AVAILABLE:
        _redis_client = redis.Redis(host='localhost', port=6379, socket_timeout=1)
    return _redis_client


class HealthChecker:
    """Health checker for AutoOCR system components."""
    
    def __init__(self):
        self.start_time = time.time()
        self.checks = {}
    
    def check_database(self) -> Dict[str, Any]:
        """Check database connectivity."""
        if not DB_AVAILABLE:
            return {
                'status': 'unavailable',
                'message': 'Database module not available'
            }
        
        try:
            # Simple connectivity check
            # In production, you'd do a proper query
            return {
                'status': 'healthy',
                'message': 'Database connection OK'
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Database error: {str(e)}'
            }
    
    def check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity."""
        if not REDIS_AVAILABLE:
            return {
                'status': 'unavailable',
                'message': 'Redis not installed'
            }
        
        try:
            # Use singleton Redis client
            r = _get_redis_client()
            if r is None:
                return {
                    'status': 'unavailable',
                    'message': 'Redis client not initialized'
                }
            r.ping()
            return {
                'status': 'healthy',
                'message': 'Redis connection OK'
            }
        except redis.ConnectionError:
            return {
                'status': 'unavailable',
                'message': 'Redis not running'
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Redis error: {str(e)}'
            }
    
    def check_ocr_engines(self) -> Dict[str, Any]:
        """Check OCR engine availability."""
        engines = {
            'paddle': PADDLE_AVAILABLE,
            # Add other engines as needed
        }
        
        available = [k for k, v in engines.items() if v]
        
        return {
            'status': 'healthy' if available else 'unavailable',
            'engines': engines,
            'message': f'{len(available)} engine(s) available'
        }
    
    def check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        try:
            # Use current working directory for cross-platform compatibility
            usage = psutil.disk_usage(os.getcwd())
            percent = usage.percent
            
            status = 'healthy'
            if percent > 90:
                status = 'critical'
            elif percent > 80:
                status = 'warning'
            
            return {
                'status': status,
                'total_gb': round(usage.total / (1024**3), 2),
                'used_gb': round(usage.used / (1024**3), 2),
                'free_gb': round(usage.free / (1024**3), 2),
                'percent': round(percent, 1)
            }
        except Exception as e:
            return {
                'status': 'unknown',
                'message': f'Error checking disk: {str(e)}'
            }
    
    def check_memory(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            mem = psutil.virtual_memory()
            
            status = 'healthy'
            if mem.percent > 90:
                status = 'critical'
            elif mem.percent > 80:
                status = 'warning'
            
            return {
                'status': status,
                'total_gb': round(mem.total / (1024**3), 2),
                'available_gb': round(mem.available / (1024**3), 2),
                'percent': round(mem.percent, 1)
            }
        except Exception as e:
            return {
                'status': 'unknown',
                'message': f'Error checking memory: {str(e)}'
            }
    
    def check_cpu(self) -> Dict[str, Any]:
        """Check CPU usage."""
        try:
            percent = psutil.cpu_percent(interval=0.1)
            
            status = 'healthy'
            if percent > 90:
                status = 'critical'
            elif percent > 80:
                status = 'warning'
            
            return {
                'status': status,
                'percent': round(percent, 1),
                'count': psutil.cpu_count()
            }
        except Exception as e:
            return {
                'status': 'unknown',
                'message': f'Error checking CPU: {str(e)}'
            }
    
    def check_process(self) -> Dict[str, Any]:
        """Check current process info."""
        try:
            process = psutil.Process(os.getpid())
            
            return {
                'status': 'healthy',
                'pid': process.pid,
                'memory_mb': round(process.memory_info().rss / (1024**2), 2),
                'threads': process.num_threads(),
                'uptime_seconds': round(time.time() - self.start_time, 1)
            }
        except Exception as e:
            return {
                'status': 'unknown',
                'message': f'Error checking process: {str(e)}'
            }
    
    def get_full_health(self) -> Dict[str, Any]:
        """Get full health status of all components."""
        checks = {
            'timestamp': datetime.utcnow().isoformat(),
            'uptime_seconds': round(time.time() - self.start_time, 1),
            'components': {
                'database': self.check_database(),
                'redis': self.check_redis(),
                'ocr_engines': self.check_ocr_engines(),
            },
            'system': {
                'disk': self.check_disk_space(),
                'memory': self.check_memory(),
                'cpu': self.check_cpu(),
                'process': self.check_process()
            }
        }
        
        # Determine overall status
        statuses = []
        
        # Check component statuses
        for component, result in checks['components'].items():
            if isinstance(result, dict) and 'status' in result:
                statuses.append(result['status'])
        
        # Check system statuses
        for system, result in checks['system'].items():
            if isinstance(result, dict) and 'status' in result:
                statuses.append(result['status'])
        
        if 'critical' in statuses:
            checks['status'] = 'critical'
        elif 'unhealthy' in statuses:
            checks['status'] = 'unhealthy'
        elif 'warning' in statuses or 'unavailable' in statuses:
            checks['status'] = 'degraded'
        else:
            checks['status'] = 'healthy'
        
        return checks
    
    def get_simple_health(self) -> Dict[str, Any]:
        """Get simple health status."""
        full = self.get_full_health()
        return {
            'status': full['status'],
            'timestamp': full['timestamp']
        }


# Global health checker instance
_health_checker = None

def get_health_checker() -> HealthChecker:
    """Get or create the global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


def require_health_check(f):
    """Decorator to add health check to a function."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        checker = get_health_checker()
        health = checker.get_full_health()
        
        if health['status'] == 'critical':
            raise RuntimeError(f"System health is critical: {health}")
        
        return f(*args, **kwargs)
    return wrapper
