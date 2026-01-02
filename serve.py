from waitress import serve
from web_app.app import app, PROJECT_ROOT, get_logger
import os
import socket

def get_ip_address():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

if __name__ == "__main__":
    logger = get_logger()
    host_ip = get_ip_address()
    
    print("\n" + "="*60)
    print(f"🚀  AutoOCR V2.5 - PRODUCTION SERVER")
    print("="*60)
    print(f"✅  Status:    ONLINE")
    print(f"🌍  Local:     http://127.0.0.1:8000")
    print(f"📡  Network:   http://{host_ip}:8000")
    print(f"📂  Root:      {PROJECT_ROOT}")
    print("="*60 + "\n")
    
    logger.info("Starting Waitress production server on 0.0.0.0:8000")
    
    # Production settings:
    # - threads=6: Handle multiple concurrent requests (DB pool is size 5 by default)
    # - url_scheme='http': Standard for local network
    serve(app, host="0.0.0.0", port=8000, threads=6)
