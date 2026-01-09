import os
import sys
from pathlib import Path
print("DEBUG: serve.py loaded", flush=True)

# CRITICAL: Windows GPU environment setup must happen before any other imports
if os.name == "nt":
    cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
    if os.path.exists(cuda_bin):
        os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(cuda_bin)

            os.add_dll_directory(cuda_bin)

print("DEBUG: Importing waitress", flush=True)
try:
    from waitress import serve
except ImportError:
    serve = None
print("DEBUG: Importing web_app", flush=True)
from web_app.app import app, PROJECT_ROOT, get_logger
print("DEBUG: Imports complete", flush=True)
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

def kill_zombies(port=8000):
    """Encuentra y elimina procesos escuchando en el puerto dado."""
    import psutil
    current_pid = os.getpid()
    killed = False
    
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.connections(kind='inet'):
                if conn.laddr.port == port:
                    if proc.pid != current_pid:
                        try:
                            print(f"⚠️  Matando proceso zombie (PID: {proc.pid}) ocupando puerto {port}...")
                            proc.terminate()
                            proc.wait(timeout=3)
                            killed = True
                        except psutil.TimeoutExpired:
                            proc.kill()
                            killed = True
                        except Exception as e:
                            print(f"Error matando proceso {proc.pid}: {e}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    if killed:
        print(f"✅  Puerto {port} liberado.")

if __name__ == "__main__":
    print("DEBUG: Entering main", flush=True)
    # Check for Development Mode check early
    is_debug = os.environ.get("FLASK_DEBUG", "0") == "1" or os.environ.get("FLASK_ENV") == "development"

    if not is_debug:
        try:
            kill_zombies(8000)
        except Exception as e:
            print(f"WARNING: kill_zombies failed: {e}", flush=True)
    else:
        print("DEBUG: kill_zombies skipped in dev/hot-reload mode to prevent self-termination.", flush=True)

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
    
    # Start Huey Consumer in a separate thread
    from modules.tasks import huey
    # Setting worker_type='thread' is cleaner for Windows integration here, 
    # but 'process' is default. 'thread' avoids pickling issues with some objects if env is complex.
    # However, standard huey consumer usually runs in main process or spawns subprocesses.
    # We will use a simple wrapper to run the consumer loop.
    import threading
    
    def run_consumer():
        try:
            get_logger().info("Starting Background Worker...")
            # Disable signal handling for thread compatibility
            import signal
            original_signal = signal.signal
            signal.signal = lambda s, h: None
            
            consumer = huey.create_consumer(workers=2, worker_type='thread') 
            consumer.run()
        except Exception as e:
            get_logger().error(f"Worker failed: {e}")

    consumer_thread = threading.Thread(target=run_consumer, daemon=True)
    consumer_thread.start()
    
    # Initialize Web App (Lazy Load)
    from web_app.app import init_app
    init_app()

    # Production settings:
    # - threads=6: Handle multiple concurrent requests (DB pool is size 5 by default)
    # - url_scheme='http': Standard for local network
    # Check for Development Mode (Hot Reload)
    is_debug = os.environ.get("FLASK_DEBUG", "0") == "1" or os.environ.get("FLASK_ENV") == "development"

    if is_debug:
        logger.warning("⚠️  DEVELOPMENT MODE ENABLED: Using Flask Dev Server with Hot Reload")
        print("⚠️  HOT RELOAD IS ACTIVE. Changes to files will restart the server.")
        app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=True)
    else:
        try:
            from waitress import serve
            serve(app, host="0.0.0.0", port=8000, threads=6)
        except ImportError:
            logger.warning("Waitress not found. Falling back to Flask Development Server.")
            app.run(host="0.0.0.0", port=8000, debug=True)
