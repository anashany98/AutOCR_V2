import time
import threading
import logging
from pathlib import Path
from typing import Optional, List, Callable

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent

logger = logging.getLogger(__name__)

class AutoOCRHandler(FileSystemEventHandler):
    """Handles file system events for the hot folder."""

    def __init__(self, callback: Callable[[Path], None], extensions: List[str]):
        self.callback = callback
        self.extensions = {ext.lower() for ext in extensions}
        self.processing_lock = threading.Lock()

    def on_created(self, event):
        if event.is_directory:
            return

        path = Path(event.src_path)
        if path.suffix.lower() not in self.extensions:
            return

        logger.info(f"New file detected in hot folder: {path.name}")
        
        # Debounce/Wait for file copy completion
        threading.Thread(target=self._process_with_delay, args=(path,), daemon=True).start()

    def _process_with_delay(self, path: Path):
        """Wait for file to be fully written before processing."""
        time.sleep(2)  # Initial wait
        
        # Simple check: Try to open the file in append mode to see if it's locked
        retries = 5
        while retries > 0:
            try:
                if not path.exists():
                    return # File removed
                
                # If we can open it, it's likely done copying
                with open(path, "ab"):
                    pass
                break
            except OSError:
                time.sleep(1)
                retries -= 1
        
        if retries == 0:
            logger.warning(f"Timeout waiting for file unlock: {path.name}")
            return

        try:
            with self.processing_lock:
                 self.callback(path)
        except Exception as e:
            logger.error(f"Error processing hot folder file {path.name}: {e}")


class FolderWatcher:
    """Manages the background observer for hot folders."""

    def __init__(self, watch_dir: str, callback: Callable[[Path], None], extensions: List[str] = None):
        if not extensions:
            extensions = [".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
            
        self.watch_dir = watch_dir
        self.callback = callback
        self.extensions = extensions
        self.observer: Optional[Observer] = None
        self.handler: Optional[AutoOCRHandler] = None

    def start(self):
        """Start monitoring the directory."""
        if self.observer and self.observer.is_alive():
            return

        if not Path(self.watch_dir).exists():
            logger.error(f"Watch directory does not exist: {self.watch_dir}")
            return

        self.handler = AutoOCRHandler(self.callback, self.extensions)
        self.observer = Observer()
        self.observer.schedule(self.handler, self.watch_dir, recursive=False)
        self.observer.start()
        logger.info(f"Hot folder watcher started for: {self.watch_dir}")

    def stop(self):
        """Stop monitoring."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            logger.info("Hot folder watcher stopped.")
