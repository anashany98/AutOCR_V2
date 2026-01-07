import os
import sys
from pathlib import Path

# Fix path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.tasks import huey
from huey.consumer import Consumer

if __name__ == '__main__':
    print("Starting AutoOCR Huey Worker...")
    print("Press Ctrl+C to stop.")
    
    # Configure consumer
    consumer = huey.create_consumer(
        workers=2,
        periodic=True
    )
    consumer.run()
