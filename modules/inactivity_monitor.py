"""
Inactivity monitor for AutOCR.

In automatic mode the AutOCR system should wait until scanning is complete
before beginning the post‑processing workflow.  Rather than requiring user
interaction this monitor watches for filesystem changes in the input
folder and only returns once no new files have appeared for a specified
number of minutes.  A simple polling mechanism is used instead of OS
specific file system events for cross‑platform compatibility.
"""

from __future__ import annotations

import os
import time
from typing import Optional


class InactivityMonitor:
    """Monitor a folder and wait until no new files have been added."""

    def __init__(self, folder: str, inactivity_minutes: int = 10, poll_interval: int = 10) -> None:
        self.folder = folder
        self.inactivity_seconds = max(0, inactivity_minutes) * 60
        self.poll_interval = poll_interval

    def wait(self) -> None:
        """Block until the folder has been idle for the configured period."""
        last_change: float = time.time()
        try:
            previous_snapshot = set(os.listdir(self.folder))
        except FileNotFoundError:
            previous_snapshot = set()
        while True:
            time.sleep(self.poll_interval)
            try:
                current_snapshot = set(os.listdir(self.folder))
            except FileNotFoundError:
                current_snapshot = set()
            # If the set of files has changed, reset the timer
            if current_snapshot != previous_snapshot:
                last_change = time.time()
                previous_snapshot = current_snapshot
            # Check if inactivity threshold has been exceeded
            if (time.time() - last_change) >= self.inactivity_seconds:
                break