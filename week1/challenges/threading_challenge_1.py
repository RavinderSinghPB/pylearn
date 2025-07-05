#!/usr/bin/env python3
"""
Threading Challenge 1: Thread-Safe File Download Manager

Build a multi-threaded file download manager that can download multiple files
concurrently while maintaining thread safety and providing progress tracking.

Requirements:
1. Download multiple files from URLs concurrently (max 5 concurrent downloads)
2. Implement a thread-safe progress tracker
3. Handle download failures with automatic retry (max 3 retries per file)
4. Ensure thread-safe file writing (no corruption)
5. Provide ability to pause/resume downloads
6. Display real-time statistics in a formatted table

Time: 30-45 minutes
"""

import threading
import time
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from dataclasses import dataclass
from typing import Dict, List, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
import signal
import sys

console = Console()

@dataclass
class DownloadTask:
    """Represents a file download task"""
    url: str
    filename: str
    total_size: int = 0
    downloaded: int = 0
    retries: int = 0
    max_retries: int = 3
    status: str = "pending"  # pending, downloading, completed, failed, paused
    speed: float = 0.0

    def progress_percent(self) -> float:
        if self.total_size == 0:
            return 0.0
        return (self.downloaded / self.total_size) * 100

class ThreadSafeProgressTracker:
    """Thread-safe progress tracking for downloads"""

    def __init__(self):
        self._lock = threading.Lock()
        self._tasks: Dict[str, DownloadTask] = {}
        self._total_files = 0
        self._completed_files = 0

    def add_task(self, task: DownloadTask):
        """Add a new download task"""
        # TODO: Implement thread-safe task addition
        pass

    def update_progress(self, filename: str, downloaded: int, speed: float):
        """Update download progress for a file"""
        # TODO: Implement thread-safe progress update
        pass

    def update_status(self, filename: str, status: str):
        """Update download status for a file"""
        # TODO: Implement thread-safe status update
        pass

    def get_overall_progress(self) -> Dict:
        """Get overall download statistics"""
        # TODO: Calculate and return overall progress statistics
        pass

    def display_status(self):
        """Display current download status in a formatted table"""
        # TODO: Create and display a rich table with download status
        pass

class FileDownloadManager:
    """Multi-threaded file download manager with progress tracking"""

    def __init__(self, max_concurrent_downloads: int = 5):
        self.max_concurrent_downloads = max_concurrent_downloads
        self.progress_tracker = ThreadSafeProgressTracker()
        self._shutdown_event = threading.Event()
        self._download_queue = Queue()
        self._active_downloads = {}

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        console.print("\n[yellow]Received shutdown signal. Stopping downloads...[/yellow]")
        self._shutdown_event.set()

    def add_download(self, url: str, filename: str):
        """Add a download to the queue"""
        # TODO: Implement download task creation and queuing
        pass

    def _download_file(self, task: DownloadTask) -> bool:
        """Download a single file with progress tracking and retry logic"""
        # TODO: Implement the actual file download logic with:
        # - HTTP requests with proper headers
        # - Progress tracking
        # - Retry logic on failures
        # - Thread-safe file writing
        # - Speed calculation
        pass

    def _get_file_size(self, url: str) -> int:
        """Get the size of a file from URL headers"""
        # TODO: Implement HEAD request to get file size
        pass

    def pause_download(self, filename: str):
        """Pause a specific download"""
        # TODO: Implement download pausing
        pass

    def resume_download(self, filename: str):
        """Resume a paused download"""
        # TODO: Implement download resuming
        pass

    def start_downloads(self):
        """Start the download manager"""
        # TODO: Implement the main download loop using ThreadPoolExecutor
        # - Use ThreadPoolExecutor with max_concurrent_downloads
        # - Process downloads from queue
        # - Handle retries
        # - Update progress in real-time
        # - Display status updates
        pass

    def _display_status_loop(self):
        """Run status display in a separate thread"""
        # TODO: Implement continuous status display update
        pass

def main():
    """Demo the download manager with sample files"""

    # Sample download URLs (replace with actual URLs for testing)
    download_urls = [
        ("https://httpbin.org/bytes/1024000", "file1.bin"),  # 1MB test file
        ("https://httpbin.org/bytes/2048000", "file2.bin"),  # 2MB test file
        ("https://httpbin.org/bytes/512000", "file3.bin"),   # 512KB test file
        ("https://httpbin.org/status/500", "file4.bin"),     # This will fail for retry testing
    ]

    manager = FileDownloadManager(max_concurrent_downloads=3)

    # Add downloads to manager
    for url, filename in download_urls:
        manager.add_download(url, filename)

    try:
        console.print("[bold green]Starting File Download Manager[/bold green]")
        console.print(f"Downloads to process: {len(download_urls)}")
        console.print("Press Ctrl+C to stop\n")

        manager.start_downloads()

    except KeyboardInterrupt:
        console.print("\n[yellow]Download manager stopped by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
    finally:
        # Cleanup
        console.print("[blue]Cleaning up...[/blue]")

if __name__ == "__main__":
    main()

# TODO: Implementation checklist
"""
□ Implement ThreadSafeProgressTracker methods
□ Implement FileDownloadManager._download_file() with:
  - HTTP request handling
  - Progress tracking
  - Retry logic
  - Thread-safe file writing
□ Implement queue processing in start_downloads()
□ Add pause/resume functionality
□ Implement real-time status display
□ Handle edge cases (network errors, disk full, etc.)
□ Add comprehensive error handling
□ Test with various file sizes and failure scenarios
"""