#!/usr/bin/env python3
"""
Threading Challenge 2: Producer-Consumer Log Processor

Implement a multi-threaded log processing system with multiple producers generating
log entries and multiple consumers processing them with different priorities.

Requirements:
1. Producers (3 threads): Generate log entries with different log levels
2. Consumers (2 threads): Process logs based on priority (CRITICAL/ERROR first)
3. Thread-safe counters and shared data
4. Graceful shutdown with proper cleanup
5. Rate limiting to prevent memory overflow

Time: 35-50 minutes
"""

import threading
import time
import random
import queue
import signal
import sys
from dataclasses import dataclass
from typing import Dict, Set
from enum import Enum
from collections import defaultdict, Counter
import re
from rich.console import Console
from rich.table import Table
import json

console = Console()

class LogLevel(Enum):
    """Log levels with priority values"""
    CRITICAL = 5
    ERROR = 4
    WARN = 3
    INFO = 2
    DEBUG = 1

@dataclass
class LogEntry:
    """Represents a log entry"""
    timestamp: float
    level: LogLevel
    message: str
    ip_address: str
    source: str

    def __lt__(self, other):
        """Enable priority queue ordering by log level"""
        return self.level.value > other.level.value  # Higher priority first

class ThreadSafeStatistics:
    """Thread-safe statistics collector for log processing"""

    def __init__(self):
        self._lock = threading.Lock()
        self._log_counts = Counter()
        self._ip_addresses = set()
        self._error_patterns = Counter()
        self._total_processed = 0
        self._start_time = time.time()

    def update_log_count(self, level: LogLevel):
        """Thread-safe log level counter update"""
        # TODO: Implement thread-safe log level counting
        pass

    def add_ip_address(self, ip: str):
        """Thread-safe IP address tracking"""
        # TODO: Implement thread-safe IP address collection
        pass

    def add_error_pattern(self, pattern: str):
        """Thread-safe error pattern counting"""
        # TODO: Implement thread-safe error pattern tracking
        pass

    def increment_processed(self):
        """Thread-safe increment of processed count"""
        # TODO: Implement thread-safe counter increment
        pass

    def get_statistics(self) -> Dict:
        """Get current statistics in a thread-safe manner"""
        # TODO: Return current statistics safely
        pass

    def get_processing_rate(self) -> float:
        """Calculate current processing rate (logs/sec)"""
        # TODO: Calculate and return processing rate
        pass

class LogProducer:
    """Produces log entries at a controlled rate"""

    def __init__(self, producer_id: int, log_queue: queue.PriorityQueue,
                 shutdown_event: threading.Event, rate_limit: float = 10.0):
        self.producer_id = producer_id
        self.log_queue = log_queue
        self.shutdown_event = shutdown_event
        self.rate_limit = rate_limit  # logs per second
        self.logs_generated = 0

    def _generate_log_entry(self) -> LogEntry:
        """Generate a realistic log entry"""
        # TODO: Implement log entry generation with:
        # - Random log levels (weighted by frequency)
        # - Realistic messages
        # - Random IP addresses
        # - Various sources
        pass

    def _get_sample_messages(self) -> Dict[LogLevel, list]:
        """Get sample log messages for different levels"""
        return {
            LogLevel.CRITICAL: [
                "System crash detected - immediate attention required",
                "Database connection pool exhausted",
                "Memory usage critical - 95% utilized",
                "Disk space critical - less than 1% remaining"
            ],
            LogLevel.ERROR: [
                "Failed to connect to database: Connection timeout",
                "Authentication failed for user {user}",
                "File not found: /var/log/app.log",
                "Invalid JSON in request body",
                "HTTP 500 - Internal server error"
            ],
            LogLevel.WARN: [
                "High memory usage detected: 85%",
                "Deprecated API endpoint accessed",
                "Slow query detected: {query_time}ms",
                "Rate limit approaching for IP {ip}"
            ],
            LogLevel.INFO: [
                "User {user} logged in successfully",
                "New user registration: {user}",
                "API request processed: GET /api/users",
                "Cache refreshed successfully",
                "Scheduled backup completed"
            ],
            LogLevel.DEBUG: [
                "Processing request ID: {request_id}",
                "Cache hit for key: {cache_key}",
                "SQL query executed: SELECT * FROM users",
                "Function entered: validate_user_input"
            ]
        }

    def run(self):
        """Main producer loop"""
        # TODO: Implement producer main loop with:
        # - Rate limiting
        # - Log generation
        # - Queue management
        # - Shutdown handling
        pass

class LogConsumer:
    """Consumes and processes log entries"""

    def __init__(self, consumer_id: int, log_queue: queue.PriorityQueue,
                 stats: ThreadSafeStatistics, shutdown_event: threading.Event):
        self.consumer_id = consumer_id
        self.log_queue = log_queue
        self.stats = stats
        self.shutdown_event = shutdown_event
        self.logs_processed = 0

    def _extract_ip_address(self, message: str) -> str:
        """Extract IP address from log message"""
        # TODO: Implement IP address extraction using regex
        pass

    def _detect_error_patterns(self, entry: LogEntry) -> list:
        """Detect error patterns in log entries"""
        # TODO: Implement error pattern detection:
        # - SQL injection attempts
        # - Failed login attempts
        # - 404 errors
        # - Timeout errors
        # - Security violations
        pass

    def _process_log_entry(self, entry: LogEntry):
        """Process a single log entry"""
        # TODO: Implement log processing:
        # - Update statistics
        # - Extract IP addresses
        # - Detect error patterns
        # - Simulate processing time
        pass

    def run(self):
        """Main consumer loop"""
        # TODO: Implement consumer main loop with:
        # - Priority queue processing
        # - Log entry processing
        # - Statistics updates
        # - Graceful shutdown
        pass

class LogProcessingSystem:
    """Main log processing system coordinator"""

    def __init__(self, num_producers: int = 3, num_consumers: int = 2,
                 queue_size: int = 1000, producer_rate: float = 15.0):
        self.num_producers = num_producers
        self.num_consumers = num_consumers
        self.log_queue = queue.PriorityQueue(maxsize=queue_size)
        self.stats = ThreadSafeStatistics()
        self.shutdown_event = threading.Event()
        self.producer_rate = producer_rate

        self.producer_threads = []
        self.consumer_threads = []
        self.status_thread = None

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        console.print("\n[yellow]Received shutdown signal. Stopping log processing...[/yellow]")
        self.shutdown()

    def start(self):
        """Start the log processing system"""
        # TODO: Implement system startup:
        # - Create and start producer threads
        # - Create and start consumer threads
        # - Start status display thread
        pass

    def shutdown(self):
        """Graceful shutdown of the processing system"""
        # TODO: Implement graceful shutdown:
        # - Set shutdown event
        # - Wait for producers to finish
        # - Process remaining queue items
        # - Wait for consumers to finish
        # - Join all threads
        pass

    def _display_status_loop(self):
        """Display real-time processing status"""
        # TODO: Implement status display loop with:
        # - Real-time statistics
        # - Queue status
        # - Processing rates
        # - Thread status
        pass

    def display_final_report(self):
        """Display final processing report"""
        # TODO: Implement final report display
        pass

def main():
    """Demo the log processing system"""

    try:
        console.print("[bold green]Starting Log Processing System[/bold green]")
        console.print("Producers: 3 | Consumers: 2 | Rate Limit: 15 logs/sec")
        console.print("Press Ctrl+C to stop\n")

        # Create and start the processing system
        system = LogProcessingSystem(
            num_producers=3,
            num_consumers=2,
            queue_size=1000,
            producer_rate=15.0
        )

        system.start()

        # Let it run for a demo period
        time.sleep(30)  # Run for 30 seconds
        system.shutdown()

        # Display final report
        system.display_final_report()

    except KeyboardInterrupt:
        console.print("\n[yellow]Log processing stopped by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")

if __name__ == "__main__":
    main()

# TODO: Implementation checklist
"""
□ Implement ThreadSafeStatistics methods with proper locking
□ Implement LogProducer._generate_log_entry() with realistic data
□ Implement LogProducer.run() with rate limiting
□ Implement LogConsumer._process_log_entry() with pattern detection
□ Implement LogConsumer.run() with priority queue processing
□ Implement LogProcessingSystem.start() with thread management
□ Implement graceful shutdown mechanism
□ Add real-time status display
□ Implement comprehensive error handling
□ Test with various load scenarios
"""