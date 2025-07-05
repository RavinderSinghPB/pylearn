#!/usr/bin/env python3
"""
Multiprocessing Challenge 2: Distributed Prime Number Calculator

Create a distributed prime number calculator that uses multiple processes to find
all prime numbers in a large range, with work distribution and result aggregation.

Requirements:
1. Work distribution among worker processes
2. Dynamic load balancing
3. Result aggregation using shared memory
4. Progress monitoring across workers
5. Parallel Sieve of Eratosthenes implementation
6. Fault tolerance for worker failures

Time: 30-40 minutes
"""

import multiprocessing as mp
from multiprocessing import Process, Queue, Value, Array, Manager
import math
import time
import signal
import sys
import numpy as np
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
import psutil

console = Console()

@dataclass
class WorkUnit:
    """Represents a work unit for prime calculation"""
    start: int
    end: int
    worker_id: int = -1
    status: str = "pending"  # pending, assigned, processing, completed, failed
    primes_found: int = 0
    processing_time: float = 0.0

class SharedMemoryPrimeStore:
    """Shared memory storage for prime numbers"""

    def __init__(self, max_number: int):
        self.max_number = max_number
        # TODO: Initialize shared memory arrays for:
        # - Prime numbers list
        # - Prime flags (sieve array)
        # - Result counters
        pass

    def mark_prime(self, number: int):
        """Mark a number as prime in shared memory"""
        # TODO: Implement thread-safe prime marking
        pass

    def is_prime(self, number: int) -> bool:
        """Check if a number is marked as prime"""
        # TODO: Implement prime checking from shared memory
        pass

    def get_primes_in_range(self, start: int, end: int) -> List[int]:
        """Get all primes in a given range"""
        # TODO: Implement range-based prime retrieval
        pass

    def get_total_prime_count(self) -> int:
        """Get total count of primes found"""
        # TODO: Implement total count calculation
        pass

class WorkDistributor:
    """Distributes work units among worker processes"""

    def __init__(self, max_number: int, num_workers: int, chunk_size: int = None):
        self.max_number = max_number
        self.num_workers = num_workers

        # Calculate optimal chunk size if not provided
        if chunk_size is None:
            self.chunk_size = max(1000, max_number // (num_workers * 4))
        else:
            self.chunk_size = chunk_size

        self.work_units = []
        self.completed_units = []
        self.failed_units = []

        # Create work distribution
        self._create_work_units()

    def _create_work_units(self):
        """Create initial work units"""
        # TODO: Implement work unit creation:
        # - Divide range into chunks
        # - Create WorkUnit objects
        # - Balance chunk sizes
        pass

    def get_next_work_unit(self, worker_id: int) -> WorkUnit:
        """Get next available work unit for a worker"""
        # TODO: Implement work unit assignment with:
        # - Thread-safe assignment
        # - Load balancing
        # - Failed unit reassignment
        pass

    def complete_work_unit(self, work_unit: WorkUnit):
        """Mark a work unit as completed"""
        # TODO: Implement work unit completion
        pass

    def fail_work_unit(self, work_unit: WorkUnit):
        """Mark a work unit as failed and reschedule"""
        # TODO: Implement work unit failure handling
        pass

    def get_progress_stats(self) -> Dict:
        """Get current progress statistics"""
        # TODO: Calculate and return progress statistics
        pass

class PrimeWorker(Process):
    """Worker process for prime number calculation"""

    def __init__(self, worker_id: int, work_queue: Queue, result_queue: Queue,
                 prime_store: SharedMemoryPrimeStore, shutdown_event: mp.Event,
                 progress_dict: Dict):
        super().__init__()
        self.worker_id = worker_id
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.prime_store = prime_store
        self.shutdown_event = shutdown_event
        self.progress_dict = progress_dict
        self.primes_found = 0

    def run(self):
        """Main worker process loop"""
        # TODO: Implement worker main loop:
        # - Get work units from queue
        # - Calculate primes in assigned range
        # - Update shared memory and progress
        # - Handle shutdown gracefully
        pass

    def _sieve_of_eratosthenes_range(self, start: int, end: int) -> List[int]:
        """Optimized sieve for a specific range"""
        # TODO: Implement range-specific sieve algorithm:
        # - Handle range boundaries correctly
        # - Optimize for memory usage
        # - Use mathematical optimizations
        pass

    def _is_prime_trial_division(self, n: int) -> bool:
        """Trial division method for prime checking"""
        # TODO: Implement trial division algorithm
        pass

    def _update_progress(self, work_unit: WorkUnit):
        """Update progress information"""
        # TODO: Implement progress updating
        pass

class PrimeCalculatorManager:
    """Manages the distributed prime calculation"""

    def __init__(self, max_number: int, num_workers: int = None,
                 chunk_size: int = None):
        self.max_number = max_number

        # Auto-detect optimal number of workers
        if num_workers is None:
            self.num_workers = min(mp.cpu_count(), max(1, max_number // 1000000))
        else:
            self.num_workers = num_workers

        self.chunk_size = chunk_size

        # Initialize components
        self.prime_store = SharedMemoryPrimeStore(max_number)
        self.work_distributor = WorkDistributor(max_number, self.num_workers, chunk_size)

        # Communication queues
        self.work_queue = Queue()
        self.result_queue = Queue()

        # Shared objects
        self.manager = Manager()
        self.shutdown_event = mp.Event()
        self.progress_dict = self.manager.dict()

        # Worker processes
        self.workers = []
        self.monitor_process = None

        # Statistics
        self.start_time = None
        self.end_time = None

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        console.print("\n[yellow]Received shutdown signal. Stopping calculation...[/yellow]")
        self.shutdown()

    def _populate_work_queue(self):
        """Populate the work queue with initial work units"""
        # TODO: Add work units to queue
        pass

    def _monitor_progress(self):
        """Monitor and display calculation progress"""
        # TODO: Implement progress monitoring:
        # - Display worker status
        # - Show overall progress
        # - Calculate performance metrics
        # - Handle worker failures
        pass

    def _handle_results(self):
        """Handle results from worker processes"""
        # TODO: Implement result handling:
        # - Collect results from workers
        # - Update shared memory
        # - Track completion status
        pass

    def start_calculation(self):
        """Start the distributed prime calculation"""
        # TODO: Implement calculation startup:
        # - Initialize shared memory
        # - Populate work queue
        # - Start worker processes
        # - Start monitoring
        pass

    def shutdown(self):
        """Graceful shutdown of calculation"""
        # TODO: Implement graceful shutdown:
        # - Set shutdown event
        # - Wait for workers to finish
        # - Collect final results
        # - Clean up resources
        pass

    def get_results(self) -> Dict:
        """Get final calculation results"""
        # TODO: Compile and return final results:
        # - Total primes found
        # - Execution time
        # - Performance metrics
        # - Worker statistics
        pass

    def _verify_results(self) -> bool:
        """Verify calculation results using alternative method"""
        # TODO: Implement result verification
        pass

def simple_sieve(limit: int) -> int:
    """Simple sieve for small numbers (verification)"""
    # TODO: Implement simple sieve for verification
    pass

def benchmark_sequential_vs_parallel(max_number: int):
    """Benchmark sequential vs parallel prime calculation"""
    # TODO: Implement benchmarking comparison
    pass

def main():
    """Demo the distributed prime calculator"""

    try:
        console.print("[bold green]Distributed Prime Number Calculator[/bold green]")

        # Configuration
        max_number = 10_000_000  # Find primes up to 10 million
        num_workers = mp.cpu_count()

        console.print(f"Finding primes from 1 to {max_number:,}")
        console.print(f"Using {num_workers} worker processes")
        console.print("Press Ctrl+C to stop\n")

        # Create calculator
        calculator = PrimeCalculatorManager(
            max_number=max_number,
            num_workers=num_workers
        )

        # Start calculation
        start_time = time.time()
        calculator.start_calculation()

        # Wait for completion or interruption
        try:
            while not calculator.shutdown_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass

        # Shutdown and get results
        calculator.shutdown()
        end_time = time.time()

        results = calculator.get_results()

        # Display results
        console.print("\n[bold blue]Calculation Results:[/bold blue]")
        console.print(f"Primes found: {results.get('total_primes', 0):,}")
        console.print(f"Execution time: {end_time - start_time:.2f} seconds")
        console.print(f"Primes per second: {results.get('primes_per_second', 0):,.0f}")
        console.print(f"Workers used: {num_workers}")

        # Optional: Run benchmark comparison
        if max_number <= 1_000_000:  # Only for smaller numbers
            console.print("\n[yellow]Running sequential comparison...[/yellow]")
            benchmark_sequential_vs_parallel(max_number)

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# TODO: Implementation checklist
"""
□ Implement SharedMemoryPrimeStore with efficient memory management
□ Implement WorkDistributor with dynamic load balancing
□ Implement PrimeWorker with optimized sieve algorithm
□ Implement work queue management and distribution
□ Implement progress monitoring and display
□ Implement result aggregation and verification
□ Add fault tolerance for worker failures
□ Implement graceful shutdown mechanism
□ Add performance benchmarking
□ Optimize memory usage for large ranges
□ Test with various range sizes and worker counts
□ Add comprehensive error handling
"""