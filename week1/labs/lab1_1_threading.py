"""
Lab 1.1: URL Checker with Threading
===================================

This lab demonstrates threading basics by checking URL response times.
You'll learn about ThreadPoolExecutor, thread-safe operations, and performance characteristics.

Learning Objectives:
- Understand threading basics and the ThreadPoolExecutor
- Implement thread-safe URL checking with proper error handling
- Measure performance and understand GIL impact on I/O-bound tasks
- Learn best practices for resource management in threading
"""

import threading
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse
import queue
import logging
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()

@dataclass
class URLResult:
    """Container for URL check results"""
    url: str
    status_code: Optional[int] = None
    response_time: Optional[float] = None
    error: Optional[str] = None
    thread_id: Optional[int] = None
    success: bool = False

class ThreadSafeCounter:
    """Thread-safe counter for tracking operations"""

    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    def increment(self):
        with self._lock:
            self._value += 1

    def get_value(self):
        with self._lock:
            return self._value

class URLChecker:
    """Thread-based URL checker with comprehensive features"""

    def __init__(self, timeout: float = 10.0, max_workers: int = 10):
        self.timeout = timeout
        self.max_workers = max_workers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Python-URLChecker/1.0'
        })
        self.counter = ThreadSafeCounter()
        self.results_queue = queue.Queue()

    def check_single_url(self, url: str) -> URLResult:
        """Check a single URL and return result"""
        start_time = time.time()
        thread_id = threading.current_thread().ident

        try:
            # Ensure URL has scheme
            if not url.startswith(('http://', 'https://')):
                url = f'https://{url}'

            # Make request
            response = self.session.get(url, timeout=self.timeout)
            response_time = time.time() - start_time

            result = URLResult(
                url=url,
                status_code=response.status_code,
                response_time=response_time,
                thread_id=thread_id,
                success=response.status_code < 400
            )

            # Increment counter in thread-safe manner
            self.counter.increment()

            return result

        except requests.exceptions.RequestException as e:
            response_time = time.time() - start_time
            return URLResult(
                url=url,
                response_time=response_time,
                error=str(e),
                thread_id=thread_id,
                success=False
            )

    def check_urls_sequential(self, urls: List[str]) -> List[URLResult]:
        """Check URLs sequentially (for comparison)"""
        console.print("\n[bold blue]Sequential URL Checking[/bold blue]")
        results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Checking URLs sequentially...", total=len(urls))

            for url in urls:
                result = self.check_single_url(url)
                results.append(result)
                progress.update(task, advance=1)

        return results

    def check_urls_threaded(self, urls: List[str]) -> List[URLResult]:
        """Check URLs using ThreadPoolExecutor"""
        console.print(f"\n[bold green]Threaded URL Checking (max_workers={self.max_workers})[/bold green]")
        results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Checking URLs with threading...", total=len(urls))

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_url = {executor.submit(self.check_single_url, url): url for url in urls}

                # Collect results as they complete
                for future in as_completed(future_to_url):
                    result = future.result()
                    results.append(result)
                    progress.update(task, advance=1)

        return results

    def check_urls_producer_consumer(self, urls: List[str]) -> List[URLResult]:
        """Check URLs using producer-consumer pattern"""
        console.print(f"\n[bold yellow]Producer-Consumer Pattern (workers={self.max_workers})[/bold yellow]")

        url_queue = queue.Queue()
        results = []
        results_lock = threading.Lock()

        # Producer: Add URLs to queue
        for url in urls:
            url_queue.put(url)

        # Add sentinel values to signal workers to stop
        for _ in range(self.max_workers):
            url_queue.put(None)

        def worker():
            """Worker function that processes URLs from queue"""
            while True:
                url = url_queue.get()
                if url is None:
                    break

                result = self.check_single_url(url)

                with results_lock:
                    results.append(result)

                url_queue.task_done()

        # Start worker threads
        threads = []
        for _ in range(self.max_workers):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)

        # Wait for all threads to complete
        for t in threads:
            t.join()

        return results

    def analyze_results(self, results: List[URLResult], method_name: str) -> Dict[str, Any]:
        """Analyze and display results"""
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]

        if successful_results:
            response_times = [r.response_time for r in successful_results]
            avg_response_time = sum(response_times) / len(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
        else:
            avg_response_time = min_response_time = max_response_time = 0

        total_time = sum(r.response_time for r in results if r.response_time)

        analysis = {
            'method': method_name,
            'total_urls': len(results),
            'successful': len(successful_results),
            'failed': len(failed_results),
            'success_rate': len(successful_results) / len(results) * 100,
            'avg_response_time': avg_response_time,
            'min_response_time': min_response_time,
            'max_response_time': max_response_time,
            'total_time': total_time,
            'unique_threads': len(set(r.thread_id for r in results if r.thread_id))
        }

        return analysis

    def display_results_table(self, results: List[URLResult], method_name: str):
        """Display results in a formatted table"""
        table = Table(title=f"Results: {method_name}")
        table.add_column("URL", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Response Time", style="yellow")
        table.add_column("Thread ID", style="magenta")
        table.add_column("Success", style="bold")

        for result in results[:10]:  # Show first 10 results
            status = str(result.status_code) if result.status_code else "Error"
            response_time = f"{result.response_time:.3f}s" if result.response_time else "N/A"
            thread_id = str(result.thread_id) if result.thread_id else "N/A"
            success = "✓" if result.success else "✗"
            success_style = "green" if result.success else "red"

            table.add_row(
                result.url[:50] + "..." if len(result.url) > 50 else result.url,
                status,
                response_time,
                thread_id,
                f"[{success_style}]{success}[/{success_style}]"
            )

        console.print(table)
        if len(results) > 10:
            console.print(f"\n[dim]... and {len(results) - 10} more results[/dim]")

def get_test_urls() -> List[str]:
    """Get a list of URLs for testing"""
    urls = [
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/2",
        "https://httpbin.org/status/200",
        "https://httpbin.org/status/404",
        "https://httpbin.org/status/500",
        "https://www.google.com",
        "https://www.github.com",
        "https://www.stackoverflow.com",
        "https://www.reddit.com",
        "https://www.python.org",
        "https://fastapi.tiangolo.com",
        "https://docs.python.org",
        "https://realpython.com",
        "https://www.w3.org",
        "https://httpbin.org/delay/3",
        "https://httpbin.org/delay/1",
        "https://httpbin.org/status/301",
        "https://httpbin.org/status/302",
        "https://httpbin.org/json",
        "https://httpbin.org/xml",
    ]
    return urls

def demonstrate_threading_concepts():
    """Demonstrate basic threading concepts"""
    console.print("\n[bold red]Threading Concepts Demonstration[/bold red]")

    # 1. Basic thread creation
    console.print("\n1. Basic Thread Creation:")

    def print_numbers(name: str, delay: float):
        for i in range(5):
            print(f"Thread {name}: {i}")
            time.sleep(delay)

    # Create and start threads
    thread1 = threading.Thread(target=print_numbers, args=("A", 0.5))
    thread2 = threading.Thread(target=print_numbers, args=("B", 0.3))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    # 2. Thread synchronization with Lock
    console.print("\n2. Thread Synchronization with Lock:")

    # Use a class to encapsulate shared state
    class SharedCounter:
        def __init__(self):
            self.value = 0
            self.lock = threading.Lock()

        def increment(self):
            with self.lock:
                self.value += 1

        def get_value(self):
            with self.lock:
                return self.value

    shared_counter = SharedCounter()

    def increment_counter(name: str, counter: SharedCounter):
        for _ in range(1000):
            counter.increment()

    threads = []
    for i in range(5):
        t = threading.Thread(target=increment_counter, args=(f"Thread-{i}", shared_counter))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    console.print(f"Final counter value: {shared_counter.get_value()} (expected: 5000)")

    # 3. Thread-local storage
    console.print("\n3. Thread-Local Storage:")

    thread_local_data = threading.local()

    def process_data(name: str):
        thread_local_data.name = name
        thread_local_data.value = threading.current_thread().ident
        time.sleep(0.1)
        console.print(f"Thread {thread_local_data.name}: {thread_local_data.value}")

    threads = []
    for i in range(3):
        t = threading.Thread(target=process_data, args=(f"Worker-{i}",))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

def main():
    """Main demonstration function"""
    console.print("[bold magenta]Lab 1.1: URL Checker with Threading[/bold magenta]")
    console.print("=" * 50)

    # Demonstrate basic threading concepts
    demonstrate_threading_concepts()

    # URL checking demonstration
    urls = get_test_urls()
    checker = URLChecker(timeout=5.0, max_workers=8)

    console.print(f"\n[bold]Testing with {len(urls)} URLs[/bold]")

    # Method 1: Sequential checking
    start_time = time.time()
    sequential_results = checker.check_urls_sequential(urls)
    sequential_time = time.time() - start_time

    # Method 2: Threaded checking
    start_time = time.time()
    threaded_results = checker.check_urls_threaded(urls)
    threaded_time = time.time() - start_time

    # Method 3: Producer-consumer pattern
    start_time = time.time()
    producer_consumer_results = checker.check_urls_producer_consumer(urls)
    producer_consumer_time = time.time() - start_time

    # Display results
    checker.display_results_table(sequential_results, "Sequential")
    checker.display_results_table(threaded_results, "Threaded")
    checker.display_results_table(producer_consumer_results, "Producer-Consumer")

    # Analyze and compare performance
    seq_analysis = checker.analyze_results(sequential_results, "Sequential")
    threaded_analysis = checker.analyze_results(threaded_results, "Threaded")
    pc_analysis = checker.analyze_results(producer_consumer_results, "Producer-Consumer")

    # Performance comparison table
    comparison_table = Table(title="Performance Comparison")
    comparison_table.add_column("Method", style="cyan")
    comparison_table.add_column("Total Time", style="yellow")
    comparison_table.add_column("Success Rate", style="green")
    comparison_table.add_column("Avg Response Time", style="blue")
    comparison_table.add_column("Unique Threads", style="magenta")
    comparison_table.add_column("Speedup", style="red")

    methods = [
        ("Sequential", sequential_time, seq_analysis),
        ("Threaded", threaded_time, threaded_analysis),
        ("Producer-Consumer", producer_consumer_time, pc_analysis)
    ]

    baseline_time = sequential_time

    for method_name, total_time, analysis in methods:
        speedup = baseline_time / total_time if total_time > 0 else 0
        comparison_table.add_row(
            method_name,
            f"{total_time:.2f}s",
            f"{analysis['success_rate']:.1f}%",
            f"{analysis['avg_response_time']:.3f}s",
            str(analysis['unique_threads']),
            f"{speedup:.2f}x"
        )

    console.print(comparison_table)

    # Key insights
    console.print("\n[bold green]Key Insights:[/bold green]")
    console.print("1. Threading provides significant speedup for I/O-bound tasks")
    console.print("2. ThreadPoolExecutor simplifies thread management")
    console.print("3. Producer-consumer pattern is useful for decoupling production and consumption")
    console.print("4. Thread-safe operations are crucial for shared resources")
    console.print("5. GIL doesn't significantly impact I/O-bound operations")

    # Threading best practices
    console.print("\n[bold yellow]Threading Best Practices:[/bold yellow]")
    console.print("• Use ThreadPoolExecutor for most threading tasks")
    console.print("• Always use proper synchronization for shared resources")
    console.print("• Set appropriate timeouts to prevent hanging")
    console.print("• Use thread-local storage for per-thread data")
    console.print("• Handle exceptions properly in threaded code")
    console.print("• Consider using Queue for producer-consumer patterns")

if __name__ == "__main__":
    main()