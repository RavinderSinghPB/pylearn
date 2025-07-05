"""
Lab 1.2: URL Checker with Multiprocessing
==========================================

This lab demonstrates multiprocessing fundamentals using the same URL checking task.
You'll learn about ProcessPoolExecutor, process communication, and CPU vs I/O bound performance.

Learning Objectives:
- Understand multiprocessing basics and ProcessPoolExecutor
- Learn about process communication and shared memory
- Compare multiprocessing vs threading performance
- Understand when to use multiprocessing vs threading
"""

import multiprocessing as mp
import time
import requests
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import os
import queue
import logging
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class URLResult:
    """Container for URL check results"""
    url: str
    status_code: Optional[int] = None
    response_time: Optional[float] = None
    error: Optional[str] = None
    process_id: Optional[int] = None
    success: bool = False

class URLChecker:
    """Process-based URL checker"""

    def __init__(self, timeout: float = 10.0, max_workers: int = None):
        self.timeout = timeout
        self.max_workers = max_workers or mp.cpu_count()

    def check_single_url(self, url: str) -> URLResult:
        """Check a single URL and return result"""
        start_time = time.time()
        process_id = os.getpid()

        try:
            # Ensure URL has scheme
            if not url.startswith(('http://', 'https://')):
                url = f'https://{url}'

            # Create a new session for each process
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Python-URLChecker-MP/1.0'
            })

            # Make request
            response = session.get(url, timeout=self.timeout)
            response_time = time.time() - start_time

            return URLResult(
                url=url,
                status_code=response.status_code,
                response_time=response_time,
                process_id=process_id,
                success=response.status_code < 400
            )

        except requests.exceptions.RequestException as e:
            response_time = time.time() - start_time
            return URLResult(
                url=url,
                response_time=response_time,
                error=str(e),
                process_id=process_id,
                success=False
            )

    def check_urls_sequential(self, urls: List[str]) -> List[URLResult]:
        """Check URLs sequentially (for comparison)"""
        print("\nSequential URL Checking with Multiprocessing")
        results = []

        for i, url in enumerate(urls):
            result = self.check_single_url(url)
            results.append(result)
            print(f"Progress: {i+1}/{len(urls)} - {url}")

        return results

    def check_urls_multiprocessing(self, urls: List[str]) -> List[URLResult]:
        """Check URLs using ProcessPoolExecutor"""
        print(f"\nMultiprocessing URL Checking (max_workers={self.max_workers})")
        results = []

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_url = {executor.submit(self.check_single_url, url): url for url in urls}

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_url):
                result = future.result()
                results.append(result)
                completed += 1
                print(f"Progress: {completed}/{len(urls)} - {result.url}")

        return results

    def check_urls_with_shared_memory(self, urls: List[str]) -> List[URLResult]:
        """Check URLs using multiprocessing with shared memory for counters"""
        print(f"\nMultiprocessing with Shared Memory (workers={self.max_workers})")

        # Create shared counter
        with mp.Manager() as manager:
            counter = manager.Value('i', 0)
            counter_lock = manager.Lock()

            def check_url_with_counter(url: str) -> URLResult:
                result = self.check_single_url(url)
                with counter_lock:
                    counter.value += 1
                    print(f"Completed {counter.value}/{len(urls)}: {url}")
                return result

            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(check_url_with_counter, urls))

        return results

    def check_urls_with_queues(self, urls: List[str]) -> List[URLResult]:
        """Check URLs using multiprocessing with queues"""
        print(f"\nMultiprocessing with Queues (workers={self.max_workers})")

        # Create queues
        task_queue = mp.Queue()
        result_queue = mp.Queue()

        # Add tasks to queue
        for url in urls:
            task_queue.put(url)

        # Add sentinel values
        for _ in range(self.max_workers):
            task_queue.put(None)

        def worker(task_q, result_q):
            """Worker process function"""
            while True:
                url = task_q.get()
                if url is None:
                    break

                result = self.check_single_url(url)
                result_q.put(result)

        # Start worker processes
        processes = []
        for _ in range(self.max_workers):
            p = mp.Process(target=worker, args=(task_queue, result_queue))
            p.start()
            processes.append(p)

        # Collect results
        results = []
        for _ in range(len(urls)):
            result = result_queue.get()
            results.append(result)
            print(f"Received result for: {result.url}")

        # Wait for all processes to complete
        for p in processes:
            p.join()

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
            'unique_processes': len(set(r.process_id for r in results if r.process_id))
        }

        return analysis

    def display_results_table(self, results: List[URLResult], method_name: str):
        """Display results in a formatted table"""
        print(f"\n{method_name} Results:")
        print("-" * 80)
        print(f"{'URL':<40} {'Status':<10} {'Time':<10} {'Process':<10} {'Success':<10}")
        print("-" * 80)

        for result in results[:10]:  # Show first 10 results
            url = result.url[:37] + "..." if len(result.url) > 40 else result.url
            status = str(result.status_code) if result.status_code else "Error"
            response_time = f"{result.response_time:.3f}s" if result.response_time else "N/A"
            process_id = str(result.process_id) if result.process_id else "N/A"
            success = "✓" if result.success else "✗"

            print(f"{url:<40} {status:<10} {response_time:<10} {process_id:<10} {success:<10}")

        if len(results) > 10:
            print(f"... and {len(results) - 10} more results")
        print("-" * 80)

def demonstrate_multiprocessing_concepts():
    """Demonstrate basic multiprocessing concepts"""
    print("\nMultiprocessing Concepts Demonstration")
    print("=" * 50)

    # 1. Basic process creation
    print("\n1. Basic Process Creation:")

    def print_process_info(name: str):
        pid = os.getpid()
        print(f"Process {name}: PID={pid}")
        time.sleep(1)
        print(f"Process {name} finished")

    # Create and start processes
    processes = []
    for i in range(3):
        p = mp.Process(target=print_process_info, args=(f"Worker-{i}",))
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # 2. Process communication with Queue
    print("\n2. Process Communication with Queue:")

    def producer(queue, name):
        for i in range(5):
            item = f"{name}-item-{i}"
            queue.put(item)
            print(f"Producer {name} produced: {item}")
            time.sleep(0.1)

    def consumer(queue, name):
        while True:
            try:
                item = queue.get(timeout=2)
                print(f"Consumer {name} consumed: {item}")
                time.sleep(0.1)
            except:
                break

    # Create queue and processes
    q = mp.Queue()

    prod_process = mp.Process(target=producer, args=(q, "P1"))
    cons_process = mp.Process(target=consumer, args=(q, "C1"))

    prod_process.start()
    cons_process.start()

    prod_process.join()
    cons_process.join()

    # 3. Shared memory demonstration
    print("\n3. Shared Memory Demonstration:")

    def worker_with_shared_value(shared_val, lock, name):
        for _ in range(1000):
            with lock:
                shared_val.value += 1
        print(f"Worker {name} finished, current value: {shared_val.value}")

    # Create shared value and lock
    shared_value = mp.Value('i', 0)
    lock = mp.Lock()

    # Create and start processes
    processes = []
    for i in range(3):
        p = mp.Process(target=worker_with_shared_value, args=(shared_value, lock, f"W{i}"))
        processes.append(p)
        p.start()

    # Wait for completion
    for p in processes:
        p.join()

    print(f"Final shared value: {shared_value.value} (expected: 3000)")

def cpu_bound_task(n: int) -> int:
    """CPU-intensive task for performance comparison"""
    total = 0
    for i in range(n):
        total += i * i
    return total

def compare_cpu_bound_performance():
    """Compare threading vs multiprocessing for CPU-bound tasks"""
    print("\nCPU-Bound Performance Comparison")
    print("=" * 50)

    task_size = 1000000
    num_tasks = 8

    # Sequential execution
    print("1. Sequential execution:")
    start_time = time.time()
    sequential_results = [cpu_bound_task(task_size) for _ in range(num_tasks)]
    sequential_time = time.time() - start_time
    print(f"Sequential time: {sequential_time:.2f}s")

    # Multiprocessing execution
    print("2. Multiprocessing execution:")
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        mp_results = list(executor.map(cpu_bound_task, [task_size] * num_tasks))
    mp_time = time.time() - start_time
    print(f"Multiprocessing time: {mp_time:.2f}s")

    # Compare results
    speedup = sequential_time / mp_time if mp_time > 0 else 0
    print(f"Speedup: {speedup:.2f}x")
    print(f"Results match: {sequential_results == mp_results}")

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
        "https://httpbin.org/delay/1",
        "https://httpbin.org/status/301",
        "https://httpbin.org/status/302",
        "https://httpbin.org/json",
        "https://httpbin.org/xml",
        "https://httpbin.org/uuid",
    ]
    return urls

def main():
    """Main demonstration function"""
    print("Lab 1.2: URL Checker with Multiprocessing")
    print("=" * 50)

    # Demonstrate basic multiprocessing concepts
    demonstrate_multiprocessing_concepts()

    # CPU-bound performance comparison
    compare_cpu_bound_performance()

    # URL checking demonstration
    urls = get_test_urls()
    checker = URLChecker(timeout=5.0, max_workers=4)

    print(f"\nTesting with {len(urls)} URLs")
    print(f"CPU count: {mp.cpu_count()}")
    print(f"Max workers: {checker.max_workers}")

    # Method 1: Sequential checking
    start_time = time.time()
    sequential_results = checker.check_urls_sequential(urls)
    sequential_time = time.time() - start_time

    # Method 2: Multiprocessing checking
    start_time = time.time()
    mp_results = checker.check_urls_multiprocessing(urls)
    mp_time = time.time() - start_time

    # Method 3: Shared memory approach
    start_time = time.time()
    shared_mem_results = checker.check_urls_with_shared_memory(urls)
    shared_mem_time = time.time() - start_time

    # Method 4: Queue-based approach
    start_time = time.time()
    queue_results = checker.check_urls_with_queues(urls)
    queue_time = time.time() - start_time

    # Display results
    checker.display_results_table(sequential_results, "Sequential")
    checker.display_results_table(mp_results, "Multiprocessing")
    checker.display_results_table(shared_mem_results, "Shared Memory")
    checker.display_results_table(queue_results, "Queue-based")

    # Analyze and compare performance
    seq_analysis = checker.analyze_results(sequential_results, "Sequential")
    mp_analysis = checker.analyze_results(mp_results, "Multiprocessing")
    shared_analysis = checker.analyze_results(shared_mem_results, "Shared Memory")
    queue_analysis = checker.analyze_results(queue_results, "Queue-based")

    # Performance comparison
    print("\nPerformance Comparison")
    print("=" * 80)
    print(f"{'Method':<20} {'Total Time':<12} {'Success Rate':<12} {'Avg Time':<12} {'Processes':<12} {'Speedup':<12}")
    print("=" * 80)

    methods = [
        ("Sequential", sequential_time, seq_analysis),
        ("Multiprocessing", mp_time, mp_analysis),
        ("Shared Memory", shared_mem_time, shared_analysis),
        ("Queue-based", queue_time, queue_analysis)
    ]

    baseline_time = sequential_time

    for method_name, total_time, analysis in methods:
        speedup = baseline_time / total_time if total_time > 0 else 0
        print(f"{method_name:<20} {total_time:<12.2f} {analysis['success_rate']:<12.1f} {analysis['avg_response_time']:<12.3f} {analysis['unique_processes']:<12} {speedup:<12.2f}")

    print("=" * 80)

    # Key insights
    print("\nKey Insights:")
    print("1. Multiprocessing has higher overhead due to process creation")
    print("2. For I/O-bound tasks, threading is usually more efficient")
    print("3. Multiprocessing excels at CPU-bound tasks")
    print("4. Process communication adds overhead")
    print("5. Each process has its own memory space")

    # Multiprocessing best practices
    print("\nMultiprocessing Best Practices:")
    print("• Use ProcessPoolExecutor for most multiprocessing tasks")
    print("• Minimize data sharing between processes")
    print("• Use queues for process communication")
    print("• Be aware of serialization overhead")
    print("• Consider using multiprocessing for CPU-bound tasks")
    print("• Use shared memory sparingly and with proper synchronization")

if __name__ == "__main__":
    # Required for multiprocessing on Windows
    mp.set_start_method('spawn', force=True)
    main()