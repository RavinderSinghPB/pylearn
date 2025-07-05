"""
Lab 1.3: URL Checker with Asyncio
=================================

This lab demonstrates asyncio fundamentals using the same URL checking task.
You'll learn about async/await, coroutines, event loops, and async performance characteristics.

Learning Objectives:
- Understand asyncio basics and async/await syntax
- Learn about coroutines, tasks, and event loops
- Implement async HTTP requests with proper error handling
- Compare async performance with threading and multiprocessing
"""

import asyncio
import aiohttp
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import sys
import logging
from urllib.parse import urlparse
import json

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
    task_id: Optional[str] = None
    success: bool = False

class AsyncURLChecker:
    """Async-based URL checker"""

    def __init__(self, timeout: float = 10.0, max_concurrent: int = 50):
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def check_single_url(self, session: aiohttp.ClientSession, url: str) -> URLResult:
        """Check a single URL asynchronously"""
        start_time = time.time()
        task_id = id(asyncio.current_task())

        async with self.semaphore:  # Limit concurrent requests
            try:
                # Ensure URL has scheme
                if not url.startswith(('http://', 'https://')):
                    url = f'https://{url}'

                # Make async request
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.timeout)) as response:
                    response_time = time.time() - start_time

                    return URLResult(
                        url=url,
                        status_code=response.status,
                        response_time=response_time,
                        task_id=str(task_id),
                        success=response.status < 400
                    )

            except asyncio.TimeoutError:
                response_time = time.time() - start_time
                return URLResult(
                    url=url,
                    response_time=response_time,
                    error="Timeout",
                    task_id=str(task_id),
                    success=False
                )
            except Exception as e:
                response_time = time.time() - start_time
                return URLResult(
                    url=url,
                    response_time=response_time,
                    error=str(e),
                    task_id=str(task_id),
                    success=False
                )

    async def check_urls_sequential(self, urls: List[str]) -> List[URLResult]:
        """Check URLs sequentially (for comparison)"""
        print("\nSequential URL Checking with Asyncio")
        results = []

        connector = aiohttp.TCPConnector(limit=100)
        async with aiohttp.ClientSession(
            connector=connector,
            headers={'User-Agent': 'Python-URLChecker-Async/1.0'}
        ) as session:
            for i, url in enumerate(urls):
                result = await self.check_single_url(session, url)
                results.append(result)
                print(f"Progress: {i+1}/{len(urls)} - {url}")

        return results

    async def check_urls_async(self, urls: List[str]) -> List[URLResult]:
        """Check URLs asynchronously with gather"""
        print(f"\nAsync URL Checking with gather (max_concurrent={self.max_concurrent})")

        connector = aiohttp.TCPConnector(limit=100)
        async with aiohttp.ClientSession(
            connector=connector,
            headers={'User-Agent': 'Python-URLChecker-Async/1.0'}
        ) as session:
            # Create tasks for all URLs
            tasks = [self.check_single_url(session, url) for url in urls]

            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle any exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append(URLResult(
                        url=urls[i],
                        error=str(result),
                        success=False
                    ))
                else:
                    processed_results.append(result)

            return processed_results

    async def check_urls_as_completed(self, urls: List[str]) -> List[URLResult]:
        """Check URLs using as_completed for incremental results"""
        print(f"\nAsync URL Checking with as_completed (max_concurrent={self.max_concurrent})")
        results = []

        connector = aiohttp.TCPConnector(limit=100)
        async with aiohttp.ClientSession(
            connector=connector,
            headers={'User-Agent': 'Python-URLChecker-Async/1.0'}
        ) as session:
            # Create tasks for all URLs
            tasks = [self.check_single_url(session, url) for url in urls]

            # Process results as they complete
            completed = 0
            async for task in asyncio.as_completed(tasks):
                result = await task
                results.append(result)
                completed += 1
                print(f"Progress: {completed}/{len(urls)} - {result.url}")

            return results

    async def check_urls_with_queue(self, urls: List[str]) -> List[URLResult]:
        """Check URLs using asyncio Queue pattern"""
        print(f"\nAsync URL Checking with Queue (workers={self.max_concurrent})")

        # Create queue and add URLs
        url_queue = asyncio.Queue()
        for url in urls:
            await url_queue.put(url)

        # Results storage
        results = []
        results_lock = asyncio.Lock()

        async def worker(session: aiohttp.ClientSession, worker_id: int):
            """Worker coroutine that processes URLs from queue"""
            while True:
                try:
                    url = await asyncio.wait_for(url_queue.get(), timeout=1.0)
                    result = await self.check_single_url(session, url)

                    async with results_lock:
                        results.append(result)
                        print(f"Worker {worker_id} completed: {url}")

                    url_queue.task_done()
                except asyncio.TimeoutError:
                    break

        # Start workers
        connector = aiohttp.TCPConnector(limit=100)
        async with aiohttp.ClientSession(
            connector=connector,
            headers={'User-Agent': 'Python-URLChecker-Async/1.0'}
        ) as session:
            # Create worker tasks
            workers = [
                asyncio.create_task(worker(session, i))
                for i in range(min(self.max_concurrent, len(urls)))
            ]

            # Wait for all URLs to be processed
            await url_queue.join()

            # Cancel workers
            for w in workers:
                w.cancel()

        return results

    async def check_urls_with_batch_processing(self, urls: List[str], batch_size: int = 10) -> List[URLResult]:
        """Check URLs in batches to control memory usage"""
        print(f"\nAsync URL Checking with Batching (batch_size={batch_size})")

        all_results = []

        connector = aiohttp.TCPConnector(limit=100)
        async with aiohttp.ClientSession(
            connector=connector,
            headers={'User-Agent': 'Python-URLChecker-Async/1.0'}
        ) as session:
            # Process URLs in batches
            for i in range(0, len(urls), batch_size):
                batch = urls[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(urls)-1)//batch_size + 1}")

                # Create tasks for current batch
                tasks = [self.check_single_url(session, url) for url in batch]

                # Execute batch
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        all_results.append(URLResult(
                            url=batch[j],
                            error=str(result),
                            success=False
                        ))
                    else:
                        all_results.append(result)

                # Small delay between batches
                await asyncio.sleep(0.1)

        return all_results

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
            'unique_tasks': len(set(r.task_id for r in results if r.task_id))
        }

        return analysis

    def display_results_table(self, results: List[URLResult], method_name: str):
        """Display results in a formatted table"""
        print(f"\n{method_name} Results:")
        print("-" * 80)
        print(f"{'URL':<40} {'Status':<10} {'Time':<10} {'Task ID':<15} {'Success':<10}")
        print("-" * 80)

        for result in results[:10]:  # Show first 10 results
            url = result.url[:37] + "..." if len(result.url) > 40 else result.url
            status = str(result.status_code) if result.status_code else "Error"
            response_time = f"{result.response_time:.3f}s" if result.response_time else "N/A"
            task_id = result.task_id[:12] + "..." if result.task_id and len(result.task_id) > 15 else (result.task_id or "N/A")
            success = "✓" if result.success else "✗"

            print(f"{url:<40} {status:<10} {response_time:<10} {task_id:<15} {success:<10}")

        if len(results) > 10:
            print(f"... and {len(results) - 10} more results")
        print("-" * 80)

async def demonstrate_asyncio_concepts():
    """Demonstrate basic asyncio concepts"""
    print("\nAsyncio Concepts Demonstration")
    print("=" * 50)

    # 1. Basic coroutine
    print("\n1. Basic Coroutine:")

    async def hello_world(name: str, delay: float):
        print(f"Hello from {name}!")
        await asyncio.sleep(delay)
        print(f"Goodbye from {name}!")

    # Run multiple coroutines concurrently
    await asyncio.gather(
        hello_world("Alice", 1.0),
        hello_world("Bob", 0.5),
        hello_world("Charlie", 1.5)
    )

    # 2. Tasks and event loop
    print("\n2. Tasks and Event Loop:")

    async def task_function(name: str, duration: float):
        print(f"Task {name} starting...")
        await asyncio.sleep(duration)
        print(f"Task {name} completed!")
        return f"Result from {name}"

    # Create tasks
    task1 = asyncio.create_task(task_function("Task-1", 1.0))
    task2 = asyncio.create_task(task_function("Task-2", 0.5))
    task3 = asyncio.create_task(task_function("Task-3", 1.5))

    # Wait for tasks to complete
    results = await asyncio.gather(task1, task2, task3)
    print(f"Task results: {results}")

    # 3. Async context manager
    print("\n3. Async Context Manager:")

    class AsyncResource:
        def __init__(self, name: str):
            self.name = name

        async def __aenter__(self):
            print(f"Acquiring resource {self.name}")
            await asyncio.sleep(0.1)
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            print(f"Releasing resource {self.name}")
            await asyncio.sleep(0.1)

    async with AsyncResource("Database") as resource:
        print(f"Using {resource.name}")
        await asyncio.sleep(0.5)

    # 4. Async generator
    print("\n4. Async Generator:")

    async def async_range(n):
        for i in range(n):
            print(f"Yielding {i}")
            await asyncio.sleep(0.1)
            yield i

    async for value in async_range(3):
        print(f"Received {value}")

    # 5. Semaphore for limiting concurrency
    print("\n5. Semaphore for Limiting Concurrency:")

    semaphore = asyncio.Semaphore(2)  # Only 2 concurrent operations

    async def limited_operation(name: str):
        async with semaphore:
            print(f"Starting {name}")
            await asyncio.sleep(1.0)
            print(f"Finishing {name}")

    # This will run in groups of 2
    await asyncio.gather(
        limited_operation("Op-1"),
        limited_operation("Op-2"),
        limited_operation("Op-3"),
        limited_operation("Op-4")
    )

async def compare_async_patterns():
    """Compare different async patterns"""
    print("\nAsync Pattern Comparison")
    print("=" * 50)

    async def mock_api_call(url: str, delay: float = 0.5):
        """Mock API call with delay"""
        await asyncio.sleep(delay)
        return f"Data from {url}"

    urls = [f"https://api.example.com/data/{i}" for i in range(10)]

    # Pattern 1: Sequential
    print("\n1. Sequential Pattern:")
    start_time = time.time()
    sequential_results = []
    for url in urls:
        result = await mock_api_call(url)
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    print(f"Sequential time: {sequential_time:.2f}s")

    # Pattern 2: Concurrent with gather
    print("\n2. Concurrent with gather:")
    start_time = time.time()
    concurrent_results = await asyncio.gather(
        *[mock_api_call(url) for url in urls]
    )
    concurrent_time = time.time() - start_time
    print(f"Concurrent time: {concurrent_time:.2f}s")

    # Pattern 3: Limited concurrency with semaphore
    print("\n3. Limited concurrency with semaphore:")
    semaphore = asyncio.Semaphore(3)

    async def limited_api_call(url: str):
        async with semaphore:
            return await mock_api_call(url)

    start_time = time.time()
    limited_results = await asyncio.gather(
        *[limited_api_call(url) for url in urls]
    )
    limited_time = time.time() - start_time
    print(f"Limited concurrent time: {limited_time:.2f}s")

    # Pattern 4: as_completed for incremental results
    print("\n4. as_completed for incremental results:")
    start_time = time.time()
    as_completed_results = []
    tasks = [mock_api_call(url) for url in urls]

    async for task in asyncio.as_completed(tasks):
        result = await task
        as_completed_results.append(result)
        print(f"  Completed: {result}")

    as_completed_time = time.time() - start_time
    print(f"as_completed time: {as_completed_time:.2f}s")

    # Summary
    print(f"\nSpeedup comparison:")
    print(f"Sequential: 1.00x")
    print(f"Concurrent: {sequential_time/concurrent_time:.2f}x")
    print(f"Limited: {sequential_time/limited_time:.2f}x")
    print(f"as_completed: {sequential_time/as_completed_time:.2f}x")

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

async def main():
    """Main demonstration function"""
    print("Lab 1.3: URL Checker with Asyncio")
    print("=" * 50)

    # Demonstrate basic asyncio concepts
    await demonstrate_asyncio_concepts()

    # Compare async patterns
    await compare_async_patterns()

    # URL checking demonstration
    urls = get_test_urls()
    checker = AsyncURLChecker(timeout=5.0, max_concurrent=10)

    print(f"\nTesting with {len(urls)} URLs")
    print(f"Max concurrent: {checker.max_concurrent}")

    # Method 1: Sequential checking
    start_time = time.time()
    sequential_results = await checker.check_urls_sequential(urls)
    sequential_time = time.time() - start_time

    # Method 2: Async checking with gather
    start_time = time.time()
    async_results = await checker.check_urls_async(urls)
    async_time = time.time() - start_time

    # Method 3: as_completed approach
    start_time = time.time()
    as_completed_results = await checker.check_urls_as_completed(urls)
    as_completed_time = time.time() - start_time

    # Method 4: Queue-based approach
    start_time = time.time()
    queue_results = await checker.check_urls_with_queue(urls)
    queue_time = time.time() - start_time

    # Method 5: Batch processing
    start_time = time.time()
    batch_results = await checker.check_urls_with_batch_processing(urls, batch_size=5)
    batch_time = time.time() - start_time

    # Display results
    checker.display_results_table(sequential_results, "Sequential")
    checker.display_results_table(async_results, "Async with gather")
    checker.display_results_table(as_completed_results, "Async with as_completed")
    checker.display_results_table(queue_results, "Async with Queue")
    checker.display_results_table(batch_results, "Async with Batching")

    # Analyze and compare performance
    seq_analysis = checker.analyze_results(sequential_results, "Sequential")
    async_analysis = checker.analyze_results(async_results, "Async gather")
    completed_analysis = checker.analyze_results(as_completed_results, "as_completed")
    queue_analysis = checker.analyze_results(queue_results, "Queue")
    batch_analysis = checker.analyze_results(batch_results, "Batching")

    # Performance comparison
    print("\nPerformance Comparison")
    print("=" * 90)
    print(f"{'Method':<20} {'Total Time':<12} {'Success Rate':<12} {'Avg Time':<12} {'Tasks':<12} {'Speedup':<12}")
    print("=" * 90)

    methods = [
        ("Sequential", sequential_time, seq_analysis),
        ("Async gather", async_time, async_analysis),
        ("as_completed", as_completed_time, completed_analysis),
        ("Queue", queue_time, queue_analysis),
        ("Batching", batch_time, batch_analysis)
    ]

    baseline_time = sequential_time

    for method_name, total_time, analysis in methods:
        speedup = baseline_time / total_time if total_time > 0 else 0
        print(f"{method_name:<20} {total_time:<12.2f} {analysis['success_rate']:<12.1f} {analysis['avg_response_time']:<12.3f} {analysis['unique_tasks']:<12} {speedup:<12.2f}")

    print("=" * 90)

    # Key insights
    print("\nKey Insights:")
    print("1. Asyncio excels at I/O-bound concurrent operations")
    print("2. gather() is simple but uses more memory for large datasets")
    print("3. as_completed() provides incremental results")
    print("4. Queue pattern is useful for producer-consumer scenarios")
    print("5. Batching helps control memory usage and rate limiting")

    # Asyncio best practices
    print("\nAsyncio Best Practices:")
    print("• Use async/await consistently throughout your code")
    print("• Handle exceptions properly in async contexts")
    print("• Use semaphores to limit concurrent operations")
    print("• Consider using as_completed() for incremental processing")
    print("• Use proper connection pooling and session management")
    print("• Be mindful of the event loop and blocking operations")

if __name__ == "__main__":
    asyncio.run(main())