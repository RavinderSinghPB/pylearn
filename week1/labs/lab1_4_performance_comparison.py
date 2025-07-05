"""
Lab 1.4: Performance Comparison & GIL Analysis
==============================================

This lab provides comprehensive performance comparison between threading, multiprocessing, and asyncio.
It includes GIL impact analysis and practical recommendations for choosing the right approach.

Learning Objectives:
- Understand the GIL's impact on different workload types
- Compare performance characteristics of all three approaches
- Learn to benchmark and profile concurrent applications
- Develop decision-making skills for choosing concurrency models
"""

import time
import threading
import multiprocessing as mp
import asyncio
import aiohttp
import requests
import math
import sys
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from functools import wraps
import json
import os

@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    method: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    success_rate: float
    throughput: float
    overhead: float
    scalability_score: float

class PerformanceAnalyzer:
    """Comprehensive performance analysis toolkit"""

    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.baseline_cpu = self.process.cpu_percent()

    def measure_performance(self, func: Callable) -> Dict[str, Any]:
        """Measure performance metrics of a function"""
        # Clear garbage collector
        gc.collect()

        # Initial measurements
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = self.process.cpu_percent()
        start_time = time.time()

        # Execute function
        result = func()

        # Final measurements
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        end_cpu = self.process.cpu_percent()

        return {
            'result': result,
            'execution_time': end_time - start_time,
            'memory_delta': end_memory - start_memory,
            'cpu_usage': end_cpu - start_cpu,
            'peak_memory': end_memory
        }

class GILAnalyzer:
    """Analyze GIL impact on different workload types"""

    @staticmethod
    def cpu_bound_task(n: int) -> int:
        """CPU-intensive task for GIL analysis"""
        total = 0
        for i in range(n):
            total += math.sqrt(i) * math.sin(i) * math.cos(i)
        return int(total)

    @staticmethod
    def io_bound_task(delay: float) -> str:
        """I/O-bound task simulation"""
        time.sleep(delay)
        return f"IO task completed after {delay}s"

    @staticmethod
    def memory_bound_task(size: int) -> int:
        """Memory-intensive task"""
        data = list(range(size))
        return sum(x * x for x in data)

    def analyze_gil_impact(self, task_type: str, num_tasks: int = 8) -> Dict[str, Any]:
        """Analyze GIL impact on different task types"""
        print(f"\nAnalyzing GIL impact on {task_type} tasks")
        print("-" * 50)

        if task_type == "cpu_bound":
            task_func = lambda: self.cpu_bound_task(100000)
        elif task_type == "io_bound":
            task_func = lambda: self.io_bound_task(0.5)
        else:
            task_func = lambda: self.memory_bound_task(100000)

        results = {}

        # Sequential execution
        start_time = time.time()
        sequential_results = [task_func() for _ in range(num_tasks)]
        sequential_time = time.time() - start_time
        results['sequential'] = sequential_time

        # Threading execution
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_tasks) as executor:
            thread_results = list(executor.map(lambda x: task_func(), range(num_tasks)))
        thread_time = time.time() - start_time
        results['threading'] = thread_time

        # Multiprocessing execution
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=min(num_tasks, mp.cpu_count())) as executor:
            mp_results = list(executor.map(lambda x: task_func(), range(num_tasks)))
        mp_time = time.time() - start_time
        results['multiprocessing'] = mp_time

        # Calculate speedups
        results['thread_speedup'] = sequential_time / thread_time
        results['mp_speedup'] = sequential_time / mp_time
        results['gil_efficiency'] = thread_time / mp_time if mp_time > 0 else float('inf')

        return results

class ComprehensiveBenchmark:
    """Comprehensive benchmark suite comparing all approaches"""

    def __init__(self):
        self.analyzer = PerformanceAnalyzer()
        self.gil_analyzer = GILAnalyzer()

    def benchmark_url_checking(self, urls: List[str]) -> Dict[str, BenchmarkResult]:
        """Benchmark URL checking across all approaches"""
        print("\nBenchmarking URL Checking Performance")
        print("=" * 50)

        results = {}

        # Sequential benchmark
        print("1. Sequential approach...")
        sequential_result = self.analyzer.measure_performance(
            lambda: self._sequential_url_check(urls)
        )
        results['sequential'] = self._create_benchmark_result(
            'Sequential', sequential_result, len(urls)
        )

        # Threading benchmark
        print("2. Threading approach...")
        threading_result = self.analyzer.measure_performance(
            lambda: self._threading_url_check(urls)
        )
        results['threading'] = self._create_benchmark_result(
            'Threading', threading_result, len(urls)
        )

        # Multiprocessing benchmark
        print("3. Multiprocessing approach...")
        mp_result = self.analyzer.measure_performance(
            lambda: self._multiprocessing_url_check(urls)
        )
        results['multiprocessing'] = self._create_benchmark_result(
            'Multiprocessing', mp_result, len(urls)
        )

        # Asyncio benchmark
        print("4. Asyncio approach...")
        async_result = self.analyzer.measure_performance(
            lambda: asyncio.run(self._asyncio_url_check(urls))
        )
        results['asyncio'] = self._create_benchmark_result(
            'Asyncio', async_result, len(urls)
        )

        return results

    def _sequential_url_check(self, urls: List[str]) -> List[bool]:
        """Sequential URL checking"""
        results = []
        session = requests.Session()
        session.headers.update({'User-Agent': 'Benchmark/1.0'})

        for url in urls:
            try:
                response = session.get(url, timeout=5.0)
                results.append(response.status_code < 400)
            except:
                results.append(False)

        return results

    def _threading_url_check(self, urls: List[str]) -> List[bool]:
        """Threading URL checking"""
        def check_url(url: str) -> bool:
            try:
                response = requests.get(url, timeout=5.0)
                return response.status_code < 400
            except:
                return False

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(check_url, urls))

        return results

    def _multiprocessing_url_check(self, urls: List[str]) -> List[bool]:
        """Multiprocessing URL checking"""
        def check_url(url: str) -> bool:
            try:
                response = requests.get(url, timeout=5.0)
                return response.status_code < 400
            except:
                return False

        with ProcessPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(check_url, urls))

        return results

    async def _asyncio_url_check(self, urls: List[str]) -> List[bool]:
        """Asyncio URL checking"""
        async def check_url(session: aiohttp.ClientSession, url: str) -> bool:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5.0)) as response:
                    return response.status < 400
            except:
                return False

        connector = aiohttp.TCPConnector(limit=100)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [check_url(session, url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return [r if isinstance(r, bool) else False for r in results]

    def _create_benchmark_result(self, method: str, perf_data: Dict[str, Any], num_items: int) -> BenchmarkResult:
        """Create a benchmark result from performance data"""
        execution_time = perf_data['execution_time']
        success_count = sum(1 for r in perf_data['result'] if r)

        return BenchmarkResult(
            method=method,
            execution_time=execution_time,
            memory_usage=perf_data['memory_delta'],
            cpu_usage=perf_data['cpu_usage'],
            success_rate=success_count / num_items * 100,
            throughput=num_items / execution_time,
            overhead=perf_data['memory_delta'] / num_items if num_items > 0 else 0,
            scalability_score=num_items / execution_time / perf_data['memory_delta'] if perf_data['memory_delta'] > 0 else 0
        )

    def benchmark_cpu_intensive(self, num_tasks: int = 8) -> Dict[str, Any]:
        """Benchmark CPU-intensive tasks"""
        print(f"\nBenchmarking CPU-intensive tasks ({num_tasks} tasks)")
        print("=" * 50)

        task_size = 1000000

        # Sequential
        start_time = time.time()
        sequential_results = [self.gil_analyzer.cpu_bound_task(task_size) for _ in range(num_tasks)]
        sequential_time = time.time() - start_time

        # Threading
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_tasks) as executor:
            thread_results = list(executor.map(
                lambda x: self.gil_analyzer.cpu_bound_task(task_size),
                range(num_tasks)
            ))
        thread_time = time.time() - start_time

        # Multiprocessing
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=min(num_tasks, mp.cpu_count())) as executor:
            mp_results = list(executor.map(
                lambda x: self.gil_analyzer.cpu_bound_task(task_size),
                range(num_tasks)
            ))
        mp_time = time.time() - start_time

        return {
            'sequential_time': sequential_time,
            'thread_time': thread_time,
            'mp_time': mp_time,
            'thread_speedup': sequential_time / thread_time,
            'mp_speedup': sequential_time / mp_time,
            'gil_impact': thread_time / mp_time,
            'results_match': sequential_results == thread_results == mp_results
        }

    def benchmark_io_intensive(self, num_tasks: int = 10) -> Dict[str, Any]:
        """Benchmark I/O-intensive tasks"""
        print(f"\nBenchmarking I/O-intensive tasks ({num_tasks} tasks)")
        print("=" * 50)

        delay = 0.5

        # Sequential
        start_time = time.time()
        sequential_results = [self.gil_analyzer.io_bound_task(delay) for _ in range(num_tasks)]
        sequential_time = time.time() - start_time

        # Threading
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_tasks) as executor:
            thread_results = list(executor.map(
                lambda x: self.gil_analyzer.io_bound_task(delay),
                range(num_tasks)
            ))
        thread_time = time.time() - start_time

        # Asyncio
        async def async_io_test():
            tasks = [asyncio.sleep(delay) for _ in range(num_tasks)]
            await asyncio.gather(*tasks)
            return [f"IO task completed after {delay}s" for _ in range(num_tasks)]

        start_time = time.time()
        async_results = asyncio.run(async_io_test())
        async_time = time.time() - start_time

        return {
            'sequential_time': sequential_time,
            'thread_time': thread_time,
            'async_time': async_time,
            'thread_speedup': sequential_time / thread_time,
            'async_speedup': sequential_time / async_time,
            'thread_vs_async': thread_time / async_time
        }

    def display_benchmark_results(self, results: Dict[str, BenchmarkResult]):
        """Display benchmark results in formatted table"""
        print("\nBenchmark Results Summary")
        print("=" * 100)
        print(f"{'Method':<15} {'Time (s)':<10} {'Memory (MB)':<12} {'CPU %':<8} {'Success %':<10} {'Throughput':<12} {'Scalability':<12}")
        print("=" * 100)

        for method, result in results.items():
            print(f"{result.method:<15} {result.execution_time:<10.2f} {result.memory_usage:<12.2f} {result.cpu_usage:<8.2f} {result.success_rate:<10.1f} {result.throughput:<12.2f} {result.scalability_score:<12.4f}")

        print("=" * 100)

        # Calculate speedups relative to sequential
        if 'sequential' in results:
            baseline_time = results['sequential'].execution_time
            print(f"\nSpeedup Analysis (vs Sequential):")
            print("-" * 40)
            for method, result in results.items():
                if method != 'sequential':
                    speedup = baseline_time / result.execution_time
                    print(f"{result.method:<15}: {speedup:.2f}x")

    def memory_usage_analysis(self, num_items: int = 1000):
        """Analyze memory usage patterns"""
        print(f"\nMemory Usage Analysis ({num_items} items)")
        print("=" * 50)

        # Test data creation
        test_data = list(range(num_items))

        # Sequential processing
        start_memory = self.analyzer.process.memory_info().rss / 1024 / 1024
        sequential_result = [x * x for x in test_data]
        sequential_memory = self.analyzer.process.memory_info().rss / 1024 / 1024

        # Threading processing
        def square_number(x):
            return x * x

        start_memory_thread = self.analyzer.process.memory_info().rss / 1024 / 1024
        with ThreadPoolExecutor(max_workers=4) as executor:
            thread_result = list(executor.map(square_number, test_data))
        thread_memory = self.analyzer.process.memory_info().rss / 1024 / 1024

        # Multiprocessing processing
        start_memory_mp = self.analyzer.process.memory_info().rss / 1024 / 1024
        with ProcessPoolExecutor(max_workers=4) as executor:
            mp_result = list(executor.map(square_number, test_data))
        mp_memory = self.analyzer.process.memory_info().rss / 1024 / 1024

        print(f"Sequential memory usage: {sequential_memory - start_memory:.2f} MB")
        print(f"Threading memory usage: {thread_memory - start_memory_thread:.2f} MB")
        print(f"Multiprocessing memory usage: {mp_memory - start_memory_mp:.2f} MB")

        return {
            'sequential_memory': sequential_memory - start_memory,
            'thread_memory': thread_memory - start_memory_thread,
            'mp_memory': mp_memory - start_memory_mp
        }

def get_test_urls() -> List[str]:
    """Get test URLs for benchmarking"""
    return [
        "https://httpbin.org/status/200",
        "https://httpbin.org/status/201",
        "https://httpbin.org/status/202",
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/2",
        "https://www.google.com",
        "https://www.github.com",
        "https://www.python.org",
        "https://stackoverflow.com",
        "https://docs.python.org",
        "https://realpython.com",
        "https://fastapi.tiangolo.com"
    ]

def main():
    """Main benchmark execution"""
    print("Lab 1.4: Performance Comparison & GIL Analysis")
    print("=" * 60)

    # System information
    print(f"\nSystem Information:")
    print(f"Python version: {sys.version}")
    print(f"CPU count: {mp.cpu_count()}")
    print(f"Memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")

    # Initialize benchmark suite
    benchmark = ComprehensiveBenchmark()

    # GIL Analysis
    print("\n" + "="*60)
    print("GIL IMPACT ANALYSIS")
    print("="*60)

    # CPU-bound GIL analysis
    cpu_results = benchmark.gil_analyzer.analyze_gil_impact("cpu_bound", 4)
    print(f"\nCPU-bound task results:")
    print(f"Sequential time: {cpu_results['sequential']:.2f}s")
    print(f"Threading time: {cpu_results['threading']:.2f}s")
    print(f"Multiprocessing time: {cpu_results['multiprocessing']:.2f}s")
    print(f"Threading speedup: {cpu_results['thread_speedup']:.2f}x")
    print(f"Multiprocessing speedup: {cpu_results['mp_speedup']:.2f}x")
    print(f"GIL efficiency (lower is better): {cpu_results['gil_efficiency']:.2f}")

    # I/O-bound GIL analysis
    io_results = benchmark.gil_analyzer.analyze_gil_impact("io_bound", 4)
    print(f"\nI/O-bound task results:")
    print(f"Sequential time: {io_results['sequential']:.2f}s")
    print(f"Threading time: {io_results['threading']:.2f}s")
    print(f"Multiprocessing time: {io_results['multiprocessing']:.2f}s")
    print(f"Threading speedup: {io_results['thread_speedup']:.2f}x")
    print(f"Multiprocessing speedup: {io_results['mp_speedup']:.2f}x")
    print(f"GIL efficiency: {io_results['gil_efficiency']:.2f}")

    # Comprehensive CPU benchmark
    cpu_benchmark = benchmark.benchmark_cpu_intensive(6)
    print(f"\nCPU-intensive benchmark:")
    print(f"Sequential: {cpu_benchmark['sequential_time']:.2f}s")
    print(f"Threading: {cpu_benchmark['thread_time']:.2f}s ({cpu_benchmark['thread_speedup']:.2f}x)")
    print(f"Multiprocessing: {cpu_benchmark['mp_time']:.2f}s ({cpu_benchmark['mp_speedup']:.2f}x)")
    print(f"GIL impact factor: {cpu_benchmark['gil_impact']:.2f}")

    # Comprehensive I/O benchmark
    io_benchmark = benchmark.benchmark_io_intensive(8)
    print(f"\nI/O-intensive benchmark:")
    print(f"Sequential: {io_benchmark['sequential_time']:.2f}s")
    print(f"Threading: {io_benchmark['thread_time']:.2f}s ({io_benchmark['thread_speedup']:.2f}x)")
    print(f"Asyncio: {io_benchmark['async_time']:.2f}s ({io_benchmark['async_speedup']:.2f}x)")
    print(f"Threading vs Asyncio: {io_benchmark['thread_vs_async']:.2f}")

    # URL checking benchmark
    test_urls = get_test_urls()
    url_results = benchmark.benchmark_url_checking(test_urls)
    benchmark.display_benchmark_results(url_results)

    # Memory usage analysis
    memory_results = benchmark.memory_usage_analysis(10000)

    # Decision matrix
    print("\n" + "="*60)
    print("DECISION MATRIX")
    print("="*60)

    print("\nWhen to use each approach:")
    print("\n1. THREADING:")
    print("   ✓ I/O-bound tasks (file operations, network requests)")
    print("   ✓ Tasks that spend time waiting")
    print("   ✓ Moderate concurrency levels (10-100 tasks)")
    print("   ✗ CPU-bound tasks (limited by GIL)")
    print("   ✗ High-concurrency scenarios (thousands of tasks)")

    print("\n2. MULTIPROCESSING:")
    print("   ✓ CPU-bound tasks (calculations, data processing)")
    print("   ✓ True parallelism needed")
    print("   ✓ Tasks that can be easily parallelized")
    print("   ✗ I/O-bound tasks (overhead outweighs benefits)")
    print("   ✗ Shared state requirements")

    print("\n3. ASYNCIO:")
    print("   ✓ High-concurrency I/O-bound tasks")
    print("   ✓ Network programming (servers, clients)")
    print("   ✓ Cooperative multitasking")
    print("   ✓ Event-driven applications")
    print("   ✗ CPU-bound tasks")
    print("   ✗ Blocking operations")

    print("\n4. SEQUENTIAL:")
    print("   ✓ Simple, single-threaded tasks")
    print("   ✓ Debugging and development")
    print("   ✓ Low-latency requirements")
    print("   ✗ Any concurrent workload")

    # Performance recommendations
    print("\n" + "="*60)
    print("PERFORMANCE RECOMMENDATIONS")
    print("="*60)

    print("\nBased on benchmark results:")

    # Find best performer for each category
    best_overall = min(url_results.items(), key=lambda x: x[1].execution_time)
    best_memory = min(url_results.items(), key=lambda x: x[1].memory_usage)
    best_throughput = max(url_results.items(), key=lambda x: x[1].throughput)

    print(f"\nBest overall performance: {best_overall[0]} ({best_overall[1].execution_time:.2f}s)")
    print(f"Most memory efficient: {best_memory[0]} ({best_memory[1].memory_usage:.2f} MB)")
    print(f"Highest throughput: {best_throughput[0]} ({best_throughput[1].throughput:.2f} ops/s)")

    print("\nKey takeaways:")
    print("1. GIL significantly impacts CPU-bound threading performance")
    print("2. Asyncio excels at I/O-bound high-concurrency tasks")
    print("3. Multiprocessing has overhead but provides true parallelism")
    print("4. Choose based on workload characteristics, not assumptions")
    print("5. Always benchmark with realistic workloads")

if __name__ == "__main__":
    main()