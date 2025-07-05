# Week 1: Quick Start - All Concepts Overview

Welcome to Week 1 of the Advanced Python Concurrency Course! This week provides a comprehensive introduction to all three Python concurrency paradigms through hands-on labs.

## 🎯 Learning Objectives

By the end of this week, you will:
- Understand the fundamental differences between threading, multiprocessing, and asyncio
- Have practical experience implementing the same task using all three approaches
- Know when to choose each concurrency model based on workload characteristics
- Understand the GIL (Global Interpreter Lock) and its impact on performance
- Have a decision framework for choosing the right approach for your projects

## 📋 Prerequisites

- Python 3.8+ installed
- Basic understanding of Python (functions, classes, exception handling)
- Command line/terminal access
- Internet connection (for URL checking examples)

## 🛠️ Setup Instructions

1. **Navigate to the week1 directory:**
   ```bash
   cd week1
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import requests, aiohttp, psutil; print('Dependencies installed successfully!')"
   ```

4. **Run the lab runner:**
   ```bash
   python run_all_labs.py
   ```

## 📚 Lab Overview

### Lab 1.1: URL Checker with Threading
**File:** `labs/lab1_1_threading.py`
**Duration:** 15-20 minutes

**What you'll learn:**
- Threading basics and ThreadPoolExecutor
- Thread synchronization with locks and queues
- Thread-safe operations and shared resources
- Producer-consumer pattern implementation
- Threading performance characteristics

**Key concepts:**
- `threading.Thread` and `ThreadPoolExecutor`
- `threading.Lock`, `threading.RLock`, `threading.Semaphore`
- `queue.Queue` for thread communication
- Thread-local storage with `threading.local()`

**Practical application:**
Build a multi-threaded web scraper that checks URL response times with proper synchronization and error handling.

---

### Lab 1.2: URL Checker with Multiprocessing
**File:** `labs/lab1_2_multiprocessing.py`
**Duration:** 15-20 minutes

**What you'll learn:**
- Multiprocessing fundamentals and ProcessPoolExecutor
- Inter-process communication (IPC) with queues and pipes
- Shared memory and process synchronization
- CPU-bound vs I/O-bound task performance
- Process creation overhead and optimization

**Key concepts:**
- `multiprocessing.Process` and `ProcessPoolExecutor`
- `multiprocessing.Queue`, `multiprocessing.Pipe`
- `multiprocessing.Manager` for shared objects
- `multiprocessing.Value` and `multiprocessing.Array` for shared memory

**Practical application:**
Build a parallel URL checker and compare performance with threading, including CPU-intensive task demonstrations.

---

### Lab 1.3: URL Checker with Asyncio
**File:** `labs/lab1_3_asyncio.py`
**Duration:** 20-25 minutes

**What you'll learn:**
- Asyncio fundamentals and event loop concepts
- async/await syntax and coroutine management
- Async HTTP requests with aiohttp
- Concurrency control with semaphores
- Different async patterns (gather, as_completed, queues)

**Key concepts:**
- `async def` and `await` keywords
- `asyncio.gather()`, `asyncio.as_completed()`
- `asyncio.Queue` and `asyncio.Semaphore`
- `aiohttp.ClientSession` for HTTP requests
- Event loop management

**Practical application:**
Build a high-concurrency async URL checker with multiple implementation patterns and performance optimization techniques.

---

### Lab 1.4: Performance Comparison & GIL Analysis
**File:** `labs/lab1_4_performance_comparison.py`
**Duration:** 20-25 minutes

**What you'll learn:**
- Comprehensive performance benchmarking
- GIL impact analysis on different workload types
- Memory usage patterns across paradigms
- Performance profiling and optimization techniques
- Real-world performance trade-offs

**Key concepts:**
- CPU-bound vs I/O-bound performance characteristics
- GIL efficiency measurements
- Memory profiling with `psutil`
- Benchmark result analysis and interpretation
- Performance optimization strategies

**Practical application:**
Run comprehensive benchmarks comparing all three approaches across different workload types with detailed analysis and recommendations.

---

### Lab 1.5: Quick Decision Framework
**File:** `labs/lab1_5_decision_framework.py`
**Duration:** 15-20 minutes

**What you'll learn:**
- Decision-making framework for choosing concurrency models
- Workload characteristic analysis
- Automated recommendation system
- Trade-off analysis and optimization tips
- Best practices for each paradigm

**Key concepts:**
- Workload classification (CPU-bound, I/O-bound, network-bound)
- Decision matrices and scoring algorithms
- Performance trade-off analysis
- Optimization recommendations
- Interactive decision tools

**Practical application:**
Build and use an interactive decision-making tool that recommends the best concurrency approach based on workload characteristics.

## 🚀 Quick Start Guide

### Option 1: Interactive Menu
```bash
python run_all_labs.py
```
This launches an interactive menu where you can choose which labs to run.

### Option 2: Run All Labs
```bash
python run_all_labs.py --all
```
Runs all 5 labs in sequence with guided progression.

### Option 3: Run Specific Lab
```bash
python run_all_labs.py --lab 1    # Run Lab 1 (Threading)
python run_all_labs.py --lab 2    # Run Lab 2 (Multiprocessing)
python run_all_labs.py --lab 3    # Run Lab 3 (Asyncio)
python run_all_labs.py --lab 4    # Run Lab 4 (Performance)
python run_all_labs.py --lab 5    # Run Lab 5 (Decision Framework)
```

### Option 4: Quick Demo
```bash
python run_all_labs.py --quick
```
Runs a quick demonstration of all three concurrency paradigms.

### Option 5: Performance Benchmark
```bash
python run_all_labs.py --benchmark
```
Runs performance benchmarks comparing all approaches.

### Option 6: Interactive Decision Tool
```bash
python run_all_labs.py --interactive
```
Launches the interactive concurrency decision-making tool.

## 📊 Expected Results

After completing all labs, you should see results similar to:

### Performance Comparison Example:
```
Method           Total Time    Success Rate    Avg Time    Throughput    Speedup
Sequential       15.23s        95.0%          0.761s      1.31 ops/s    1.00x
Threading        3.45s         95.0%          0.172s      5.80 ops/s    4.41x
Multiprocessing  4.67s         95.0%          0.234s      4.28 ops/s    3.26x
Asyncio          2.89s         95.0%          0.144s      6.92 ops/s    5.27x
```

### Decision Framework Example:
```
🎯 Primary Choice: ASYNCIO
🔄 Secondary Choice: THREADING
📊 Confidence: 85%

📋 Reasoning:
• Asyncio chosen for network_bound workload
• High I/O intensity ideal for async operations
• High concurrency level suits asyncio architecture
```

## 🧪 Hands-On Exercises

### Exercise 1: Custom URL List
Modify the test URLs in any lab to check your own websites:
```python
urls = [
    "https://your-website.com",
    "https://api.your-service.com/health",
    # Add more URLs...
]
```

### Exercise 2: Parameter Tuning
Experiment with different parameters:
- Change `max_workers` in threading/multiprocessing
- Adjust `max_concurrent` in asyncio
- Modify timeout values
- Try different batch sizes

### Exercise 3: Error Injection
Add deliberately failing URLs to test error handling:
```python
urls = [
    "https://httpbin.org/status/500",  # Server error
    "https://invalid-domain-xyz.com",  # DNS error
    "https://httpbin.org/delay/10",    # Timeout error
]
```

## 🔍 Key Insights

After completing the labs, you should understand:

1. **Threading is best for:**
   - I/O-bound tasks (file operations, network requests)
   - Moderate concurrency (10-100 tasks)
   - Shared state between tasks
   - Quick implementation and debugging

2. **Multiprocessing is best for:**
   - CPU-bound tasks (calculations, data processing)
   - True parallelism requirements
   - Independent tasks without shared state
   - Utilizing multiple CPU cores

3. **Asyncio is best for:**
   - High-concurrency I/O operations
   - Network programming and web servers
   - Event-driven applications
   - Thousands of concurrent tasks

4. **Sequential is best for:**
   - Single tasks or very few tasks
   - Debugging and development
   - Simple, straightforward processing
   - Low-latency requirements

## 🐛 Troubleshooting

### Common Issues:

1. **Import Errors:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Network Timeouts:**
   - Check internet connection
   - Increase timeout values in code
   - Use alternative test URLs

3. **Performance Variations:**
   - Network latency affects results
   - System load impacts benchmarks
   - Run multiple times for average results

4. **Memory Issues:**
   - Reduce concurrent task count
   - Use smaller test datasets
   - Monitor system resources

### Platform-Specific Notes:

**Windows:**
- Multiprocessing requires `if __name__ == "__main__":` guard
- Some features may have different performance characteristics

**macOS:**
- Default multiprocessing start method is 'spawn'
- Performance may vary on M1/M2 processors

**Linux:**
- Generally best performance for all paradigms
- Default multiprocessing start method is 'fork'

## 📈 Performance Tips

1. **For better results:**
   - Close unnecessary applications
   - Use a stable internet connection
   - Run benchmarks multiple times
   - Consider system specifications

2. **Optimization techniques:**
   - Tune worker/concurrent counts
   - Use connection pooling
   - Implement proper error handling
   - Monitor memory usage

## 🎓 Next Steps

After completing Week 1:

1. **Review your results** and compare performance characteristics
2. **Experiment with different parameters** to see how they affect performance
3. **Try the interactive decision tool** with your own project requirements
4. **Move to Week 2** for deep-dive threading concepts and advanced patterns
5. **Practice** implementing concurrency in your own projects

## 📚 Additional Resources

- [Python Threading Documentation](https://docs.python.org/3/library/threading.html)
- [Python Multiprocessing Documentation](https://docs.python.org/3/library/multiprocessing.html)
- [Python Asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
- [Real Python Concurrency Guide](https://realpython.com/python-concurrency/)
- [David Beazley's Concurrency Talks](https://www.youtube.com/results?search_query=david+beazley+python+concurrency)

## 🏆 Completion Checklist

- [ ] Successfully installed all dependencies
- [ ] Completed Lab 1.1 (Threading)
- [ ] Completed Lab 1.2 (Multiprocessing)
- [ ] Completed Lab 1.3 (Asyncio)
- [ ] Completed Lab 1.4 (Performance Comparison)
- [ ] Completed Lab 1.5 (Decision Framework)
- [ ] Understood performance trade-offs
- [ ] Can choose appropriate concurrency model for different scenarios
- [ ] Ready to proceed to Week 2

**Congratulations! You've completed Week 1 and now have a solid foundation in Python concurrency concepts!** 🎉