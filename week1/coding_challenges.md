# Week 1 Coding Challenges
## Medium-Level Tasks to Test Your Knowledge

Each concept has 2 medium-level coding tasks designed to test your practical understanding and implementation skills. Complete these after finishing the labs to validate your learning.

---

## 🧵 Threading Challenges

### Challenge T1: Thread-Safe File Download Manager
**Difficulty: Medium**
**Time: 30-45 minutes**

**Requirements:**
Build a multi-threaded file download manager that can download multiple files concurrently while maintaining thread safety and providing progress tracking.

**Specifications:**
1. Download multiple files from URLs concurrently (max 5 concurrent downloads)
2. Implement a thread-safe progress tracker showing:
   - Individual file progress (bytes downloaded/total bytes)
   - Overall progress across all downloads
   - Download speed for each file
3. Handle download failures with automatic retry (max 3 retries per file)
4. Ensure thread-safe file writing (no corruption)
5. Provide ability to pause/resume downloads
6. Display real-time statistics in a formatted table

**Expected Output:**
```
Download Manager Status:
================================================================
File                    Progress    Speed       Status    Retries
================================================================
file1.zip              45.2%       1.2 MB/s    Downloading   0
file2.pdf              100%        856 KB/s    Complete      1
file3.mp4              12.8%       2.1 MB/s    Downloading   0
================================================================
Overall Progress: 52.7% | Active Downloads: 2 | Completed: 1
```

**Key Concepts Tested:**
- ThreadPoolExecutor usage
- Thread synchronization with locks
- Shared state management
- Error handling in threaded environments
- Progress tracking and reporting

**Starter Code Location:** `challenges/threading_challenge_1.py`

---

### Challenge T2: Producer-Consumer Log Processor
**Difficulty: Medium**
**Time: 35-50 minutes**

**Requirements:**
Implement a multi-threaded log processing system with multiple producers generating log entries and multiple consumers processing them with different priorities.

**Specifications:**
1. **Producers (3 threads):** Generate log entries with different log levels (DEBUG, INFO, WARN, ERROR, CRITICAL)
2. **Consumers (2 threads):** Process logs based on priority (CRITICAL/ERROR processed first)
3. **Log Processing:**
   - Count occurrences of each log level
   - Extract and count unique IP addresses
   - Detect and count error patterns
   - Calculate processing statistics
4. **Thread Safety:** All counters and shared data must be thread-safe
5. **Graceful Shutdown:** Implement proper cleanup when stopping
6. **Rate Limiting:** Limit producers to prevent memory overflow

**Expected Output:**
```
Log Processing Statistics:
==========================
Total Logs Processed: 1,247
Log Level Counts:
  CRITICAL: 23
  ERROR: 156
  WARN: 344
  INFO: 502
  DEBUG: 222

Unique IP Addresses: 89
Error Patterns Detected: 34
Processing Rate: 85.3 logs/sec
Queue Status: 12 pending
```

**Key Concepts Tested:**
- Producer-consumer pattern
- Priority queues
- Thread synchronization primitives
- Graceful shutdown patterns
- Rate limiting and backpressure

**Starter Code Location:** `challenges/threading_challenge_2.py`

---

## ⚡ Multiprocessing Challenges

### Challenge M1: Parallel Data Processing Pipeline
**Difficulty: Medium**
**Time: 40-55 minutes**

**Requirements:**
Build a parallel data processing pipeline that processes large CSV files using multiple processes with different stages of data transformation.

**Specifications:**
1. **Stage 1:** Data validation and cleaning (remove invalid rows, standardize formats)
2. **Stage 2:** Data transformation (calculations, aggregations, feature engineering)
3. **Stage 3:** Data analysis (statistical analysis, pattern detection)
4. **Pipeline Features:**
   - Process multiple CSV files simultaneously
   - Each stage runs in separate processes
   - Use queues for inter-process communication
   - Implement checkpointing (save intermediate results)
   - Handle large files that don't fit in memory
   - Provide progress reporting across processes

**Expected Output:**
```
Data Processing Pipeline Status:
================================
Files in Queue: 3
Stage 1 (Validation): Processing file_2.csv (67% complete)
Stage 2 (Transform): Processing file_1.csv (23% complete)
Stage 3 (Analysis): Processing file_0.csv (89% complete)

Performance Metrics:
- Rows/sec: 12,450
- Memory Usage: 245 MB
- CPU Utilization: 87%
- Completed Files: 7/10
```

**Key Concepts Tested:**
- Process pipeline architecture
- Inter-process communication
- Memory management with large datasets
- Process synchronization
- Performance monitoring

**Starter Code Location:** `challenges/multiprocessing_challenge_1.py`

---

### Challenge M2: Distributed Prime Number Calculator
**Difficulty: Medium**
**Time: 30-40 minutes**

**Requirements:**
Create a distributed prime number calculator that uses multiple processes to find all prime numbers in a large range, with work distribution and result aggregation.

**Specifications:**
1. **Work Distribution:** Divide large range (1 to N) among worker processes
2. **Dynamic Load Balancing:** Redistribute work if some processes finish early
3. **Result Aggregation:** Collect results from all processes safely
4. **Progress Monitoring:** Track progress across all workers
5. **Optimization:** Implement Sieve of Eratosthenes algorithm in parallel
6. **Shared Memory:** Use shared memory for large prime lists
7. **Fault Tolerance:** Handle worker process failures

**Expected Output:**
```
Distributed Prime Calculator
============================
Range: 1 to 10,000,000
Processes: 8
Algorithm: Parallel Sieve of Eratosthenes

Worker Status:
Worker 1: Range 1-1,250,000     [████████████████████] 100% (78,498 primes)
Worker 2: Range 1,250,001-2,500,000  [████████████████████] 100% (70,435 primes)
Worker 3: Range 2,500,001-3,750,000  [██████████████████▓▓] 90%  (65,123 primes)
...

Total Primes Found: 664,579
Elapsed Time: 12.45 seconds
Primes/second: 53,379
```

**Key Concepts Tested:**
- Work distribution algorithms
- Shared memory usage
- Process communication
- Load balancing
- Mathematical algorithm parallelization

**Starter Code Location:** `challenges/multiprocessing_challenge_2.py`

---

## 🚀 Asyncio Challenges

### Challenge A1: Async Web Scraping with Rate Limiting
**Difficulty: Medium**
**Time: 45-60 minutes**

**Requirements:**
Build an async web scraper that extracts data from multiple websites while respecting rate limits, handling failures, and providing real-time monitoring.

**Specifications:**
1. **Scraping Features:**
   - Scrape multiple websites concurrently (different domains)
   - Extract specific data (titles, links, metadata)
   - Handle different response formats (HTML, JSON, XML)
   - Follow robots.txt rules
2. **Rate Limiting:**
   - Different rate limits per domain
   - Implement exponential backoff for failures
   - Queue requests to prevent overwhelming servers
3. **Error Handling:**
   - Retry failed requests with different strategies
   - Handle timeouts, connection errors, HTTP errors
   - Circuit breaker pattern for persistent failures
4. **Monitoring:**
   - Real-time dashboard showing scraping progress
   - Success/failure rates per domain
   - Response time statistics

**Expected Output:**
```
Async Web Scraper Dashboard
===========================
Active Scrapers: 15 | Queue Size: 234 | Success Rate: 94.2%

Domain Statistics:
Domain              Requests  Success  Failed  Avg Time  Rate Limit
================================================================================
example.com         1,247     1,189    58      0.45s     2 req/sec
api.service.com     856       834      22      0.23s     5 req/sec
news.site.org       2,134     2,098    36      0.67s     1 req/sec

Circuit Breakers:
- slow-site.com: OPEN (too many timeouts)
- error-site.net: HALF_OPEN (testing recovery)
```

**Key Concepts Tested:**
- Async HTTP client implementation
- Rate limiting and backpressure
- Error handling and retry strategies
- Circuit breaker pattern
- Real-time monitoring

**Starter Code Location:** `challenges/asyncio_challenge_1.py`

---

### Challenge A2: Real-Time Chat Server
**Difficulty: Medium**
**Time: 50-65 minutes**

**Requirements:**
Implement a real-time chat server using asyncio with WebSocket connections, supporting multiple chat rooms, user management, and message broadcasting.

**Specifications:**
1. **Server Features:**
   - Handle multiple concurrent WebSocket connections
   - Support multiple chat rooms
   - User authentication and management
   - Message broadcasting to room members
   - Private messaging between users
2. **Advanced Features:**
   - Message history and persistence
   - User presence tracking (online/offline)
   - Typing indicators
   - File sharing capabilities
   - Rate limiting per user
3. **Administration:**
   - Admin commands (kick, ban, mute)
   - Room creation and management
   - Real-time server statistics
4. **Performance:**
   - Handle 1000+ concurrent connections
   - Message delivery guarantees
   - Connection management and cleanup

**Expected Output:**
```
Chat Server Status
==================
Active Connections: 1,247
Active Rooms: 23
Messages/minute: 1,856
Uptime: 2d 14h 32m

Room Statistics:
Room Name          Users  Messages  Activity
===============================================
general            234    45,678    High
tech-talk          89     12,345    Medium
random             156    23,456    High

Recent Activity:
[14:32:15] user123 joined #general
[14:32:18] alice: Hello everyone!
[14:32:20] bob is typing in #tech-talk...
[14:32:22] admin kicked spammer from #general
```

**Key Concepts Tested:**
- WebSocket server implementation
- Connection management
- Broadcasting patterns
- State management in async environments
- Real-time communication protocols

**Starter Code Location:** `challenges/asyncio_challenge_2.py`

---

## 📊 Performance Analysis Challenges

### Challenge P1: Concurrency Performance Profiler
**Difficulty: Medium**
**Time: 35-45 minutes**

**Requirements:**
Build a comprehensive performance profiler that analyzes and compares the performance of threading, multiprocessing, and asyncio for different types of workloads.

**Specifications:**
1. **Workload Generation:**
   - Create different workload types (CPU-bound, I/O-bound, mixed)
   - Parameterize workload characteristics (duration, intensity, patterns)
   - Generate realistic test scenarios
2. **Performance Measurement:**
   - Execution time and throughput
   - Memory usage patterns
   - CPU utilization
   - Context switching overhead
   - Scalability analysis
3. **Analysis Features:**
   - GIL impact quantification
   - Bottleneck identification
   - Optimal worker count determination
   - Resource utilization efficiency
4. **Reporting:**
   - Performance comparison charts
   - Recommendation engine
   - Detailed analysis reports

**Expected Output:**
```
Concurrency Performance Analysis Report
=======================================

Workload: I/O-Bound (Network Requests)
Tasks: 100 | Concurrency Levels: [1, 5, 10, 20, 50]

Performance Results:
Method          1 worker  5 workers  10 workers  20 workers  50 workers
========================================================================
Sequential      45.2s     -          -           -           -
Threading       45.1s     9.8s       5.2s        3.1s        2.9s
Multiprocessing 46.3s     11.2s      7.8s        6.1s        7.2s
Asyncio         44.8s     9.1s       4.8s        2.7s        2.4s

Recommendations:
- Best approach: Asyncio with 50 workers
- GIL impact: Minimal (I/O-bound workload)
- Memory efficiency: Asyncio > Threading > Multiprocessing
- Optimal concurrency: 20-50 for this workload
```

**Key Concepts Tested:**
- Performance measurement techniques
- Statistical analysis of benchmarks
- GIL impact analysis
- Resource utilization monitoring
- Recommendation algorithms

**Starter Code Location:** `challenges/performance_challenge_1.py`

---

### Challenge P2: Memory-Efficient Concurrent Data Processor
**Difficulty: Medium**
**Time: 40-50 minutes**

**Requirements:**
Design a memory-efficient concurrent data processor that can handle datasets larger than available RAM while maintaining optimal performance across different concurrency models.

**Specifications:**
1. **Memory Management:**
   - Process data in chunks to stay within memory limits
   - Implement streaming data processing
   - Monitor and control memory usage
   - Garbage collection optimization
2. **Concurrency Strategies:**
   - Compare memory usage across threading/multiprocessing/asyncio
   - Implement memory-aware task scheduling
   - Dynamic worker count adjustment based on memory pressure
3. **Data Processing:**
   - Handle multiple data formats (CSV, JSON, binary)
   - Implement data transformations and aggregations
   - Maintain processing state across chunks
4. **Monitoring:**
   - Real-time memory usage tracking
   - Performance metrics per concurrency model
   - Memory leak detection

**Expected Output:**
```
Memory-Efficient Data Processor
===============================
Dataset: large_data.csv (5.2 GB)
Available Memory: 2.1 GB
Chunk Size: 50 MB

Processing Strategy Analysis:
Method          Memory Peak  Processing Time  Efficiency Score
================================================================
Threading       1.8 GB       142.3s          8.2/10
Multiprocessing 3.2 GB       98.7s           6.1/10 (memory exceeded)
Asyncio         1.2 GB       156.8s          9.1/10

Selected Strategy: Asyncio with dynamic chunking
Memory Usage: [██████▓▓▓▓] 65% of available
Progress: 78% complete | ETA: 34.5s

Optimizations Applied:
- Reduced chunk size due to memory pressure
- Implemented lazy loading for large objects
- Enhanced garbage collection frequency
```

**Key Concepts Tested:**
- Memory management in concurrent environments
- Streaming data processing
- Performance optimization
- Resource monitoring
- Adaptive algorithms

**Starter Code Location:** `challenges/performance_challenge_2.py`

---

## 🎯 Decision Framework Challenges

### Challenge D1: Intelligent Concurrency Advisor
**Difficulty: Medium**
**Time: 35-45 minutes**

**Requirements:**
Create an intelligent advisor system that analyzes code patterns and automatically recommends the best concurrency approach with detailed reasoning and optimization suggestions.

**Specifications:**
1. **Code Analysis:**
   - Parse Python code to identify concurrency patterns
   - Analyze function calls to determine I/O vs CPU bound operations
   - Detect shared state usage and synchronization needs
   - Identify potential race conditions and bottlenecks
2. **Recommendation Engine:**
   - Machine learning-like scoring algorithm
   - Consider multiple factors (workload type, scale, complexity)
   - Provide confidence levels for recommendations
   - Generate alternative approaches with trade-offs
3. **Optimization Suggestions:**
   - Specific code improvements
   - Configuration recommendations
   - Performance tuning tips
   - Architecture patterns

**Expected Output:**
```
Intelligent Concurrency Advisor
===============================

Code Analysis Results:
======================
File: data_processor.py
Functions analyzed: 12
I/O operations detected: 8 (file reads, HTTP requests)
CPU-intensive operations: 2 (data calculations)
Shared state usage: Medium (3 shared variables)
Synchronization complexity: Low

Recommendation: ASYNCIO
Confidence: 92%

Reasoning:
✓ High I/O intensity (67%) favors async operations
✓ Network requests detected - ideal for asyncio
✓ Low CPU intensity - GIL won't be a bottleneck
✓ Moderate shared state can be handled with async patterns

Alternative Approaches:
2. Threading (Score: 78%) - Good for I/O, simpler implementation
3. Multiprocessing (Score: 34%) - Overkill for this workload

Optimization Suggestions:
• Use aiohttp.ClientSession for HTTP requests
• Implement connection pooling
• Add semaphore for rate limiting (max 50 concurrent)
• Consider using asyncio.gather() for parallel operations
```

**Key Concepts Tested:**
- Code analysis and pattern recognition
- Decision algorithm implementation
- Multi-factor scoring systems
- Recommendation confidence calculation
- Optimization strategy generation

**Starter Code Location:** `challenges/decision_challenge_1.py`

---

### Challenge D2: Dynamic Concurrency Switcher
**Difficulty: Medium**
**Time: 45-55 minutes**

**Requirements:**
Build a system that can dynamically switch between different concurrency models at runtime based on changing workload characteristics and performance metrics.

**Specifications:**
1. **Runtime Monitoring:**
   - Monitor performance metrics in real-time
   - Detect workload pattern changes
   - Track resource utilization
   - Identify performance degradation
2. **Dynamic Switching:**
   - Seamlessly transition between concurrency models
   - Migrate active tasks without data loss
   - Maintain state consistency during transitions
   - Handle concurrent operations during switches
3. **Adaptive Algorithms:**
   - Learn from performance history
   - Predict optimal concurrency model
   - Adjust parameters automatically
   - Implement hysteresis to prevent oscillation
4. **Monitoring Dashboard:**
   - Real-time performance visualization
   - Concurrency model history
   - Decision reasoning logs

**Expected Output:**
```
Dynamic Concurrency Switcher
=============================

Current Configuration:
Active Model: ASYNCIO
Worker Count: 25
Switch Count: 3
Uptime: 1h 23m

Performance History:
Time      Model      Workers  Throughput  CPU%  Memory   Decision Reason
=========================================================================
14:32:15  Threading  10       145 ops/s   67%   234 MB   Initial config
14:45:22  Asyncio    20       287 ops/s   45%   189 MB   I/O pattern detected
14:52:18  Asyncio    25       312 ops/s   48%   201 MB   Scale up for load
15:01:45  Asyncio    25       298 ops/s   51%   198 MB   Performance stable

Monitoring Alerts:
⚠ Memory usage trending up - monitoring for potential switch
✓ Throughput within optimal range
✓ No performance degradation detected

Next Evaluation: 15:17:45 (in 16m)
```

**Key Concepts Tested:**
- Runtime performance monitoring
- Dynamic system reconfiguration
- State migration patterns
- Adaptive algorithms
- Decision history and learning

**Starter Code Location:** `challenges/decision_challenge_2.py`

---

## 🏆 Challenge Completion Guidelines

### Evaluation Criteria
Each challenge will be evaluated on:
1. **Correctness** (40%) - Does it work as specified?
2. **Code Quality** (25%) - Clean, readable, well-structured code
3. **Performance** (20%) - Efficient implementation
4. **Error Handling** (10%) - Robust error handling
5. **Documentation** (5%) - Clear comments and docstrings

### Difficulty Progression
- **Basic Understanding** - Complete any 3 challenges
- **Intermediate Mastery** - Complete 6+ challenges with good scores
- **Advanced Proficiency** - Complete all 10 challenges with excellent scores

### Time Expectations
- **Total Time**: 6-8 hours for all challenges
- **Per Challenge**: 30-65 minutes
- **Recommendation**: Complete 1-2 challenges per day

### Getting Started
1. Choose a challenge based on your interest/strength
2. Read requirements carefully
3. Plan your approach before coding
4. Implement incrementally with testing
5. Optimize and refine your solution

### Support Resources
- Week 1 lab code for reference
- Python documentation
- Online resources for specific algorithms
- Peer discussion and code review

**Ready to test your skills? Start with any challenge that interests you most!** 🚀