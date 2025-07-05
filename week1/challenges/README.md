# Week 1 Coding Challenges

This directory contains 10 medium-level coding challenges designed to test your understanding of Python concurrency concepts. Each challenge focuses on different aspects of threading, multiprocessing, asyncio, performance analysis, and decision-making frameworks.

## 📋 Challenge Overview

| Challenge | Concept | Difficulty | Time | Focus Area |
|-----------|---------|------------|------|------------|
| **T1** | Threading | Medium | 30-45 min | Thread-safe file downloads |
| **T2** | Threading | Medium | 35-50 min | Producer-consumer patterns |
| **M1** | Multiprocessing | Medium | 40-55 min | Parallel data pipelines |
| **M2** | Multiprocessing | Medium | 30-40 min | Distributed computing |
| **A1** | Asyncio | Medium | 45-60 min | Web scraping & rate limiting |
| **A2** | Asyncio | Medium | 50-65 min | Real-time chat server |
| **P1** | Performance | Medium | 35-45 min | Concurrency profiler |
| **P2** | Performance | Medium | 40-50 min | Memory-efficient processing |
| **D1** | Decision | Medium | 35-45 min | Intelligent advisor |
| **D2** | Decision | Medium | 45-55 min | Dynamic model switching |

## 🎯 Learning Objectives

By completing these challenges, you will:

1. **Threading Mastery**: Implement thread-safe operations, manage shared resources, and coordinate multiple threads
2. **Multiprocessing Skills**: Build parallel processing systems, handle inter-process communication, and optimize for CPU-intensive tasks
3. **Asyncio Proficiency**: Create asynchronous applications, manage concurrent I/O operations, and build event-driven systems
4. **Performance Analysis**: Profile and optimize concurrent code, understand GIL implications, and measure efficiency
5. **Decision Making**: Build intelligent systems that can choose optimal concurrency approaches based on workload characteristics

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)
- Understanding of basic Python concepts

### Setup
```bash
# Navigate to the challenges directory
cd week1/challenges

# Install dependencies
pip install -r ../requirements.txt

# Run a specific challenge
python threading_challenge_1.py
```

### Dependencies
All challenges require the following packages:
- `rich` - For beautiful terminal output
- `psutil` - For system performance monitoring
- `aiohttp` - For async HTTP requests
- `websockets` - For WebSocket communication
- `pandas` - For data processing
- `numpy` - For numerical operations
- `matplotlib` - For visualization
- `requests` - For HTTP requests

## 📚 Challenge Details

### 🧵 Threading Challenges

#### T1: Thread-Safe File Download Manager
**File**: `threading_challenge_1.py`
**Time**: 30-45 minutes

Build a multi-threaded file download manager that can:
- Download multiple files concurrently (max 5 concurrent downloads)
- Track progress for each download with thread-safe counters
- Handle failures with automatic retry (max 3 retries per file)
- Provide pause/resume functionality
- Display real-time statistics

**Key Skills**: ThreadPoolExecutor, thread-safe data structures, progress tracking

#### T2: Producer-Consumer Log Processor
**File**: `threading_challenge_2.py`
**Time**: 35-50 minutes

Implement a log processing system with:
- Multiple producers generating log entries (3 threads)
- Multiple consumers processing logs by priority (2 threads)
- Thread-safe priority queues
- Graceful shutdown with proper cleanup
- Rate limiting to prevent memory overflow

**Key Skills**: Producer-consumer patterns, priority queues, thread coordination

### 🔧 Multiprocessing Challenges

#### M1: Parallel Data Processing Pipeline
**File**: `multiprocessing_challenge_1.py`
**Time**: 40-55 minutes

Build a data processing pipeline with:
- Stage 1: Data validation and cleaning
- Stage 2: Data transformation
- Stage 3: Data analysis and aggregation
- Process multiple CSV files simultaneously
- Inter-process communication with queues
- Checkpointing and progress reporting

**Key Skills**: ProcessPoolExecutor, inter-process communication, pipeline architecture

#### M2: Distributed Prime Number Calculator
**File**: `multiprocessing_challenge_2.py`
**Time**: 30-40 minutes

Create a distributed prime calculator featuring:
- Dynamic work distribution among processes
- Load balancing across workers
- Result aggregation using shared memory
- Progress monitoring across all workers
- Fault tolerance for worker failures

**Key Skills**: Process pools, shared memory, work distribution, fault tolerance

### ⚡ Asyncio Challenges

#### A1: Async Web Scraping with Rate Limiting
**File**: `asyncio_challenge_1.py`
**Time**: 45-60 minutes

Build an advanced web scraper with:
- Concurrent scraping of multiple domains
- Per-domain rate limiting with token bucket algorithm
- Circuit breaker pattern for failure handling
- Robots.txt compliance
- Real-time monitoring dashboard

**Key Skills**: Async/await, rate limiting, circuit breakers, HTTP clients

#### A2: Real-Time Chat Server
**File**: `asyncio_challenge_2.py`
**Time**: 50-65 minutes

Implement a WebSocket chat server supporting:
- Multiple concurrent WebSocket connections
- Multiple chat rooms with user management
- Private messaging between users
- Typing indicators and file sharing
- Admin commands and real-time statistics

**Key Skills**: WebSockets, async servers, real-time communication, event handling

### 📊 Performance Challenges

#### P1: Concurrency Performance Profiler
**File**: `performance_challenge_1.py`
**Time**: 35-45 minutes

Build a comprehensive profiler that:
- Generates different workload types (CPU, I/O, mixed)
- Measures execution time, throughput, memory usage
- Analyzes GIL impact on threading performance
- Provides detailed recommendations
- Creates performance visualization charts

**Key Skills**: Performance profiling, GIL analysis, benchmarking, visualization

#### P2: Memory-Efficient Concurrent Data Processor
**File**: `performance_challenge_2.py`
**Time**: 40-50 minutes

Design a memory-efficient processor that:
- Processes datasets larger than available RAM
- Compares memory usage across concurrency models
- Implements streaming data processing
- Dynamically adjusts based on memory pressure
- Detects and prevents memory leaks

**Key Skills**: Memory management, streaming processing, resource optimization

### 🎯 Decision Framework Challenges

#### D1: Intelligent Concurrency Advisor
**File**: `decision_challenge_1.py`
**Time**: 35-45 minutes

Create an AI-like advisor that:
- Parses Python code to identify patterns
- Analyzes workload characteristics (I/O vs CPU)
- Uses machine learning-like scoring algorithms
- Generates specific optimization suggestions
- Provides confidence levels for recommendations

**Key Skills**: AST parsing, pattern recognition, scoring algorithms, code analysis

#### D2: Dynamic Concurrency Switcher
**File**: `decision_challenge_2.py`
**Time**: 45-55 minutes

Build a system that:
- Monitors performance metrics in real-time
- Seamlessly transitions between concurrency models
- Learns from performance history
- Implements hysteresis to prevent oscillation
- Provides real-time monitoring dashboard

**Key Skills**: Runtime optimization, performance monitoring, adaptive systems

## 🎓 Evaluation Criteria

Each challenge will be evaluated based on:

1. **Correctness (40%)**: Does the solution work as specified?
2. **Concurrency Implementation (30%)**: Are concurrency patterns used correctly?
3. **Performance (15%)**: Is the solution efficient and scalable?
4. **Code Quality (10%)**: Is the code well-structured and readable?
5. **Error Handling (5%)**: Are edge cases and errors handled properly?

## 💡 Tips for Success

### Before Starting
- Read the challenge requirements carefully
- Understand the expected inputs and outputs
- Review the starter code and TODO comments
- Plan your approach before coding

### During Implementation
- Start with the core functionality first
- Add error handling and edge cases
- Test with different scenarios
- Use the provided console output for debugging
- Don't hesitate to look up documentation

### Common Pitfalls to Avoid
- **Threading**: Forgetting to use locks for shared data
- **Multiprocessing**: Not handling serialization properly
- **Asyncio**: Mixing blocking and non-blocking operations
- **Performance**: Not considering memory usage
- **General**: Ignoring error handling and edge cases

## 📈 Progress Tracking

Use this checklist to track your progress:

### Threading Challenges
- [ ] T1: Thread-Safe File Download Manager
- [ ] T2: Producer-Consumer Log Processor

### Multiprocessing Challenges
- [ ] M1: Parallel Data Processing Pipeline
- [ ] M2: Distributed Prime Number Calculator

### Asyncio Challenges
- [ ] A1: Async Web Scraping with Rate Limiting
- [ ] A2: Real-Time Chat Server

### Performance Challenges
- [ ] P1: Concurrency Performance Profiler
- [ ] P2: Memory-Efficient Concurrent Data Processor

### Decision Framework Challenges
- [ ] D1: Intelligent Concurrency Advisor
- [ ] D2: Dynamic Concurrency Switcher

## 🔧 Running the Challenges

### Individual Challenge
```bash
python threading_challenge_1.py
```

### With Command Line Arguments
```bash
python threading_challenge_1.py --help
```

### Debug Mode
```bash
python threading_challenge_1.py --debug
```

## 📚 Additional Resources

- [Python Threading Documentation](https://docs.python.org/3/library/threading.html)
- [Python Multiprocessing Documentation](https://docs.python.org/3/library/multiprocessing.html)
- [Python Asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
- [Concurrency Patterns in Python](https://docs.python.org/3/library/concurrent.futures.html)
- [Performance Profiling in Python](https://docs.python.org/3/library/profile.html)

## 🏆 Completion Certificate

Once you complete all challenges, you'll have demonstrated mastery of:
- Thread synchronization and coordination
- Process-based parallelism and IPC
- Asynchronous programming patterns
- Performance analysis and optimization
- Intelligent concurrency decision making

Good luck with your challenges! 🚀