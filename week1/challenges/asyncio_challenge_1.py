#!/usr/bin/env python3
"""
Asyncio Challenge 1: Async Web Scraping with Rate Limiting

Build an async web scraper that extracts data from multiple websites while
respecting rate limits, handling failures, and providing real-time monitoring.

Requirements:
1. Scrape multiple websites concurrently (different domains)
2. Implement rate limiting per domain
3. Handle failures with retry strategies
4. Provide real-time monitoring dashboard
5. Follow robots.txt rules and implement circuit breaker pattern

Time: 45-60 minutes
"""

import asyncio
import aiohttp
import time
import signal
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import json
import re
from collections import defaultdict, deque
from enum import Enum
from rich.console import Console
from rich.table import Table
from rich.live import Live
import xml.etree.ElementTree as ET

console = Console()

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass
class ScrapingTask:
    """Represents a web scraping task"""
    url: str
    domain: str
    task_type: str = "html"  # html, json, xml
    retries: int = 0
    max_retries: int = 3
    priority: int = 1
    metadata: Dict = field(default_factory=dict)

@dataclass
class ScrapingResult:
    """Result of a scraping operation"""
    url: str
    domain: str
    success: bool
    data: Dict = field(default_factory=dict)
    error: Optional[str] = None
    response_time: float = 0.0
    status_code: Optional[int] = None
    timestamp: float = field(default_factory=time.time)

class RateLimiter:
    """Per-domain rate limiter with token bucket algorithm"""

    def __init__(self, requests_per_second: float = 1.0, burst_size: int = 5):
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Acquire a token for making a request"""
        # TODO: Implement token bucket rate limiting:
        # - Calculate tokens to add based on elapsed time
        # - Check if token is available
        # - Update token count and timestamp
        pass

    def get_delay_until_available(self) -> float:
        """Calculate delay until next token is available"""
        # TODO: Calculate delay based on current token state
        pass

class CircuitBreaker:
    """Circuit breaker for handling persistent failures"""

    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0,
                 success_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.state = CircuitBreakerState.CLOSED

    async def call(self, coro):
        """Execute a coroutine through the circuit breaker"""
        # TODO: Implement circuit breaker logic:
        # - Check current state
        # - Handle OPEN state (reject calls)
        # - Handle HALF_OPEN state (test recovery)
        # - Handle CLOSED state (normal operation)
        # - Update state based on success/failure
        pass

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        # TODO: Implement reset logic based on timeout
        pass

class DomainManager:
    """Manages scraping configuration per domain"""

    def __init__(self):
        self.domain_configs = {}
        self.rate_limiters = {}
        self.circuit_breakers = {}
        self.robots_cache = {}

    async def get_domain_config(self, domain: str) -> Dict:
        """Get or create configuration for a domain"""
        # TODO: Implement domain configuration management:
        # - Create rate limiter for domain
        # - Create circuit breaker for domain
        # - Load robots.txt rules
        # - Return domain configuration
        pass

    async def is_allowed_by_robots(self, url: str, user_agent: str = "*") -> bool:
        """Check if URL is allowed by robots.txt"""
        # TODO: Implement robots.txt checking:
        # - Parse robots.txt for domain
        # - Cache results
        # - Check if URL is allowed
        pass

    async def _fetch_robots_txt(self, domain: str) -> Optional[RobotFileParser]:
        """Fetch and parse robots.txt for domain"""
        # TODO: Implement robots.txt fetching
        pass

class DataExtractor:
    """Extract data from different content types"""

    @staticmethod
    async def extract_html_data(html: str, url: str) -> Dict:
        """Extract data from HTML content"""
        # TODO: Implement HTML data extraction:
        # - Extract titles, meta tags
        # - Find all links
        # - Extract structured data (JSON-LD, microdata)
        # - Get images and their alt text
        # - Extract text content
        pass

    @staticmethod
    async def extract_json_data(json_text: str, url: str) -> Dict:
        """Extract data from JSON content"""
        # TODO: Implement JSON data extraction
        pass

    @staticmethod
    async def extract_xml_data(xml_text: str, url: str) -> Dict:
        """Extract data from XML content"""
        # TODO: Implement XML data extraction
        pass

class ScrapingStatistics:
    """Track scraping statistics and performance"""

    def __init__(self):
        self.domain_stats = defaultdict(lambda: {
            'requests': 0,
            'successes': 0,
            'failures': 0,
            'avg_response_time': 0.0,
            'response_times': deque(maxlen=100),
            'errors': defaultdict(int),
            'last_request': None
        })
        self.total_requests = 0
        self.start_time = time.time()

    def record_request(self, result: ScrapingResult):
        """Record a scraping request result"""
        # TODO: Implement statistics recording:
        # - Update domain-specific stats
        # - Calculate moving averages
        # - Track error patterns
        # - Update global statistics
        pass

    def get_success_rate(self, domain: str = None) -> float:
        """Calculate success rate for domain or overall"""
        # TODO: Calculate success rate
        pass

    def get_dashboard_data(self) -> Dict:
        """Get data for real-time dashboard"""
        # TODO: Compile dashboard data:
        # - Active scrapers count
        # - Success rates by domain
        # - Response time statistics
        # - Error summaries
        # - Circuit breaker states
        pass

class AsyncWebScraper:
    """Main async web scraper with rate limiting and monitoring"""

    def __init__(self, max_concurrent: int = 10, default_rate_limit: float = 1.0):
        self.max_concurrent = max_concurrent
        self.default_rate_limit = default_rate_limit
        self.session = None
        self.domain_manager = DomainManager()
        self.data_extractor = DataExtractor()
        self.statistics = ScrapingStatistics()
        self.task_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        self.shutdown_event = asyncio.Event()
        self.workers = []

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        console.print("\n[yellow]Received shutdown signal. Stopping scraper...[/yellow]")
        asyncio.create_task(self.shutdown())

    async def add_scraping_task(self, url: str, task_type: str = "html",
                              priority: int = 1, metadata: Dict = None):
        """Add a scraping task to the queue"""
        # TODO: Implement task queuing:
        # - Parse domain from URL
        # - Create ScrapingTask object
        # - Add to task queue
        pass

    async def _worker(self, worker_id: int):
        """Worker coroutine for processing scraping tasks"""
        # TODO: Implement worker logic:
        # - Get tasks from queue
        # - Process tasks with rate limiting
        # - Handle retries and circuit breaker
        # - Put results in result queue
        pass

    async def _scrape_url(self, task: ScrapingTask) -> ScrapingResult:
        """Scrape a single URL"""
        # TODO: Implement URL scraping:
        # - Check robots.txt
        # - Apply rate limiting
        # - Make HTTP request
        # - Extract data based on content type
        # - Handle errors and retries
        # - Measure response time
        pass

    async def _make_request(self, url: str, domain: str) -> tuple:
        """Make HTTP request with proper error handling"""
        # TODO: Implement HTTP request:
        # - Use aiohttp session
        # - Set appropriate headers
        # - Handle timeouts
        # - Return response data and metadata
        pass

    async def _result_processor(self):
        """Process scraping results"""
        # TODO: Implement result processing:
        # - Get results from result queue
        # - Update statistics
        # - Store/process extracted data
        # - Handle errors
        pass

    async def start_scraping(self):
        """Start the scraping process"""
        # TODO: Implement scraper startup:
        # - Create aiohttp session
        # - Start worker coroutines
        # - Start result processor
        # - Start monitoring dashboard
        pass

    async def shutdown(self):
        """Graceful shutdown of scraper"""
        # TODO: Implement graceful shutdown:
        # - Set shutdown event
        # - Wait for workers to finish
        # - Process remaining results
        # - Close aiohttp session
        # - Display final statistics
        pass

    async def _display_dashboard(self):
        """Display real-time scraping dashboard"""
        # TODO: Implement dashboard display:
        # - Create rich table with statistics
        # - Update in real-time
        # - Show domain-specific stats
        # - Display circuit breaker states
        pass

async def create_sample_tasks() -> List[ScrapingTask]:
    """Create sample scraping tasks for testing"""
    # TODO: Create realistic scraping tasks:
    # - Different domains with various content types
    # - Mix of valid and invalid URLs for testing
    # - Different priorities
    sample_urls = [
        "https://httpbin.org/json",
        "https://httpbin.org/html",
        "https://httpbin.org/xml",
        "https://jsonplaceholder.typicode.com/posts/1",
        "https://api.github.com/users/octocat",
        "https://httpbin.org/status/404",  # Will fail
        "https://httpbin.org/delay/10",    # Will timeout
    ]
    # TODO: Convert URLs to ScrapingTask objects
    pass

async def main():
    """Demo the async web scraper"""

    try:
        console.print("[bold green]Async Web Scraper with Rate Limiting[/bold green]")
        console.print("Max concurrent requests: 10")
        console.print("Default rate limit: 1 req/sec per domain")
        console.print("Press Ctrl+C to stop\n")

        # Create scraper
        scraper = AsyncWebScraper(
            max_concurrent=10,
            default_rate_limit=1.0
        )

        # Add sample tasks
        sample_tasks = await create_sample_tasks()
        for task in sample_tasks:
            await scraper.add_scraping_task(
                url=task.url,
                task_type=task.task_type,
                priority=task.priority,
                metadata=task.metadata
            )

        console.print(f"Added {len(sample_tasks)} scraping tasks")

        # Start scraping
        await scraper.start_scraping()

        # Let it run for demo (or until Ctrl+C)
        try:
            await asyncio.sleep(60)  # Run for 60 seconds
        except KeyboardInterrupt:
            pass

        # Shutdown
        await scraper.shutdown()

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

# TODO: Implementation checklist
"""
□ Implement RateLimiter with token bucket algorithm
□ Implement CircuitBreaker with state management
□ Implement DomainManager with robots.txt support
□ Implement DataExtractor for different content types
□ Implement ScrapingStatistics with real-time metrics
□ Implement AsyncWebScraper worker logic
□ Implement HTTP request handling with aiohttp
□ Implement retry strategies and error handling
□ Implement real-time dashboard display
□ Add comprehensive logging and monitoring
□ Test with various websites and failure scenarios
□ Optimize for performance and memory usage
"""