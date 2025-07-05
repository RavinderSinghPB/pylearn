"""
Lab 1.5: Quick Decision Framework
=================================

This lab provides a practical decision framework for choosing between threading,
multiprocessing, and asyncio based on workload characteristics and requirements.

Learning Objectives:
- Develop intuition for choosing the right concurrency model
- Create automated decision-making tools
- Understand trade-offs and optimization strategies
- Build practical assessment skills
"""

import time
import math
import threading
import multiprocessing as mp
import asyncio
import aiohttp
import requests
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import sys
import psutil

class WorkloadType(Enum):
    """Types of workloads"""
    CPU_BOUND = "cpu_bound"
    IO_BOUND = "io_bound"
    MEMORY_BOUND = "memory_bound"
    NETWORK_BOUND = "network_bound"
    MIXED = "mixed"

class ConcurrencyModel(Enum):
    """Concurrency models available"""
    SEQUENTIAL = "sequential"
    THREADING = "threading"
    MULTIPROCESSING = "multiprocessing"
    ASYNCIO = "asyncio"

@dataclass
class WorkloadCharacteristics:
    """Characteristics of a workload"""
    workload_type: WorkloadType
    task_count: int
    task_duration: float
    memory_usage: int  # MB
    cpu_intensity: float  # 0-1 scale
    io_intensity: float  # 0-1 scale
    concurrency_level: int
    shared_state: bool
    error_tolerance: float  # 0-1 scale

@dataclass
class RecommendationResult:
    """Result of concurrency model recommendation"""
    primary_choice: ConcurrencyModel
    secondary_choice: ConcurrencyModel
    reasoning: List[str]
    confidence: float  # 0-1 scale
    trade_offs: Dict[str, str]
    optimization_tips: List[str]

class ConcurrencyDecisionEngine:
    """Engine for making concurrency model decisions"""

    def __init__(self):
        self.cpu_count = mp.cpu_count()
        self.memory_total = psutil.virtual_memory().total / 1024 / 1024 / 1024  # GB

    def analyze_workload(self, characteristics: WorkloadCharacteristics) -> RecommendationResult:
        """Analyze workload and recommend concurrency model"""

        # Calculate scores for each model
        scores = {
            ConcurrencyModel.SEQUENTIAL: self._score_sequential(characteristics),
            ConcurrencyModel.THREADING: self._score_threading(characteristics),
            ConcurrencyModel.MULTIPROCESSING: self._score_multiprocessing(characteristics),
            ConcurrencyModel.ASYNCIO: self._score_asyncio(characteristics)
        }

        # Sort by score
        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        primary_choice = sorted_models[0][0]
        secondary_choice = sorted_models[1][0]

        # Calculate confidence based on score difference
        confidence = min(1.0, (sorted_models[0][1] - sorted_models[1][1]) / 10.0)

        # Generate reasoning
        reasoning = self._generate_reasoning(characteristics, primary_choice, scores)

        # Generate trade-offs
        trade_offs = self._generate_trade_offs(primary_choice, secondary_choice)

        # Generate optimization tips
        optimization_tips = self._generate_optimization_tips(primary_choice, characteristics)

        return RecommendationResult(
            primary_choice=primary_choice,
            secondary_choice=secondary_choice,
            reasoning=reasoning,
            confidence=confidence,
            trade_offs=trade_offs,
            optimization_tips=optimization_tips
        )

    def _score_sequential(self, char: WorkloadCharacteristics) -> float:
        """Score sequential processing"""
        score = 5.0  # Base score

        # Penalty for multiple tasks
        if char.task_count > 1:
            score -= min(5.0, char.task_count * 0.1)

        # Bonus for simplicity
        score += 2.0

        # Penalty for I/O intensity
        score -= char.io_intensity * 3.0

        return max(0.0, score)

    def _score_threading(self, char: WorkloadCharacteristics) -> float:
        """Score threading model"""
        score = 7.0  # Base score

        # Bonus for I/O bound tasks
        if char.workload_type == WorkloadType.IO_BOUND:
            score += 5.0
        elif char.workload_type == WorkloadType.NETWORK_BOUND:
            score += 4.0

        # Penalty for CPU bound tasks (GIL)
        if char.workload_type == WorkloadType.CPU_BOUND:
            score -= 4.0

        # Bonus for moderate concurrency
        if 10 <= char.concurrency_level <= 100:
            score += 2.0
        elif char.concurrency_level > 100:
            score -= 1.0

        # Bonus for shared state
        if char.shared_state:
            score += 1.0

        # Penalty for high CPU intensity
        score -= char.cpu_intensity * 3.0

        return max(0.0, score)

    def _score_multiprocessing(self, char: WorkloadCharacteristics) -> float:
        """Score multiprocessing model"""
        score = 6.0  # Base score

        # Bonus for CPU bound tasks
        if char.workload_type == WorkloadType.CPU_BOUND:
            score += 6.0
        elif char.workload_type == WorkloadType.MEMORY_BOUND:
            score += 3.0

        # Penalty for I/O bound tasks
        if char.workload_type == WorkloadType.IO_BOUND:
            score -= 2.0

        # Bonus for high CPU intensity
        score += char.cpu_intensity * 4.0

        # Penalty for shared state
        if char.shared_state:
            score -= 3.0

        # Penalty for high memory usage
        if char.memory_usage > 1000:  # > 1GB
            score -= 2.0

        # Bonus for appropriate concurrency level
        if char.concurrency_level <= self.cpu_count * 2:
            score += 1.0

        return max(0.0, score)

    def _score_asyncio(self, char: WorkloadCharacteristics) -> float:
        """Score asyncio model"""
        score = 6.0  # Base score

        # Bonus for I/O bound tasks
        if char.workload_type == WorkloadType.IO_BOUND:
            score += 4.0
        elif char.workload_type == WorkloadType.NETWORK_BOUND:
            score += 6.0

        # Penalty for CPU bound tasks
        if char.workload_type == WorkloadType.CPU_BOUND:
            score -= 5.0

        # Bonus for high concurrency
        if char.concurrency_level > 100:
            score += 3.0
        elif char.concurrency_level > 1000:
            score += 5.0

        # Bonus for high I/O intensity
        score += char.io_intensity * 3.0

        # Penalty for shared state complexity
        if char.shared_state:
            score -= 1.0

        # Penalty for blocking operations
        if char.cpu_intensity > 0.3:
            score -= 2.0

        return max(0.0, score)

    def _generate_reasoning(self, char: WorkloadCharacteristics, choice: ConcurrencyModel, scores: Dict[ConcurrencyModel, float]) -> List[str]:
        """Generate reasoning for the choice"""
        reasoning = []

        if choice == ConcurrencyModel.THREADING:
            reasoning.append(f"Threading chosen for {char.workload_type.value} workload")
            if char.io_intensity > 0.5:
                reasoning.append("High I/O intensity favors threading due to GIL release")
            if char.shared_state:
                reasoning.append("Shared state easier to manage with threading")
            if char.concurrency_level <= 100:
                reasoning.append("Moderate concurrency level suitable for threading")

        elif choice == ConcurrencyModel.MULTIPROCESSING:
            reasoning.append(f"Multiprocessing chosen for {char.workload_type.value} workload")
            if char.cpu_intensity > 0.5:
                reasoning.append("High CPU intensity benefits from true parallelism")
            if not char.shared_state:
                reasoning.append("Independent tasks suit multiprocessing well")
            if char.concurrency_level <= self.cpu_count * 2:
                reasoning.append("Concurrency level appropriate for available CPUs")

        elif choice == ConcurrencyModel.ASYNCIO:
            reasoning.append(f"Asyncio chosen for {char.workload_type.value} workload")
            if char.io_intensity > 0.7:
                reasoning.append("High I/O intensity ideal for async operations")
            if char.concurrency_level > 100:
                reasoning.append("High concurrency level suits asyncio architecture")
            if char.workload_type == WorkloadType.NETWORK_BOUND:
                reasoning.append("Network-bound tasks excel with asyncio")

        else:  # SEQUENTIAL
            reasoning.append("Sequential processing recommended for simplicity")
            if char.task_count == 1:
                reasoning.append("Single task doesn't benefit from concurrency")
            if char.task_duration < 0.1:
                reasoning.append("Short tasks have overhead > benefit for concurrency")

        return reasoning

    def _generate_trade_offs(self, primary: ConcurrencyModel, secondary: ConcurrencyModel) -> Dict[str, str]:
        """Generate trade-offs for the chosen model"""
        trade_offs = {}

        if primary == ConcurrencyModel.THREADING:
            trade_offs["Advantages"] = "Simple shared state, good for I/O-bound tasks"
            trade_offs["Disadvantages"] = "GIL limits CPU-bound performance"
            trade_offs["Memory"] = "Moderate memory usage, shared memory space"
            trade_offs["Complexity"] = "Medium complexity, need synchronization"

        elif primary == ConcurrencyModel.MULTIPROCESSING:
            trade_offs["Advantages"] = "True parallelism, excellent for CPU-bound tasks"
            trade_offs["Disadvantages"] = "High memory overhead, complex state sharing"
            trade_offs["Memory"] = "High memory usage, separate memory spaces"
            trade_offs["Complexity"] = "High complexity, serialization overhead"

        elif primary == ConcurrencyModel.ASYNCIO:
            trade_offs["Advantages"] = "High concurrency, excellent for I/O-bound tasks"
            trade_offs["Disadvantages"] = "Single-threaded, complex error handling"
            trade_offs["Memory"] = "Low memory usage, efficient event loop"
            trade_offs["Complexity"] = "High complexity, async/await paradigm"

        else:  # SEQUENTIAL
            trade_offs["Advantages"] = "Simple, predictable, easy to debug"
            trade_offs["Disadvantages"] = "No parallelism, poor resource utilization"
            trade_offs["Memory"] = "Low memory usage"
            trade_offs["Complexity"] = "Low complexity"

        return trade_offs

    def _generate_optimization_tips(self, choice: ConcurrencyModel, char: WorkloadCharacteristics) -> List[str]:
        """Generate optimization tips for the chosen model"""
        tips = []

        if choice == ConcurrencyModel.THREADING:
            tips.append("Use ThreadPoolExecutor for task management")
            tips.append("Minimize shared state and use proper synchronization")
            tips.append("Set appropriate worker count (typically 2-4x CPU count for I/O)")
            if char.io_intensity > 0.8:
                tips.append("Consider connection pooling for network operations")

        elif choice == ConcurrencyModel.MULTIPROCESSING:
            tips.append("Use ProcessPoolExecutor for automatic process management")
            tips.append("Minimize data sharing between processes")
            tips.append("Use queues for inter-process communication")
            tips.append("Consider using shared memory for large datasets")
            tips.append(f"Optimal worker count: {self.cpu_count} (CPU count)")

        elif choice == ConcurrencyModel.ASYNCIO:
            tips.append("Use aiohttp for HTTP requests")
            tips.append("Implement proper error handling with try/except")
            tips.append("Use semaphores to limit concurrent operations")
            tips.append("Avoid blocking operations in async functions")
            if char.concurrency_level > 1000:
                tips.append("Consider using asyncio.gather() with batching")

        else:  # SEQUENTIAL
            tips.append("Focus on algorithm optimization")
            tips.append("Use efficient data structures")
            tips.append("Consider caching for repeated operations")

        return tips

class InteractiveDecisionTool:
    """Interactive tool for making concurrency decisions"""

    def __init__(self):
        self.engine = ConcurrencyDecisionEngine()

    def run_interactive_session(self):
        """Run an interactive decision-making session"""
        print("Interactive Concurrency Decision Tool")
        print("=" * 40)

        while True:
            print("\nChoose an option:")
            print("1. Quick assessment")
            print("2. Detailed analysis")
            print("3. Workload examples")
            print("4. Exit")

            choice = input("\nEnter your choice (1-4): ").strip()

            if choice == "1":
                self.quick_assessment()
            elif choice == "2":
                self.detailed_analysis()
            elif choice == "3":
                self.show_examples()
            elif choice == "4":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")

    def quick_assessment(self):
        """Quick assessment questionnaire"""
        print("\nQuick Assessment")
        print("-" * 20)

        # Simple questions
        workload_type = self._ask_workload_type()
        task_count = self._ask_int("How many tasks will you run?", 1, 10000)
        concurrency_level = self._ask_int("How many concurrent operations?", 1, 1000)

        # Create characteristics
        characteristics = WorkloadCharacteristics(
            workload_type=workload_type,
            task_count=task_count,
            task_duration=1.0,  # Default
            memory_usage=100,   # Default
            cpu_intensity=0.5 if workload_type == WorkloadType.CPU_BOUND else 0.1,
            io_intensity=0.8 if workload_type == WorkloadType.IO_BOUND else 0.2,
            concurrency_level=concurrency_level,
            shared_state=False,  # Default
            error_tolerance=0.9  # Default
        )

        # Get recommendation
        recommendation = self.engine.analyze_workload(characteristics)
        self._display_recommendation(recommendation)

    def detailed_analysis(self):
        """Detailed analysis with all parameters"""
        print("\nDetailed Analysis")
        print("-" * 20)

        workload_type = self._ask_workload_type()
        task_count = self._ask_int("Number of tasks", 1, 10000)
        task_duration = self._ask_float("Average task duration (seconds)", 0.001, 3600)
        memory_usage = self._ask_int("Memory usage per task (MB)", 1, 10000)
        cpu_intensity = self._ask_float("CPU intensity (0-1 scale)", 0.0, 1.0)
        io_intensity = self._ask_float("I/O intensity (0-1 scale)", 0.0, 1.0)
        concurrency_level = self._ask_int("Desired concurrency level", 1, 10000)
        shared_state = self._ask_bool("Do tasks share state?")
        error_tolerance = self._ask_float("Error tolerance (0-1 scale)", 0.0, 1.0)

        characteristics = WorkloadCharacteristics(
            workload_type=workload_type,
            task_count=task_count,
            task_duration=task_duration,
            memory_usage=memory_usage,
            cpu_intensity=cpu_intensity,
            io_intensity=io_intensity,
            concurrency_level=concurrency_level,
            shared_state=shared_state,
            error_tolerance=error_tolerance
        )

        recommendation = self.engine.analyze_workload(characteristics)
        self._display_recommendation(recommendation)

    def show_examples(self):
        """Show examples of different workload types"""
        print("\nWorkload Examples")
        print("-" * 20)

        examples = [
            ("Web scraping", WorkloadCharacteristics(
                workload_type=WorkloadType.NETWORK_BOUND,
                task_count=100,
                task_duration=2.0,
                memory_usage=50,
                cpu_intensity=0.1,
                io_intensity=0.9,
                concurrency_level=20,
                shared_state=False,
                error_tolerance=0.8
            )),
            ("Image processing", WorkloadCharacteristics(
                workload_type=WorkloadType.CPU_BOUND,
                task_count=50,
                task_duration=5.0,
                memory_usage=200,
                cpu_intensity=0.9,
                io_intensity=0.1,
                concurrency_level=8,
                shared_state=False,
                error_tolerance=0.95
            )),
            ("Database queries", WorkloadCharacteristics(
                workload_type=WorkloadType.IO_BOUND,
                task_count=200,
                task_duration=0.5,
                memory_usage=30,
                cpu_intensity=0.2,
                io_intensity=0.8,
                concurrency_level=50,
                shared_state=True,
                error_tolerance=0.99
            )),
            ("API server", WorkloadCharacteristics(
                workload_type=WorkloadType.NETWORK_BOUND,
                task_count=1000,
                task_duration=0.1,
                memory_usage=20,
                cpu_intensity=0.1,
                io_intensity=0.9,
                concurrency_level=500,
                shared_state=False,
                error_tolerance=0.95
            ))
        ]

        for name, characteristics in examples:
            print(f"\n{name}:")
            recommendation = self.engine.analyze_workload(characteristics)
            print(f"  Recommendation: {recommendation.primary_choice.value}")
            print(f"  Confidence: {recommendation.confidence:.1%}")
            print(f"  Key reason: {recommendation.reasoning[0] if recommendation.reasoning else 'N/A'}")

    def _ask_workload_type(self) -> WorkloadType:
        """Ask for workload type"""
        print("\nWorkload type:")
        print("1. CPU-bound (calculations, data processing)")
        print("2. I/O-bound (file operations, database queries)")
        print("3. Network-bound (web requests, API calls)")
        print("4. Memory-bound (large data structures)")
        print("5. Mixed workload")

        while True:
            choice = input("Enter choice (1-5): ").strip()
            if choice == "1":
                return WorkloadType.CPU_BOUND
            elif choice == "2":
                return WorkloadType.IO_BOUND
            elif choice == "3":
                return WorkloadType.NETWORK_BOUND
            elif choice == "4":
                return WorkloadType.MEMORY_BOUND
            elif choice == "5":
                return WorkloadType.MIXED
            else:
                print("Invalid choice. Please try again.")

    def _ask_int(self, prompt: str, min_val: int, max_val: int) -> int:
        """Ask for integer input"""
        while True:
            try:
                value = int(input(f"{prompt} ({min_val}-{max_val}): "))
                if min_val <= value <= max_val:
                    return value
                else:
                    print(f"Please enter a value between {min_val} and {max_val}")
            except ValueError:
                print("Please enter a valid integer")

    def _ask_float(self, prompt: str, min_val: float, max_val: float) -> float:
        """Ask for float input"""
        while True:
            try:
                value = float(input(f"{prompt} ({min_val}-{max_val}): "))
                if min_val <= value <= max_val:
                    return value
                else:
                    print(f"Please enter a value between {min_val} and {max_val}")
            except ValueError:
                print("Please enter a valid number")

    def _ask_bool(self, prompt: str) -> bool:
        """Ask for boolean input"""
        while True:
            choice = input(f"{prompt} (y/n): ").strip().lower()
            if choice in ['y', 'yes', 'true', '1']:
                return True
            elif choice in ['n', 'no', 'false', '0']:
                return False
            else:
                print("Please enter 'y' or 'n'")

    def _display_recommendation(self, recommendation: RecommendationResult):
        """Display the recommendation result"""
        print("\n" + "=" * 50)
        print("RECOMMENDATION RESULT")
        print("=" * 50)

        print(f"\n🎯 Primary Choice: {recommendation.primary_choice.value.upper()}")
        print(f"🔄 Secondary Choice: {recommendation.secondary_choice.value.upper()}")
        print(f"📊 Confidence: {recommendation.confidence:.1%}")

        print(f"\n📋 Reasoning:")
        for reason in recommendation.reasoning:
            print(f"  • {reason}")

        print(f"\n⚖️  Trade-offs:")
        for key, value in recommendation.trade_offs.items():
            print(f"  {key}: {value}")

        print(f"\n🔧 Optimization Tips:")
        for tip in recommendation.optimization_tips:
            print(f"  • {tip}")

def create_decision_framework() -> Dict[str, Any]:
    """Create a simple decision framework function"""

    def choose_concurrency_model(
        task_type: str,
        cpu_bound: bool,
        io_bound: bool,
        scalability_needs: str,
        shared_state: bool = False,
        task_count: int = 10
    ) -> Dict[str, Any]:
        """
        Simple decision function for choosing concurrency model

        Args:
            task_type: 'cpu', 'io', 'network', 'mixed'
            cpu_bound: True if CPU-intensive
            io_bound: True if I/O-intensive
            scalability_needs: 'low', 'medium', 'high'
            shared_state: True if tasks share state
            task_count: Number of tasks to execute

        Returns:
            Dictionary with recommendation and reasoning
        """

        scores = {
            'sequential': 0,
            'threading': 0,
            'multiprocessing': 0,
            'asyncio': 0
        }

        # Base scoring
        if task_count == 1:
            scores['sequential'] += 10

        if cpu_bound:
            scores['multiprocessing'] += 8
            scores['threading'] -= 3
            scores['asyncio'] -= 5

        if io_bound:
            scores['threading'] += 6
            scores['asyncio'] += 8
            scores['multiprocessing'] -= 2

        if scalability_needs == 'high':
            scores['asyncio'] += 5
            scores['multiprocessing'] += 3
        elif scalability_needs == 'medium':
            scores['threading'] += 3
            scores['asyncio'] += 2

        if shared_state:
            scores['threading'] += 2
            scores['multiprocessing'] -= 3

        # Task type specific scoring
        if task_type == 'network':
            scores['asyncio'] += 7
            scores['threading'] += 4
        elif task_type == 'cpu':
            scores['multiprocessing'] += 6
        elif task_type == 'io':
            scores['threading'] += 5
            scores['asyncio'] += 6

        # Find best choice
        best_choice = max(scores, key=scores.get)

        return {
            'recommendation': best_choice,
            'scores': scores,
            'reasoning': f"Best choice for {task_type} tasks with {scalability_needs} scalability needs"
        }

    return {
        'choose_concurrency_model': choose_concurrency_model,
        'examples': {
            'web_scraping': choose_concurrency_model('network', False, True, 'high', False, 100),
            'image_processing': choose_concurrency_model('cpu', True, False, 'medium', False, 20),
            'database_queries': choose_concurrency_model('io', False, True, 'medium', True, 50),
            'api_server': choose_concurrency_model('network', False, True, 'high', False, 1000)
        }
    }

def main():
    """Main function demonstrating the decision framework"""
    print("Lab 1.5: Quick Decision Framework")
    print("=" * 40)

    # Create simple decision framework
    framework = create_decision_framework()

    print("\n1. Simple Decision Framework Examples:")
    print("-" * 40)

    examples = framework['examples']
    for name, result in examples.items():
        print(f"\n{name.replace('_', ' ').title()}:")
        print(f"  Recommendation: {result['recommendation']}")
        print(f"  Reasoning: {result['reasoning']}")
        print(f"  Scores: {result['scores']}")

    print("\n2. Interactive Decision Tool:")
    print("-" * 40)

    # Ask user if they want to run interactive tool
    choice = input("\nRun interactive decision tool? (y/n): ").strip().lower()
    if choice in ['y', 'yes']:
        tool = InteractiveDecisionTool()
        tool.run_interactive_session()

    print("\n3. Quick Reference Guide:")
    print("-" * 40)
    print("\n✅ Use THREADING when:")
    print("  • I/O-bound tasks (file ops, network requests)")
    print("  • Moderate concurrency (10-100 tasks)")
    print("  • Shared state between tasks")
    print("  • Quick to implement and debug")

    print("\n⚡ Use MULTIPROCESSING when:")
    print("  • CPU-bound tasks (calculations, data processing)")
    print("  • True parallelism needed")
    print("  • Independent tasks")
    print("  • Can utilize multiple CPU cores")

    print("\n🚀 Use ASYNCIO when:")
    print("  • High-concurrency I/O operations")
    print("  • Network programming")
    print("  • Event-driven applications")
    print("  • Thousands of concurrent tasks")

    print("\n📝 Use SEQUENTIAL when:")
    print("  • Single tasks or very few tasks")
    print("  • Debugging and development")
    print("  • Simple, straightforward processing")

    print("\n🎯 Decision Matrix Summary:")
    print("=" * 50)
    print("Task Type     | Low Concurrency | High Concurrency")
    print("CPU-bound     | Sequential      | Multiprocessing")
    print("I/O-bound     | Threading       | Asyncio")
    print("Network-bound | Threading       | Asyncio")
    print("Mixed         | Threading       | Depends on bottleneck")
    print("=" * 50)

if __name__ == "__main__":
    main()