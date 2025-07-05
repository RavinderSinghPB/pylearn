#!/usr/bin/env python3
"""
Week 1 Lab Runner
================

This script runs all Week 1 labs in sequence, providing a comprehensive
introduction to threading, multiprocessing, and asyncio concepts.

Usage:
    python run_all_labs.py [options]

Options:
    --lab <number>    Run specific lab (1-5)
    --quick          Run quick versions of labs
    --benchmark      Run performance benchmarks
    --interactive    Run interactive decision tool
    --help           Show this help message
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add labs directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'labs'))

def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_separator():
    """Print a separator line"""
    print("\n" + "-" * 60)

def run_lab_1():
    """Run Lab 1: Threading basics"""
    print_header("LAB 1: URL Checker with Threading")

    try:
        from lab1_1_threading import main
        main()
    except ImportError as e:
        print(f"Error importing lab1_1_threading: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"Error running Lab 1: {e}")

    print_separator()

def run_lab_2():
    """Run Lab 2: Multiprocessing basics"""
    print_header("LAB 2: URL Checker with Multiprocessing")

    try:
        from lab1_2_multiprocessing import main
        main()
    except ImportError as e:
        print(f"Error importing lab1_2_multiprocessing: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"Error running Lab 2: {e}")

    print_separator()

def run_lab_3():
    """Run Lab 3: Asyncio basics"""
    print_header("LAB 3: URL Checker with Asyncio")

    try:
        from lab1_3_asyncio import main
        import asyncio
        asyncio.run(main())
    except ImportError as e:
        print(f"Error importing lab1_3_asyncio: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"Error running Lab 3: {e}")

    print_separator()

def run_lab_4():
    """Run Lab 4: Performance comparison"""
    print_header("LAB 4: Performance Comparison & GIL Analysis")

    try:
        from lab1_4_performance_comparison import main
        main()
    except ImportError as e:
        print(f"Error importing lab1_4_performance_comparison: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"Error running Lab 4: {e}")

    print_separator()

def run_lab_5():
    """Run Lab 5: Decision framework"""
    print_header("LAB 5: Quick Decision Framework")

    try:
        from lab1_5_decision_framework import main
        main()
    except ImportError as e:
        print(f"Error importing lab1_5_decision_framework: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"Error running Lab 5: {e}")

    print_separator()

def run_quick_demo():
    """Run a quick demonstration of all concepts"""
    print_header("QUICK DEMO: All Concepts Overview")

    print("\n1. Threading Demo:")
    try:
        import threading
        import time

        def worker(name):
            print(f"  Thread {name} starting")
            time.sleep(1)
            print(f"  Thread {name} finished")

        threads = []
        for i in range(3):
            t = threading.Thread(target=worker, args=(f"T{i}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        print("  Threading demo completed!")
    except Exception as e:
        print(f"  Threading demo failed: {e}")

    print("\n2. Multiprocessing Demo:")
    try:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        def cpu_task(n):
            return sum(i * i for i in range(n))

        with ProcessPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(cpu_task, [10000, 20000, 30000]))

        print(f"  Multiprocessing results: {results}")
        print("  Multiprocessing demo completed!")
    except Exception as e:
        print(f"  Multiprocessing demo failed: {e}")

    print("\n3. Asyncio Demo:")
    try:
        import asyncio

        async def async_task(name, delay):
            print(f"  Async task {name} starting")
            await asyncio.sleep(delay)
            print(f"  Async task {name} finished")
            return f"Result from {name}"

        async def async_demo():
            tasks = [
                async_task("A", 0.5),
                async_task("B", 0.3),
                async_task("C", 0.7)
            ]
            results = await asyncio.gather(*tasks)
            print(f"  Asyncio results: {results}")
            print("  Asyncio demo completed!")

        asyncio.run(async_demo())
    except Exception as e:
        print(f"  Asyncio demo failed: {e}")

    print_separator()

def run_benchmark():
    """Run performance benchmarks"""
    print_header("PERFORMANCE BENCHMARKS")

    try:
        from lab1_4_performance_comparison import ComprehensiveBenchmark

        benchmark = ComprehensiveBenchmark()

        # Quick URL benchmark
        test_urls = [
            "https://httpbin.org/status/200",
            "https://httpbin.org/status/201",
            "https://httpbin.org/delay/1",
            "https://www.google.com",
            "https://www.github.com"
        ]

        print("Running URL checking benchmark...")
        results = benchmark.benchmark_url_checking(test_urls)
        benchmark.display_benchmark_results(results)

    except Exception as e:
        print(f"Benchmark failed: {e}")

    print_separator()

def run_interactive():
    """Run interactive decision tool"""
    print_header("INTERACTIVE DECISION TOOL")

    try:
        from lab1_5_decision_framework import InteractiveDecisionTool

        tool = InteractiveDecisionTool()
        tool.run_interactive_session()
    except Exception as e:
        print(f"Interactive tool failed: {e}")

    print_separator()

def show_help():
    """Show help message"""
    print(__doc__)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Week 1 Lab Runner")
    parser.add_argument("--lab", type=int, choices=[1, 2, 3, 4, 5],
                       help="Run specific lab (1-5)")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick demonstration")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run performance benchmarks")
    parser.add_argument("--interactive", action="store_true",
                       help="Run interactive decision tool")
    parser.add_argument("--all", action="store_true",
                       help="Run all labs in sequence")

    args = parser.parse_args()

    if len(sys.argv) == 1:
        # No arguments provided, show menu
        show_menu()
        return

    if args.lab:
        lab_functions = {
            1: run_lab_1,
            2: run_lab_2,
            3: run_lab_3,
            4: run_lab_4,
            5: run_lab_5
        }
        lab_functions[args.lab]()
    elif args.quick:
        run_quick_demo()
    elif args.benchmark:
        run_benchmark()
    elif args.interactive:
        run_interactive()
    elif args.all:
        run_all_labs()
    else:
        show_help()

def show_menu():
    """Show interactive menu"""
    print_header("WEEK 1: CONCURRENCY OVERVIEW LABS")

    print("\nWelcome to Week 1 of the Advanced Python Concurrency Course!")
    print("This week covers the fundamentals of all three concurrency paradigms.")

    while True:
        print("\n" + "=" * 40)
        print("Available Options:")
        print("1. Run Lab 1 (Threading)")
        print("2. Run Lab 2 (Multiprocessing)")
        print("3. Run Lab 3 (Asyncio)")
        print("4. Run Lab 4 (Performance Comparison)")
        print("5. Run Lab 5 (Decision Framework)")
        print("6. Quick Demo (All Concepts)")
        print("7. Performance Benchmark")
        print("8. Interactive Decision Tool")
        print("9. Run All Labs")
        print("0. Exit")

        choice = input("\nEnter your choice (0-9): ").strip()

        if choice == "1":
            run_lab_1()
        elif choice == "2":
            run_lab_2()
        elif choice == "3":
            run_lab_3()
        elif choice == "4":
            run_lab_4()
        elif choice == "5":
            run_lab_5()
        elif choice == "6":
            run_quick_demo()
        elif choice == "7":
            run_benchmark()
        elif choice == "8":
            run_interactive()
        elif choice == "9":
            run_all_labs()
        elif choice == "0":
            print("Goodbye! Happy coding!")
            break
        else:
            print("Invalid choice. Please try again.")

def run_all_labs():
    """Run all labs in sequence"""
    print_header("RUNNING ALL WEEK 1 LABS")

    print("\nThis will run all 5 labs in sequence.")
    print("Each lab demonstrates different concurrency concepts.")
    print("Total estimated time: 10-15 minutes")

    confirm = input("\nContinue? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Cancelled.")
        return

    labs = [
        ("Lab 1: Threading", run_lab_1),
        ("Lab 2: Multiprocessing", run_lab_2),
        ("Lab 3: Asyncio", run_lab_3),
        ("Lab 4: Performance Comparison", run_lab_4),
        ("Lab 5: Decision Framework", run_lab_5)
    ]

    start_time = time.time()

    for lab_name, lab_func in labs:
        print(f"\n{'='*60}")
        print(f"Starting {lab_name}...")
        print(f"{'='*60}")

        try:
            lab_func()
            print(f"✅ {lab_name} completed successfully!")
        except Exception as e:
            print(f"❌ {lab_name} failed: {e}")

        print(f"\nPress Enter to continue to next lab...")
        input()

    total_time = time.time() - start_time

    print_header("ALL LABS COMPLETED")
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print("Congratulations! You've completed Week 1 of the course.")
    print("\nNext steps:")
    print("- Review the performance comparisons")
    print("- Experiment with different parameters")
    print("- Try the interactive decision tool")
    print("- Move on to Week 2 for deep-dive threading concepts")

if __name__ == "__main__":
    main()