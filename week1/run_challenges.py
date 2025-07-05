#!/usr/bin/env python3
"""
Challenge Runner for Week 1 Coding Challenges

This script provides an interactive menu to run and manage all coding challenges.
It helps you navigate through the challenges and track your progress.
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

console = Console()

class ChallengeRunner:
    """Interactive challenge runner and progress tracker"""

    def __init__(self):
        self.challenges_dir = Path(__file__).parent / "challenges"
        self.challenges = {
            "Threading": [
                {
                    "id": "T1",
                    "name": "Thread-Safe File Download Manager",
                    "file": "threading_challenge_1.py",
                    "time": "30-45 min",
                    "difficulty": "Medium"
                },
                {
                    "id": "T2",
                    "name": "Producer-Consumer Log Processor",
                    "file": "threading_challenge_2.py",
                    "time": "35-50 min",
                    "difficulty": "Medium"
                }
            ],
            "Multiprocessing": [
                {
                    "id": "M1",
                    "name": "Parallel Data Processing Pipeline",
                    "file": "multiprocessing_challenge_1.py",
                    "time": "40-55 min",
                    "difficulty": "Medium"
                },
                {
                    "id": "M2",
                    "name": "Distributed Prime Number Calculator",
                    "file": "multiprocessing_challenge_2.py",
                    "time": "30-40 min",
                    "difficulty": "Medium"
                }
            ],
            "Asyncio": [
                {
                    "id": "A1",
                    "name": "Async Web Scraping with Rate Limiting",
                    "file": "asyncio_challenge_1.py",
                    "time": "45-60 min",
                    "difficulty": "Medium"
                },
                {
                    "id": "A2",
                    "name": "Real-Time Chat Server",
                    "file": "asyncio_challenge_2.py",
                    "time": "50-65 min",
                    "difficulty": "Medium"
                }
            ],
            "Performance": [
                {
                    "id": "P1",
                    "name": "Concurrency Performance Profiler",
                    "file": "performance_challenge_1.py",
                    "time": "35-45 min",
                    "difficulty": "Medium"
                },
                {
                    "id": "P2",
                    "name": "Memory-Efficient Concurrent Data Processor",
                    "file": "performance_challenge_2.py",
                    "time": "40-50 min",
                    "difficulty": "Medium"
                }
            ],
            "Decision": [
                {
                    "id": "D1",
                    "name": "Intelligent Concurrency Advisor",
                    "file": "decision_challenge_1.py",
                    "time": "35-45 min",
                    "difficulty": "Medium"
                },
                {
                    "id": "D2",
                    "name": "Dynamic Concurrency Switcher",
                    "file": "decision_challenge_2.py",
                    "time": "45-55 min",
                    "difficulty": "Medium"
                }
            ]
        }

        self.progress_file = Path("challenge_progress.json")
        self.progress = self.load_progress()

    def load_progress(self):
        """Load challenge progress from file"""
        if self.progress_file.exists():
            try:
                import json
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def save_progress(self):
        """Save challenge progress to file"""
        try:
            import json
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2)
        except:
            pass

    def mark_challenge_started(self, challenge_id):
        """Mark a challenge as started"""
        self.progress[challenge_id] = {
            "status": "started",
            "start_time": time.time(),
            "attempts": self.progress.get(challenge_id, {}).get("attempts", 0) + 1
        }
        self.save_progress()

    def mark_challenge_completed(self, challenge_id):
        """Mark a challenge as completed"""
        if challenge_id in self.progress:
            self.progress[challenge_id]["status"] = "completed"
            self.progress[challenge_id]["end_time"] = time.time()
        else:
            self.progress[challenge_id] = {
                "status": "completed",
                "end_time": time.time(),
                "attempts": 1
            }
        self.save_progress()

    def display_main_menu(self):
        """Display the main menu"""
        console.clear()

        # Welcome header
        console.print(Panel.fit(
            "[bold green]🚀 Week 1 Coding Challenges[/bold green]\n"
            "[dim]Interactive Challenge Runner[/dim]",
            border_style="green"
        ))

        # Progress overview
        total_challenges = sum(len(challenges) for challenges in self.challenges.values())
        completed = sum(1 for p in self.progress.values() if p.get("status") == "completed")

        console.print(f"\n[bold blue]Progress Overview[/bold blue]")
        console.print(f"Completed: {completed}/{total_challenges} challenges")

        if completed > 0:
            completion_percentage = (completed / total_challenges) * 100
            console.print(f"Completion: {completion_percentage:.1f}%")

        console.print("\n[bold yellow]Available Options:[/bold yellow]")
        console.print("1. 📋 View All Challenges")
        console.print("2. 🎯 Select Challenge by Category")
        console.print("3. 🔍 View Challenge Details")
        console.print("4. 📊 View Progress Report")
        console.print("5. 🏃 Run Specific Challenge")
        console.print("6. 🔄 Reset Progress")
        console.print("7. ❌ Exit")

    def display_challenges_overview(self):
        """Display overview of all challenges"""
        console.clear()

        table = Table(title="All Coding Challenges", show_header=True, header_style="bold magenta")
        table.add_column("ID", style="cyan", width=4)
        table.add_column("Category", style="green", width=15)
        table.add_column("Challenge Name", style="white", width=35)
        table.add_column("Time", style="yellow", width=12)
        table.add_column("Status", style="blue", width=12)

        for category, challenges in self.challenges.items():
            for challenge in challenges:
                challenge_id = challenge["id"]
                status = self.progress.get(challenge_id, {}).get("status", "Not Started")

                # Style status
                if status == "completed":
                    status_style = "[green]✅ Completed[/green]"
                elif status == "started":
                    status_style = "[yellow]🔄 In Progress[/yellow]"
                else:
                    status_style = "[dim]⏳ Not Started[/dim]"

                table.add_row(
                    challenge_id,
                    category,
                    challenge["name"],
                    challenge["time"],
                    status_style
                )

        console.print(table)
        console.print("\nPress Enter to return to main menu...")
        input()

    def display_category_menu(self):
        """Display challenge categories"""
        console.clear()

        console.print("[bold blue]Select a Category:[/bold blue]\n")
        categories = list(self.challenges.keys())

        for i, category in enumerate(categories, 1):
            completed = sum(1 for c in self.challenges[category]
                          if self.progress.get(c["id"], {}).get("status") == "completed")
            total = len(self.challenges[category])

            console.print(f"{i}. {category} ({completed}/{total} completed)")

        console.print(f"{len(categories) + 1}. Back to Main Menu")

        try:
            choice = int(Prompt.ask("Enter your choice", default="1"))
            if 1 <= choice <= len(categories):
                self.display_category_challenges(categories[choice - 1])
            elif choice == len(categories) + 1:
                return
        except (ValueError, KeyboardInterrupt):
            pass

    def display_category_challenges(self, category):
        """Display challenges in a specific category"""
        console.clear()

        console.print(f"[bold green]{category} Challenges[/bold green]\n")

        challenges = self.challenges[category]

        for i, challenge in enumerate(challenges, 1):
            challenge_id = challenge["id"]
            status = self.progress.get(challenge_id, {}).get("status", "Not Started")

            console.print(f"{i}. [{challenge_id}] {challenge['name']}")
            console.print(f"   Time: {challenge['time']}")
            console.print(f"   Status: {status}")
            console.print()

        console.print(f"{len(challenges) + 1}. Back to Categories")

        try:
            choice = int(Prompt.ask("Enter choice to run challenge", default="1"))
            if 1 <= choice <= len(challenges):
                self.run_challenge(challenges[choice - 1])
            elif choice == len(challenges) + 1:
                return
        except (ValueError, KeyboardInterrupt):
            pass

    def run_challenge(self, challenge):
        """Run a specific challenge"""
        console.clear()

        challenge_id = challenge["id"]
        challenge_file = self.challenges_dir / challenge["file"]

        console.print(f"[bold green]Running Challenge {challenge_id}[/bold green]")
        console.print(f"Name: {challenge['name']}")
        console.print(f"Expected Time: {challenge['time']}")
        console.print(f"File: {challenge['file']}")

        if not challenge_file.exists():
            console.print(f"[red]Error: Challenge file {challenge_file} not found![/red]")
            input("Press Enter to continue...")
            return

        if Confirm.ask(f"Start challenge {challenge_id}?", default=True):
            self.mark_challenge_started(challenge_id)

            console.print(f"\n[yellow]Starting challenge...[/yellow]")
            console.print(f"[dim]Running: python {challenge_file}[/dim]\n")

            try:
                # Change to challenges directory
                original_dir = os.getcwd()
                os.chdir(self.challenges_dir)

                # Run the challenge
                result = subprocess.run([sys.executable, challenge["file"]],
                                      capture_output=False, text=True)

                # Return to original directory
                os.chdir(original_dir)

                if result.returncode == 0:
                    console.print(f"\n[green]✅ Challenge {challenge_id} completed successfully![/green]")
                    if Confirm.ask("Mark this challenge as completed?", default=True):
                        self.mark_challenge_completed(challenge_id)
                else:
                    console.print(f"\n[yellow]⚠️  Challenge {challenge_id} finished with errors.[/yellow]")

            except KeyboardInterrupt:
                console.print(f"\n[yellow]Challenge {challenge_id} interrupted by user.[/yellow]")
            except Exception as e:
                console.print(f"\n[red]Error running challenge: {e}[/red]")

            input("Press Enter to continue...")

    def display_progress_report(self):
        """Display detailed progress report"""
        console.clear()

        console.print("[bold blue]📊 Progress Report[/bold blue]\n")

        # Overall statistics
        total_challenges = sum(len(challenges) for challenges in self.challenges.values())
        completed = sum(1 for p in self.progress.values() if p.get("status") == "completed")
        started = sum(1 for p in self.progress.values() if p.get("status") == "started")

        console.print(f"Total Challenges: {total_challenges}")
        console.print(f"Completed: {completed}")
        console.print(f"In Progress: {started}")
        console.print(f"Not Started: {total_challenges - completed - started}")

        if completed > 0:
            completion_percentage = (completed / total_challenges) * 100
            console.print(f"Completion Rate: {completion_percentage:.1f}%")

        # Category breakdown
        console.print("\n[bold yellow]Category Breakdown:[/bold yellow]")

        for category, challenges in self.challenges.items():
            cat_completed = sum(1 for c in challenges
                              if self.progress.get(c["id"], {}).get("status") == "completed")
            cat_total = len(challenges)
            cat_percentage = (cat_completed / cat_total) * 100 if cat_total > 0 else 0

            console.print(f"{category}: {cat_completed}/{cat_total} ({cat_percentage:.1f}%)")

        # Recent activity
        if self.progress:
            console.print("\n[bold yellow]Recent Activity:[/bold yellow]")
            recent = sorted(self.progress.items(),
                          key=lambda x: x[1].get("start_time", 0), reverse=True)[:5]

            for challenge_id, data in recent:
                status = data.get("status", "unknown")
                attempts = data.get("attempts", 1)
                console.print(f"  {challenge_id}: {status} (attempts: {attempts})")

        console.print("\nPress Enter to return to main menu...")
        input()

    def reset_progress(self):
        """Reset all progress"""
        if Confirm.ask("Are you sure you want to reset all progress?", default=False):
            self.progress = {}
            self.save_progress()
            console.print("[green]Progress reset successfully![/green]")
            time.sleep(1)

    def run(self):
        """Main application loop"""
        while True:
            try:
                self.display_main_menu()

                choice = Prompt.ask("Select an option", choices=["1", "2", "3", "4", "5", "6", "7"], default="1")

                if choice == "1":
                    self.display_challenges_overview()
                elif choice == "2":
                    self.display_category_menu()
                elif choice == "3":
                    self.display_challenges_overview()  # Could be more detailed
                elif choice == "4":
                    self.display_progress_report()
                elif choice == "5":
                    # Quick run specific challenge
                    challenge_id = Prompt.ask("Enter challenge ID (T1, T2, M1, etc.)")
                    found = False
                    for category, challenges in self.challenges.items():
                        for challenge in challenges:
                            if challenge["id"].upper() == challenge_id.upper():
                                self.run_challenge(challenge)
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        console.print(f"[red]Challenge {challenge_id} not found![/red]")
                        input("Press Enter to continue...")
                elif choice == "6":
                    self.reset_progress()
                elif choice == "7":
                    console.print("[green]Good luck with your concurrency learning journey! 🚀[/green]")
                    break

            except KeyboardInterrupt:
                console.print("\n[yellow]Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"[red]An error occurred: {e}[/red]")
                input("Press Enter to continue...")

def main():
    """Main entry point"""
    try:
        runner = ChallengeRunner()
        runner.run()
    except Exception as e:
        console.print(f"[red]Failed to start challenge runner: {e}[/red]")
        console.print("[yellow]Make sure you have installed all requirements:[/yellow]")
        console.print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()