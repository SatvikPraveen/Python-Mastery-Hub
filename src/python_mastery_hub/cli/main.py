"""
Interactive CLI for Python Mastery Hub.

Provides a comprehensive command-line interface for exploring Python concepts,
running examples, and practicing with interactive exercises.
"""

# Add src to path for imports
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from python_mastery_hub.core import get_module, list_modules

app = typer.Typer(
    name="python-mastery-hub",
    help="🐍 Interactive Python Learning Platform",
    add_completion=False,
)

console = Console()


@app.command()
def list_all(
    difficulty: Optional[str] = typer.Option(
        None,
        "--difficulty",
        "-d",
        help="Filter by difficulty: beginner, intermediate, advanced",
    )
) -> None:
    """📚 List all available learning modules."""

    console.print("\n🐍 [bold blue]Python Mastery Hub - Learning Modules[/bold blue]\n")

    modules = list_modules()

    if difficulty:
        modules = [m for m in modules if m["difficulty"] == difficulty.lower()]

    if not modules:
        console.print(f"[red]No modules found for difficulty: {difficulty}[/red]")
        return

    table = Table(
        title="Available Learning Modules",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Module", style="cyan", width=20)
    table.add_column("Difficulty", style="yellow", width=12)
    table.add_column("Topics", style="green", width=8)
    table.add_column("Description", style="white", width=50)

    for module in modules:
        table.add_row(
            module["name"],
            module["difficulty"].title(),
            str(len(module["topics"])),
            module["description"][:50] + "..."
            if len(module["description"]) > 50
            else module["description"],
        )

    console.print(table)
    console.print(
        f"\n💡 Use [cyan]python-mastery-hub explore <module_name>[/cyan] to start learning!"
    )


@app.command()
def path(
    difficulty: str = typer.Option(
        "all",
        "--difficulty",
        "-d",
        help="Learning path difficulty: beginner, intermediate, advanced, all",
    )
) -> None:
    """🛤️ Show recommended learning path."""

    try:
        from python_mastery_hub.core import get_learning_path

        learning_path = get_learning_path(difficulty)

        console.print(
            f"\n🎯 [bold green]Recommended Learning Path ({difficulty.title()})[/bold green]\n"
        )

        for i, module_name in enumerate(learning_path, 1):
            module = get_module(module_name)
            module_info = module.get_module_info()

            if i == 1:
                status = "🟢 Start here"
            elif i <= len(learning_path) // 2:
                status = "🟡 Continue"
            else:
                status = "🔴 Advanced"

            console.print(f"{i:2}. {status} [cyan]{module_info['name']}[/cyan]")
            console.print(f"    [dim]{module_info['description']}[/dim]")
            console.print()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@app.command()
def explore(module_name: str = typer.Argument(..., help="Name of the module to explore")) -> None:
    """🔍 Explore a specific learning module with examples."""

    try:
        module = get_module(module_name)
        module_info = module.get_module_info()

        console.print(f"\n🎓 [bold blue]{module_info['name']}[/bold blue]")
        console.print(f"[dim]{module_info['description']}[/dim]\n")

        topics = module.get_topics()
        console.print(f"[bold]Available Topics:[/bold]")
        for i, topic in enumerate(topics, 1):
            console.print(f"  {i}. {topic.replace('_', ' ').title()}")

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("Use [cyan]python-mastery-hub list[/cyan] to see available modules.")


@app.command()
def info() -> None:
    """ℹ️ Show information about Python Mastery Hub."""

    console.print(
        """
🐍 [bold blue]Python Mastery Hub v1.0.0[/bold blue]

A comprehensive, production-ready Python learning platform that demonstrates
mastery of Python concepts, modern development practices, and industry standards.

[bold yellow]Features:[/bold yellow]
• Interactive learning modules from basics to advanced
• Hands-on coding exercises and challenges  
• Real-world examples with explanations
• Professional development practices
• Comprehensive test coverage
• Modern CLI interface with Rich formatting

[bold yellow]Available Commands:[/bold yellow]
• [cyan]list[/cyan]    - Show all available modules
• [cyan]path[/cyan]    - Get recommended learning path
• [cyan]explore[/cyan] - Dive into a specific module
• [cyan]info[/cyan]    - Show this information

[bold yellow]Examples:[/bold yellow]
• python-mastery-hub list --difficulty beginner
• python-mastery-hub explore basics
• python-mastery-hub path --difficulty intermediate

[italic]Happy Learning! 🚀[/italic]
    """
    )


if __name__ == "__main__":
    app()
