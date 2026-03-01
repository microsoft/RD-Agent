"""Rich-based terminal UI for pipeline output.

Provides a shared Console and formatting helpers used across
runner.py, phases.py, stream.py, and client.py.
"""

from __future__ import annotations

import time

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

console = Console(highlight=False)


# ---------------------------------------------------------------------------
# Colour / style tokens (centralised so every module is consistent)
# ---------------------------------------------------------------------------
STYLE_HEADER = "bold bright_white"
STYLE_SUBHEADER = "bold cyan"
STYLE_KEY = "dim"
STYLE_VALUE = "bright_white"
STYLE_SUCCESS = "bold green"
STYLE_WARNING = "bold yellow"
STYLE_ERROR = "bold red"
STYLE_DIM = "dim"
STYLE_TOOL = "bold cyan"
STYLE_TOOL_DETAIL = "cyan"
STYLE_THINKING = "dim italic"
STYLE_AGENT = "bright_magenta"
STYLE_SCORE = "bold bright_green"
STYLE_PHASE_LABEL = "bold bright_blue"


# ---------------------------------------------------------------------------
# Pipeline header & footer
# ---------------------------------------------------------------------------
def print_pipeline_header(task: str, base_model: str, workspace: str,
                          data_path: str, output_dir: str,
                          max_iterations: int, training_timeout: int,
                          max_agent_steps: int, opencode_model: str,
                          resume: bool):
    """Print the main pipeline banner with config table."""
    title = Text("OpenCode RL Pipeline", style="bold bright_white")

    tbl = Table(show_header=False, box=None, padding=(0, 2),
                show_edge=False)
    tbl.add_column("Key", style=STYLE_KEY, min_width=18)
    tbl.add_column("Value", style=STYLE_VALUE)

    tbl.add_row("Task", task)
    tbl.add_row("Base Model", base_model)
    tbl.add_row("Workspace", workspace)
    tbl.add_row("Data Path", data_path)
    tbl.add_row("Output Dir", output_dir)
    tbl.add_row("Max Iterations", str(max_iterations))
    tbl.add_row("Training Timeout", f"{training_timeout}s")
    tbl.add_row("Agent Steps/Iter", str(max_agent_steps))
    tbl.add_row("OpenCode Model", opencode_model or "(env)")
    tbl.add_row("Resume", str(resume))

    console.print(Panel(tbl, title=title, border_style="bright_blue",
                        box=box.DOUBLE_EDGE, padding=(1, 2)))


def print_data_gpu_info(data_count: int, gpu_count: int, gpu_name: str):
    """Print data & GPU summary line."""
    console.print(f"  [dim]Data:[/] [bright_white]{data_count} samples[/]"
                  f"   [dim]GPU:[/] [bright_white]{gpu_count}x {gpu_name}[/]")


def print_pipeline_footer(best_score, best_iteration, total_iters, total_time):
    """Print the final summary panel."""
    tbl = Table(show_header=False, box=None, padding=(0, 2),
                show_edge=False)
    tbl.add_column("Key", style=STYLE_KEY, min_width=16)
    tbl.add_column("Value", style=STYLE_VALUE)

    score_str = f"{best_score}" if best_score is not None else "N/A"
    tbl.add_row("Best Score", f"[bold bright_green]{score_str}[/]")
    tbl.add_row("Best Iteration", str(best_iteration) if best_iteration else "N/A")
    tbl.add_row("Total Iterations", str(total_iters))
    tbl.add_row("Total Time", f"{total_time:.0f}s")

    console.print(Panel(tbl,
                        title=Text("Pipeline Complete", style="bold bright_green"),
                        border_style="green",
                        box=box.DOUBLE_EDGE,
                        padding=(1, 2)))


# ---------------------------------------------------------------------------
# Iteration header & summary
# ---------------------------------------------------------------------------
def print_iteration_header(iteration: int, max_iterations: int, elapsed: float):
    """Print iteration banner."""
    console.print()
    console.print(Rule(
        f"[bold]ITERATION {iteration}/{max_iterations}[/]  "
        f"[dim](elapsed {elapsed:.0f}s)[/]",
        style="bright_yellow",
    ))


def print_iteration_summary(iteration, score, improvement,
                            best_score, best_iteration, elapsed):
    """Print compact iteration summary."""
    tbl = Table(show_header=False, box=None, padding=(0, 1),
                show_edge=False)
    tbl.add_column("Key", style=STYLE_KEY, min_width=18)
    tbl.add_column("Value")

    s = f"[bright_green]{score}[/]" if score is not None else "[dim]N/A[/]"
    imp = f"{improvement}" if improvement is not None else "N/A"
    b = f"[bold bright_green]{best_score}[/] (iter {best_iteration})" if best_score is not None else "N/A"

    tbl.add_row("Score", s)
    tbl.add_row("vs Baseline", imp)
    tbl.add_row("Best So Far", b)
    tbl.add_row("Iteration Time", f"{elapsed:.0f}s")

    console.print(Panel(tbl,
                        title=Text(f"Iteration {iteration} Summary", style="bold"),
                        border_style="bright_yellow", box=box.ROUNDED,
                        padding=(0, 2)))


# ---------------------------------------------------------------------------
# Phase headers
# ---------------------------------------------------------------------------
def print_phase_header(phase_name: str, subtitle: str = ""):
    """Print a phase divider."""
    label = f"[{STYLE_PHASE_LABEL}]{phase_name}[/]"
    if subtitle:
        label += f"  [dim]{subtitle}[/]"
    console.print()
    console.print(Rule(label, style="blue"))


def print_phase_status(msg: str, style: str = ""):
    """Print a phase status line (indented)."""
    if style:
        console.print(f"  [{style}]{msg}[/]")
    else:
        console.print(f"  {msg}")


# ---------------------------------------------------------------------------
# Evaluation report
# ---------------------------------------------------------------------------
def print_evaluation_report(score, improvement, best_score, submission_id=None):
    """Print evaluation results panel."""
    tbl = Table(show_header=False, box=None, padding=(0, 2),
                show_edge=False)
    tbl.add_column("Key", style=STYLE_KEY, min_width=20)
    tbl.add_column("Value")

    tbl.add_row("Source", "[green]Grading Server[/]")
    tbl.add_row("Score",
                f"[bold bright_green]{score}[/]" if score is not None else "N/A")
    tbl.add_row("vs Baseline",
                f"{improvement}" if improvement is not None else "N/A")
    tbl.add_row("Best Score",
                f"[bold bright_green]{best_score}[/]" if best_score is not None else "N/A")
    if submission_id is not None:
        tbl.add_row("Submission ID", str(submission_id))

    console.print(Panel(tbl,
                        title=Text("Evaluation Results", style="bold"),
                        border_style="bright_green", box=box.ROUNDED,
                        padding=(0, 2)))


# ---------------------------------------------------------------------------
# Agent monitor lines (used by client.py _tail_token_log)
# ---------------------------------------------------------------------------
def print_agent_thinking(elapsed: float):
    """LLM is generating tokens (no tool call)."""
    console.print(f"    [{STYLE_THINKING}]... LLM thinking ({elapsed:.0f}s)[/]")


def print_agent_tool_call(tool_name: str, elapsed: float):
    """Agent started a server-side tool call (from [TOOL] event)."""
    console.print(f"    [{STYLE_TOOL}]... agent calling: {tool_name} ({elapsed:.0f}s)[/]")


def print_agent_tool_detail(detail: str):
    """Tool call completed with detail info (from [TOOL_DETAIL] event)."""
    # Truncate very long details
    if len(detail) > 120:
        detail = detail[:117] + "..."
    console.print(f"    [{STYLE_TOOL_DETAIL}]    {detail}[/]")


def print_agent_token_line(text: str):
    """A line of streamed token content."""
    if len(text) > 200:
        text = text[:197] + "..."
    console.print(f"    [dim]| {text}[/]")


# ---------------------------------------------------------------------------
# Stream printer (Turn-level output used by stream.py)
# ---------------------------------------------------------------------------
def print_turn_header(label: str, turn: int, elapsed: float):
    """Print a turn header for agent interaction."""
    console.print(
        f"\n  [{STYLE_SUBHEADER}][{label}][/] "
        f"Turn {turn} [dim]({elapsed:.0f}s)[/]"
    )


def print_turn_done(label: str, turn: int, elapsed: float):
    """Print turn completion."""
    console.print(
        f"  [{STYLE_SUCCESS}][{label}] Done[/] "
        f"[dim](turn {turn}, {elapsed:.0f}s)[/]"
    )


def print_turn_waiting(label: str):
    """Print waiting message."""
    console.print(f"  [{STYLE_DIM}][{label}] Waiting for agent response...[/]")


def print_agent_thought(text: str):
    """Print agent reasoning summary."""
    console.print(f"    [{STYLE_AGENT}]Agent:[/] {text}")


def print_tool_call(kind: str, detail: str):
    """Print a tool call (from Turn data, not proxy)."""
    console.print(f"    [{STYLE_TOOL}]> {kind}:[/] {detail}")


def print_tool_result(text: str, ok: bool = True):
    """Print a tool call result."""
    style = "green" if ok else "red"
    console.print(f"    [{style}]< {text}[/]")
