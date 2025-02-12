# debug_tools.py
import sys
from functools import wraps
from typing import Any, List, Optional
from collections import Counter
import pandas as pd

from rich.console import Console, ConsoleOptions, RenderResult
from rich.table import Table
from rich.text import Text

# Import Node so we can check its type.
from brancharchitect.tree import Node


# ------------------------------------------------------------------
# Custom Renderable: RawHTML
# ------------------------------------------------------------------
class RawHTML:
    """
    A simple renderable that outputs raw HTML (such as an SVG) without escaping.
    """

    def __init__(self, html: str):
        self.html = html

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        # Yield a Text object marked as unsafe so that the HTML is rendered unescaped.
        yield Text.from_markup(self.html)


# ------------------------------------------------------------------
# Global Settings and Theme Colors
# ------------------------------------------------------------------
USE_COLOR = sys.stdout.isatty()
# A simple, clean theme:
COLORS = {
    "var_name": "\033[94m" if USE_COLOR else "",  # bright blue
    "component": "\033[36m" if USE_COLOR else "",  # cyan
    "label": "\033[32m" if USE_COLOR else "",  # green
    "highlight": "\033[35m" if USE_COLOR else "",  # magenta
    "green": "\033[92m" if USE_COLOR else "",  # bright green
    "red": "\033[91m" if USE_COLOR else "",  # bright red
    "reset": "\033[0m" if USE_COLOR else "",
}

_GLOBAL_LEAF_ORDER: Optional[List[str]] = None


def set_leaf_order(leaf_order: List[str]) -> None:
    """Set the global leaf order used for formatting."""
    global _GLOBAL_LEAF_ORDER
    _GLOBAL_LEAF_ORDER = leaf_order


def get_leaf_order() -> Optional[List[str]]:
    """Return the global leaf order."""
    return _GLOBAL_LEAF_ORDER


# ------------------------------------------------------------------
# Utility Formatting Functions
# ------------------------------------------------------------------
def calc_total_leaves_any(structure: Any) -> int:
    if isinstance(structure, int):
        return 1
    if isinstance(structure, (list, tuple, set)):
        return sum(calc_total_leaves_any(x) for x in structure)
    if isinstance(structure, dict):
        return sum(
            calc_total_leaves_any(k) + calc_total_leaves_any(v)
            for k, v in structure.items()
        )
    return 0


def nested_structure_to_str(structure: Any, leaf_order: List[str]) -> str:
    if isinstance(structure, int):
        return (
            f"{COLORS['label']}{leaf_order[structure]}{COLORS['reset']}"
            if 0 <= structure < len(leaf_order)
            else f"idx{structure}"
        )
    if isinstance(structure, dict):
        if not structure:
            return "{}"
        parts = [
            f"{nested_structure_to_str(k, leaf_order)} => {nested_structure_to_str(v, leaf_order)}"
            for k, v in structure.items()
        ]
        return "{" + ", ".join(parts) + "}"
    if isinstance(structure, set):
        if not structure:
            return "{}"
        sorted_items = sorted(structure, key=lambda x: str(x))
        return (
            "{"
            + ", ".join(nested_structure_to_str(x, leaf_order) for x in sorted_items)
            + "}"
        )
    if isinstance(structure, (tuple, list)):
        if not structure:
            return "{}"
        if all(isinstance(x, int) for x in structure):
            sub_labels = [
                (
                    f"{COLORS['label']}{leaf_order[x]}{COLORS['reset']}"
                    if 0 <= x < len(leaf_order)
                    else f"idx{x}"
                )
                for x in structure
            ]
            return "{" + ", ".join(sub_labels) + "}"
        return (
            "{"
            + ", ".join(nested_structure_to_str(x, leaf_order) for x in structure)
            + "}"
        )
    return str(structure)


def _format_value(value: Any, depth: int = 0) -> str:
    leaf_order = get_leaf_order() or []
    return ("  " * depth) + nested_structure_to_str(value, leaf_order)


def _print_structured(name: str, value: Any, prefix: str = ""):
    debug_console.print(f"{COLORS['var_name']}{prefix}{name}:{COLORS['reset']}\n")
    if isinstance(value, list):
        count_str = (
            f"({len(value)} groups)"
            if value and isinstance(value[0], list)
            else f"({len(value)} components)"
        )
        debug_console.print(count_str)
    debug_console.print(_format_value(value, depth=1))
    debug_console.print("")


def _convert_to_hashable(item: Any) -> Any:
    if isinstance(item, list):
        return tuple(_convert_to_hashable(x) for x in item)
    if isinstance(item, dict):
        return tuple((k, _convert_to_hashable(v)) for k, v in item.items())
    if isinstance(item, set):
        return tuple(sorted(_convert_to_hashable(x) for x in item))
    return item


# ------------------------------------------------------------------
# Display Functions
# ------------------------------------------------------------------
def display_truth_table(
    c1: List[Any],
    c2: List[Any],
    intersections: List[Any],
    symmetric_differences: List[Any],
):
    total_pairs = len(c1) * len(c2)
    if len(intersections) != total_pairs or len(symmetric_differences) != total_pairs:
        debug_console.print(
            "[bold red]Truth Table Error: Input lists length mismatch.[/bold red]"
        )
        debug_console.print(
            f"Expected: {total_pairs}, Got: intersections={len(intersections)}, symdiffs={len(symmetric_differences)}"
        )
        return
    table = Table(show_lines=True)
    table.add_column("Tree 1 Component", style="cyan")
    table.add_column("Tree 2 Component", style="magenta")
    table.add_column("Intersection", style="green")
    table.add_column("SymDiff", style="yellow")
    pair_index = 0
    for comp1 in c1:
        for comp2 in c2:
            try:
                inter = _format_value(intersections[pair_index])
                symd = _format_value(symmetric_differences[pair_index])
                table.add_row(_format_value(comp1), _format_value(comp2), inter, symd)
                pair_index += 1
            except IndexError:
                debug_console.print(
                    f"[bold red]IndexError[/bold red] at pair {pair_index}"
                )
                return
    debug_console.print("[bold yellow]=== TRUTH TABLE ===[/bold yellow]")
    debug_console.print(table)


def display_components_comparison(c1: List[Any], c2: List[Any]):
    if isinstance(c1, Node):
        c1 = [c1]
    if isinstance(c2, Node):
        c2 = [c2]
    sorted_c1 = sorted(
        c1, key=lambda comp: len(comp) if hasattr(comp, "__len__") else 1
    )
    sorted_c2 = sorted(
        c2, key=lambda comp: len(comp) if hasattr(comp, "__len__") else 1
    )
    table = Table(show_lines=True)
    table.add_column("Tree 1 Components (sorted by length)", style="cyan")
    table.add_column("Tree 2 Components (sorted by length)", style="magenta")
    max_rows = max(len(sorted_c1), len(sorted_c2))
    for i in range(max_rows):
        col1 = _format_value(sorted_c1[i]) if i < len(sorted_c1) else ""
        col2 = _format_value(sorted_c2[i]) if i < len(sorted_c2) else ""
        table.add_row(col1, col2)
    debug_console.print("[bold green]=== Components Comparison ===[/bold green]")
    debug_console.print(table)


def display_nested_components_comparison(
    nested_c1: List[List[Any]], nested_c2: List[List[Any]]
):
    def format_group(group: List[Any]) -> str:
        sorted_group = sorted(
            group, key=lambda comp: len(comp) if hasattr(comp, "__len__") else 1
        )
        return "\n".join(_format_value(comp) for comp in sorted_group)

    max_groups = max(len(nested_c1), len(nested_c2))
    rows = []
    for i in range(max_groups):
        left = format_group(nested_c1[i]) if i < len(nested_c1) else ""
        right = format_group(nested_c2[i]) if i < len(nested_c2) else ""
        rows.append((left, right))
    table = Table(show_lines=True)
    table.add_column("c1 (sorted by length)", style="cyan")
    table.add_column("c2 (sorted by length)", style="magenta")
    for left, right in rows:
        table.add_row(left, right)
    debug_console.print("[bold blue]=== Nested Components Comparison ===[/bold blue]")
    debug_console.print(table)


def display_structured_table(data: Any, listA: list, listB: list, table_type: str):
    if not (
        isinstance(data, list)
        and len(data) == len(listA)
        and all(isinstance(row, list) and len(row) == len(listB) for row in data)
    ):
        debug_console.print(
            f"{table_type} Table: Shape mismatch (expected {len(listA)}x{len(listB)}), skipping DataFrame.\n"
        )
        return
    df = pd.DataFrame(
        [[_format_value(cell) for cell in row] for row in data],
        index=[f"A{i}" for i in range(len(listA))],
        columns=[f"B{j}" for j in range(len(listB))],
    )
    debug_console.print(f"{table_type} Table:")
    debug_console.print(df)
    debug_console.print("")


# ------------------------------------------------------------------
# Global Debug Console (Write-only Log File)
# ------------------------------------------------------------------
# This console writes exclusively to a log file.
log_file = open("debug_log.html", "w", encoding="utf-8")
debug_console = Console(file=log_file, record=True)


# ------------------------------------------------------------------
# Debug Decorator: watch_variables
# ------------------------------------------------------------------
def watch_variables(*decorator_args, **decorator_kwargs):
    """
    Decorator that traces a set of local variables during function execution.
    The function name is printed as a header.
    Captured variables (including any plot renderables in 'plots') are then
    displayed using the global debug_console.
    """
    var_names = decorator_kwargs.get(
        "var_names",
        [
            "c1",
            "c2",
            "c12",
            "intersections",
            "symmetric_differences",
            "voting_map",
            "m",
            "r",
            "rr",
            "c",
            "cf1",
            "cf2",
            "plots",
        ],
    )

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            collected = {var: None for var in var_names}

            def local_tracer(frame, event, arg):
                if event == "line":
                    for var in var_names:
                        if var in frame.f_locals:
                            collected[var] = frame.f_locals[var]
                return local_tracer

            sys.settrace(local_tracer)
            try:
                result = func(*args, **kwargs)
            finally:
                sys.settrace(None)
            # --- Header: Function Name ---
            debug_console.print(
                f"\n[bold magenta]Function: {func.__name__}[/bold magenta]\n"
            )
            if collected.get("plots") is not None:
                debug_console.print("[bold blue]Captured Plots:[/bold blue]")
                for plot_obj in collected["plots"]:
                    debug_console.print(collected.get("plots"))
            # --- Display Component Comparison (if available) ---

            if collected.get("c1") is not None and collected.get("c2") is not None:
                display_components_comparison(collected["c1"], collected["c2"])
            # --- Display Captured Plot Objects ---
            # --- Display Additional Debug Information ---
            s_edge = args[0] if args else None
            debug_console.print(
                f"\n{COLORS['highlight']}=== Processing s-edge {s_edge} ==={COLORS['reset']}"
            )
            for var in ["cf1", "cf2", "c", "c12"]:
                if collected.get(var) is not None:
                    _print_structured(var, collected[var])
            if all(
                v in collected
                for v in ["c1", "c2", "intersections", "symmetric_differences"]
            ):
                try:
                    display_truth_table(
                        collected["c1"],
                        collected["c2"],
                        collected["intersections"],
                        collected["symmetric_differences"],
                    )
                except Exception as e:
                    debug_console.print(
                        f"{COLORS['red']}Truth Table Error: {e}{COLORS['reset']}"
                    )
            if collected.get("voting_map"):
                debug_console.print(
                    f"{COLORS['var_name']}Voting Map Changes:{COLORS['reset']}"
                )
                counter = Counter(
                    _convert_to_hashable(comp) for comp in collected["voting_map"]
                )
                for label, color, filter_fn in [
                    ("+ Added", COLORS["green"], lambda c: c > 1),
                    ("- Removed", COLORS["red"], lambda c: c == 1),
                ]:
                    items = [comp for comp, cnt in counter.items() if filter_fn(cnt)]
                    if items:
                        debug_console.print(f"  {color}{label}:{COLORS['reset']}")
                        for comp in items[:3]:
                            debug_console.print(
                                f"    {_format_value(comp)} (count: {counter[comp]})"
                            )
                        if len(items) > 3:
                            debug_console.print(f"    ... {len(items)-3} more")
                debug_console.print("")
            if collected.get("rr"):
                debug_console.print(
                    f"{COLORS['var_name']}Final Jumping Taxa:{COLORS['reset']}"
                )
                for comp in collected["rr"]:
                    debug_console.print(f"  {_format_value(comp)}")
                debug_console.print("")
            if collected.get("m") is not None:
                _print_structured("m", collected["m"])
            if collected.get("r") is not None:
                debug_console.print(f"{COLORS['var_name']}r:{COLORS['reset']}")
                debug_console.print(collected["r"])
            if collected.get("rr"):
                rr = collected["rr"]
                debug_console.print(
                    f"{COLORS['var_name']}Final Result:{COLORS['reset']}"
                )
                debug_console.print(f"  Jumping Taxa: {_format_value(rr)}")
                debug_console.print("")
            return result

        return wrapper

    return decorator if not decorator_args else decorator(decorator_args[0])
