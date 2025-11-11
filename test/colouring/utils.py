from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from brancharchitect.jumping_taxa import call_jumping_taxa
from brancharchitect.jumping_taxa.debug import jt_logger
from brancharchitect.jumping_taxa.debug.error_handling import (
    debug_algorithm_execution,
    log_detailed_error,
)
from brancharchitect.jumping_taxa.debug.output import write_debug_output
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree import Node


def discover_test_cases(base_dir: Path) -> List[Dict[str, Any]]:
    """Discover and load JSON test cases from subdirectories under base_dir."""
    test_data_list: List[Dict[str, Any]] = []
    if not base_dir.exists():
        return test_data_list

    for sub in base_dir.iterdir():
        if not sub.is_dir():
            continue
        for fp in sub.iterdir():
            if fp.is_file() and fp.suffix == ".json":
                with fp.open() as f:
                    data = json.load(f)
                data["name"] = fp.name
                data["source_path"] = str(fp.absolute())
                if "solutions" in data:
                    # Normalize solutions to sorted tuples of indices
                    data["solutions"] = [
                        sorted([tuple(sorted(component)) for component in solution])
                        for solution in (data["solutions"] or [[]])
                    ]
                test_data_list.append(data)
    return test_data_list


def ensure_output_dir(project_root: Path) -> Path:
    """Create and return the debug output directory under the project root."""
    out_dir = project_root / "output" / "test_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_trees_from_data(test_data: Dict[str, Any]) -> Tuple[Node, Node]:
    """Parse two trees from the provided test data, aligning order by the first tree."""
    t1 = parse_newick(test_data["tree1"])
    # Align second tree using the first tree's current order
    t2 = parse_newick(test_data["tree2"], list(t1.get_current_order()))
    return t1, t2


def translate_indices_to_names(
    solutions: Iterable[Iterable[int]], encoding: Dict[str, int]
) -> List[Tuple[str, ...]]:
    """Translate solution index tuples to taxon name tuples using the given encoding."""
    reverse = {v: k for k, v in encoding.items()}
    translated: List[Tuple[str, ...]] = []
    for sol in solutions:
        names = tuple(sorted(reverse[idx] for idx in sol))
        translated.append(names)
    return sorted(translated)


@debug_algorithm_execution
def process_and_log_test_case(
    t1: Node, t2: Node, test_name: str, output_dir: Path, source_path: str | None = None
) -> List[Tuple[int, ...]]:
    """Run the lattice algorithm, log visuals and diagnostics, and return solutions."""
    output_filename: str = f"debug_log_{os.path.splitext(test_name)[0]}.html"
    output_path = output_dir / output_filename

    try:
        jt_logger.disabled = False
        jt_logger.clear()
        jt_logger.section(f"Test Case: {test_name}")

        # Optional VS Code deep-linking button
        if source_path:
            vscode_uri = f"vscode://file/{source_path}"
            button_html = (
                f'<div style="margin: 10px 0;">'
                f'<a href="{vscode_uri}" style="display: inline-block; '
                f"background-color: #007acc; color: white; padding: 8px 16px; "
                f'text-decoration: none; border-radius: 4px; font-weight: bold;">'
                f"Open JSON Test File in VS Code</a>"
                f"</div>"
            )
            jt_logger.html(button_html)

        # Visualizations (Toytree PNG)
        jt_logger.log_tree_comparison(t1, t2, title="Tree Pair (Toytree PNG)")

        # Compute solutions via lattice algorithm
        return call_jumping_taxa(t1, t2, algorithm="lattice")

    except Exception as e:
        log_detailed_error(
            e,
            {
                "test_file": test_name,
                "tree1": t1,
                "tree2": t2,
                "timestamp": datetime.now().isoformat(),
            },
        )
        raise
    finally:
        write_debug_output(output_path=str(output_path), title=f"Test Analysis: {test_name}")
