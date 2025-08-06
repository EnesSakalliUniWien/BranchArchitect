"""
ASCII Tree Printer for Node objects.

This module provides functions to print tree structures in ASCII format,
including side-by-side comparison of two trees. Inspired by various
open-source tree visualization libraries including asciitree, py-trees,
and directory tree generators.
"""

from typing import List, LiteralString, Optional, Any, Tuple


def render_tree_lines(node: Any, depth: int = 0) -> List[str]:
    """
    Recursively render a phylogenetic tree as ASCII art lines.
    Trees grow from top to bottom with horizontal branching.

    Args:
        node: Tree node with 'children' list and optional 'name' attribute
        depth: Current depth level for positioning

    Returns:
        List[str]: Lines representing the tree structure
    """
    children = getattr(node, "children", [])

    if not children:  # Leaf node
        if hasattr(node, "name") and node.name and str(node.name).strip():
            label = str(node.name)
        else:
            label = "●"
        return [" " * (depth * 4) + label]

    # Internal node - render children first
    child_results = []
    for child in children:
        child_results.append(render_tree_lines(child, depth + 1))

    # Calculate positions for branching
    lines: List[str] = []
    total_child_lines: int = sum(len(result) for result in child_results)

    if total_child_lines == 0:
        return [" " * (depth * 4) + "●"]

    # Add the current node at the top
    node_label = "●"
    if hasattr(node, "name") and node.name and str(node.name).strip():
        node_label = str(node.name)

    lines.append(" " * (depth * 4) + node_label)

    # Add vertical line down from parent
    if children:
        lines.append(" " * (depth * 4) + "│")

    # Add horizontal branching line
    if len(children) > 1:
        branch_line = " " * (depth * 4) + "├" + "─" * (len(children) * 2 - 1)
        lines.append(branch_line)

    # Add child subtrees
    current_line = 0
    for i, child_result in enumerate(child_results):
        if i > 0:
            # Add spacing between children
            lines.append("")

        # Add vertical connector for each child
        if len(children) > 1:
            connector_pos: int = depth * 4 + 1 + (i * 2)
            connector_line: LiteralString = " " * connector_pos + "│"
            lines.append(connector_line)

        # Add the child subtree
        for line in child_result:
            lines.append(line)

    return lines


def print_tree(node: Any) -> None:
    """
    Print a single tree in ASCII format.

    Args:
        node: The root node of the tree
    """
    lines = render_tree_lines(node)
    for line in lines:
        print(line)


def print_trees_side_by_side(
    tree1: Any,
    tree2: Any,
    labels: Optional[Tuple[str, str]] = None,
    separator: str = " │ ",
) -> None:
    """
    Print two trees side by side for comparison.

    Args:
        tree1: First tree root node
        tree2: Second tree root node
        labels: Optional tuple of labels for the trees
        separator: String to separate the two trees
    """
    # Render both trees as line lists
    lines1: List[str] = render_tree_lines(tree1)
    lines2: List[str] = render_tree_lines(tree2)

    # Calculate maximum width for proper alignment
    max_width1: int = max(len(line) for line in lines1) if lines1 else 0
    max_width2: int = max(len(line) for line in lines2) if lines2 else 0

    # Ensure both trees have the same number of lines
    max_lines = max(len(lines1), len(lines2))
    lines1.extend([" " * max_width1] * (max_lines - len(lines1)))
    lines2.extend([" " * max_width2] * (max_lines - len(lines2)))

    # Print headers if labels provided
    if labels:
        label1, label2 = labels
        header1 = f"Tree 1: {label1}".ljust(max_width1)
        header2 = f"Tree 2: {label2}"
        print(f"{header1}{separator}{header2}")
        print("─" * max_width1 + separator.replace(" ", "─") + "─" * max_width2)

    # Print trees side by side
    for line1, line2 in zip(lines1, lines2):
        padded_line1 = line1.ljust(max_width1)
        print(f"{padded_line1}{separator}{line2}")


def print_tree_comparison(
    tree1: Any,
    tree2: Any,
    labels: Optional[Tuple[str, str]] = None,
    style_name: str = "default",
) -> None:
    """
    High-level function to compare two trees with various style options.

    Args:
        tree1: First tree root node
        tree2: Second tree root node
        labels: Optional tuple of labels for the trees
        style_name: Style preset ("default", "box", "ascii")
    """
    print(f"\n{'=' * 60}")
    print(f"Tree Comparison ({style_name} style)")
    print(f"{'=' * 60}")

    print_trees_side_by_side(tree1, tree2, labels)
    print(f"{'=' * 60}\n")


def trees_to_string(tree1: Any, tree2: Any) -> str:
    """
    Return string representation of two trees side by side.

    Args:
        tree1: First tree root node
        tree2: Second tree root node

    Returns:
        String containing the side-by-side tree representation
    """
    lines1 = render_tree_lines(tree1)
    lines2 = render_tree_lines(tree2)

    max_width1 = max(len(line) for line in lines1) if lines1 else 0
    max_lines = max(len(lines1), len(lines2))

    lines1.extend([" " * max_width1] * (max_lines - len(lines1)))
    lines2.extend([""] * (max_lines - len(lines2)))

    separator = " │ "
    result_lines: List[str] = []
    for line1, line2 in zip(lines1, lines2):
        padded_line1 = line1.ljust(max_width1)
        result_lines.append(f"{padded_line1}{separator}{line2}")

    return "\n".join(result_lines)
