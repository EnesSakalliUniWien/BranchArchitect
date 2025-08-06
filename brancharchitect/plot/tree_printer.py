"""
ASCII Tree Printer for Node objects.

This module provides functions to print tree structures in ASCII format,
including side-by-side comparison of two trees. Inspired by various
open-source tree visualization libraries including asciitree, py-trees,
and directory tree generators.
"""

from typing import List, Optional, Any, Tuple


def render_tree_lines(node: Any, prefix: str = "", is_last: bool = True) -> List[str]:
    """
    Recursively render a tree as ASCII art lines.

    Args:
        node: Tree node with 'children' list and optional 'name' attribute
        prefix: Current line prefix for indentation
        is_last: Whether this is the last child at current level

    Returns:
        List[str]: Lines representing the tree structure
    """
    lines: List[str] = []

    # Get node label
    if hasattr(node, "name") and node.name and str(node.name).strip():
        label = str(node.name)
    elif not hasattr(node, "children") or not node.children:  # Leaf node
        label = "leaf"
    else:  # Internal node without name
        label = "●"  # Simple dot for internal nodes

    # Add current node with proper tree structure
    if prefix == "":  # Root node
        lines.append(label)  # Always show root node
    else:
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{label}")

    # Add children
    children = getattr(node, "children", [])
    if children:
        # Determine child prefix based on whether this is the last child
        child_prefix: str = prefix + ("    " if is_last else "│   ")

        for i, child in enumerate(children):
            is_last_child: bool = i == len(children) - 1
            lines.extend(render_tree_lines(child, child_prefix, is_last_child))

    return lines


def print_tree(node: Any) -> None:
    """
    Print a single tree in ASCII format.

    Args:
        node: The root node of the tree
    """
    lines: List[str] = render_tree_lines(node)
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
    max_lines: int = max(len(lines1), len(lines2))
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
        padded_line1: str = line1.ljust(max_width1)
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
    lines1: List[str] = render_tree_lines(tree1)
    lines2: List[str] = render_tree_lines(tree2)

    max_width1: int = max(len(line) for line in lines1) if lines1 else 0
    max_lines: int = max(len(lines1), len(lines2))

    lines1.extend([" " * max_width1] * (max_lines - len(lines1)))
    lines2.extend([""] * (max_lines - len(lines2)))

    separator = " │ "
    result_lines: List[str] = []
    for line1, line2 in zip(lines1, lines2):
        padded_line1 = line1.ljust(max_width1)
        result_lines.append(f"{padded_line1}{separator}{line2}")

    return "\n".join(result_lines)
