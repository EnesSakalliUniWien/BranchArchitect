"""Tree plotting interface using svg.py implementations."""

from brancharchitect.tree import Node
from typing import List, Optional
from brancharchitect.plot.svg import (
    generate_circular_two_trees_svg,
    generate_rectangular_tree_svg,
    generate_multiple_rectangular_trees_svg,
    generate_multiple_circular_trees_svg,
)


def plot_circular_tree_pair(
    tree1: Node,
    tree2: Node,
    width: int = 800,
    height: int = 400,
    margin: int = 30,
    label_offset: int = 2,
    ignore_branch_lengths: bool = True,
) -> str:
    """Display two circular trees side-by-side."""
    return generate_circular_two_trees_svg(
        tree1,
        tree2,
        size=width // 2,
        margin=margin,
        label_offset=label_offset,
        ignore_branch_lengths=ignore_branch_lengths,
    )


def plot_rectangular_tree_pair(
    tree1: Node,
    tree2: Node,
    width: int = 800,
    height: int = 400,
    margin: int = 30,
    label_offset: int = 2,
    ignore_branch_lengths: bool = True,
) -> str:
    """Display two rectangular trees side-by-side."""
    tree_width = (width - 3 * margin) // 2
    svg_template = """
    <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
        <g transform="translate({margin}, {margin})">
            {tree1_svg}
        </g>
        <g transform="translate({tree2_x}, {margin})">
            {tree2_svg}
        </g>
    </svg>
    """.strip()

    tree1_svg = generate_rectangular_tree_svg(
        tree1, width=tree_width, height=height - 2 * margin
    )
    tree2_svg = generate_rectangular_tree_svg(
        tree2, width=tree_width, height=height - 2 * margin
    )

    return svg_template.format(
        width=width,
        height=height,
        margin=margin,
        tree2_x=tree_width + 2 * margin,
        tree1_svg=tree1_svg.split("</svg>")[0].split(">", 1)[1],
        tree2_svg=tree2_svg.split("</svg>")[0].split(">", 1)[1],
    )


def plot_circular_trees_in_a_row(
    roots: List[Node],
    size: int = 200,
    margin: int = 30,
    label_offset: int = 2,
    ignore_branch_lengths: bool = False,
) -> str:
    """Display multiple circular trees in a row."""
    return generate_multiple_circular_trees_svg(
        roots=roots,
        size=size,
        margin=margin,
        label_offset=label_offset,
        ignore_branch_lengths=ignore_branch_lengths,
    )


def plot_rectangular_trees_in_a_row(
    roots: List[Node],
    size: int = 200,
    margin: int = 30,
    label_offset: int = 2,
    ignore_branch_lengths: bool = False,
) -> str:
    """Display multiple circular trees in a row."""
    return generate_multiple_rectangular_trees_svg(
        roots=roots,
        size=size,
        margin=margin,
        label_offset=label_offset,
        ignore_branch_lengths=ignore_branch_lengths,
    )


def plot_trees_side_by_side(
    tree1: Node, tree2: Node, width: int = 800, height: int = 400
) -> str:
    """Generate SVG visualization of two circular trees side by side."""
    return generate_circular_two_trees_svg(
        tree1, tree2, size=width // 2, margin=30, label_offset=2
    )


def plot_rectangular_tree(
    tree: Node, width: Optional[int] = 400, height: Optional[int] = 200
) -> str:
    """Generate rectangular layout visualization for a single tree."""
    return generate_rectangular_tree_svg(tree, width=width, height=height)
