"""Tree plotting interface using svg.py implementations."""

from brancharchitect.tree import Node
import xml.etree.ElementTree as ET
from typing import List, Optional
from brancharchitect.plot.circular_tree import (
    generate_circular_two_trees_svg,
    generate_multiple_circular_trees_svg,
)
from brancharchitect.plot.rectangular_tree import (
    generate_rectangular_tree_svg,
    generate_multiple_rectangular_trees_svg,
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
    bezier_colors: Optional[List[List[str]]] = None,
    bezier_stroke_widths: Optional[List[List[float]]] = None,
    highlight_branches: Optional[list] = None,
    highlight_width: float = 4.0,
    highlight_colors: Optional[list] = None,
    font_family: str = "Monospace",
    font_size: str = "12",
    stroke_color: str = "#000",
    leaf_font_size: Optional[str] = None,
    show_zero_length_indicators: bool = False,
    zero_length_indicator_color: str = "#ff4444",
    zero_length_indicator_size: float = 6.0,
) -> ET.Element:
    """Display multiple circular trees in a row. Returns SVG root element."""
    # Import here to avoid circular imports
    from brancharchitect.plot.circular_bezier_trees import TreeRenderingParams
    
    # Use unified parameter normalization
    params = TreeRenderingParams.normalize(
        len(roots),
        highlight_branches=highlight_branches,
        highlight_width=highlight_width,
        highlight_colors=highlight_colors
    )
    
    # Handle font size preference
    effective_font_size = leaf_font_size if leaf_font_size is not None else font_size
    
    svg_element, _ = generate_multiple_circular_trees_svg(
        roots=roots,
        size=size,
        margin=margin,
        label_offset=label_offset,
        ignore_branch_lengths=ignore_branch_lengths,
        bezier_colors=bezier_colors,
        bezier_stroke_widths=bezier_stroke_widths,
        highlight_branches=params.highlight_branches,
        highlight_width=params.highlight_width,
        highlight_colors=params.highlight_colors,
        font_family=font_family,
        font_size=effective_font_size,
        stroke_color=stroke_color,
        show_zero_length_indicators=show_zero_length_indicators,
        zero_length_indicator_color=zero_length_indicator_color,
        zero_length_indicator_size=zero_length_indicator_size,
    )
    return svg_element


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


# --- Utility functions for annotation ---
def add_svg_gridlines(
    svg_element,
    width,
    height,
    y_steps=7,
    color="#bdbdbd",
    stroke_width=1,
    dasharray="3,4",
    opacity="0.7",
):
    """
    Add horizontal gridlines to the SVG element for visual orientation.
    """
    grid_group = ET.Element("g", {"id": "background_gridlines"})
    for i in range(y_steps + 1):
        y = int(i * height / y_steps)
        line = ET.Element(
            "line",
            {
                "x1": "0",
                "y1": str(y),
                "x2": str(width),
                "y2": str(y),
                "stroke": color,
                "stroke-width": str(stroke_width),
                "stroke-dasharray": dasharray,
                "opacity": opacity,
            },
        )
        grid_group.append(line)
    svg_element.insert(1, grid_group)


def add_tree_labels(svg_element, num_trees, size, height, y_offset=2, font_size=14):
    label_group = ET.Element("g", {"id": "tree_labels"})
    for i in range(num_trees):
        x = int(i * size + size / 2)
        y = y_offset
        label = ET.Element(
            "text",
            {
                "x": str(x),
                "y": str(y),
                "text-anchor": "middle",
                "font-size": str(font_size),
                "font-family": "sans-serif",
                "fill": "#333",
                "font-weight": "bold",
            },
        )
        label.text = f"Tree {i + 1}"
        label_group.append(label)
    svg_element.append(label_group)


def add_direction_arrow(svg_element, width, y=16):
    arrow_group = ET.Element("g", {"id": "direction_arrow"})
    line = ET.Element(
        "line",
        {
            "x1": "20",
            "y1": str(y),
            "x2": str(width - 20),
            "y2": str(y),
            "stroke": "#333",
            "stroke-width": "2",
            "marker-end": "url(#arrowhead)",
        },
    )
    arrow_group.append(line)
    defs = svg_element.find("defs")
    if defs is None:
        defs = ET.Element("defs")
        svg_element.insert(0, defs)
    marker = ET.Element(
        "marker",
        {
            "id": "arrowhead",
            "markerWidth": "10",
            "markerHeight": "7",
            "refX": "10",
            "refY": "3.5",
            "orient": "auto",
            "markerUnits": "strokeWidth",
        },
    )
    polygon = ET.Element("polygon", {"points": "0 0, 10 3.5, 0 7", "fill": "#333"})
    marker.append(polygon)
    defs.append(marker)
    svg_element.append(arrow_group)
