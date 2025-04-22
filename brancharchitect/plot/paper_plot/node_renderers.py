"""
Node rendering utilities for BranchArchitect.

This module contains functions for rendering tree nodes and branches in SVG format.
The code has been modularized for better maintainability and clarity.
"""

import xml.etree.ElementTree as ET
from typing import Dict, Set, Optional, Tuple, List

from brancharchitect.tree import Node
from brancharchitect.plot.tree_utils import get_node_label
from brancharchitect.plot.paper_plot.effects import (
    render_waterbrush_edge,
    render_blob_edge,
)
from brancharchitect.plot.paper_plot.paper_plot_constants import FONT_SANS_SERIF


# --- INTERNAL NODE RENDERING FUNCTIONS ---


def render_internal_node(
    svg_group: ET.Element,
    node: Node,
    coords: Dict,
    node_coords: Dict[Node, Dict],
    colors: Dict,
    highlight_edges: Set,
    highlight_color: str,
    highlight_stroke_width: str,
    split_depth: float,
    fade_opacity: float,
    use_dash: bool,
    use_waterbrush: bool = False,
    use_blobs: bool = False,
    branch_roundness: float = 0,
    use_elevation: bool = False,
    blob_fill: str = "",
    blob_opacity: str = "0.35",
    cut_edges: Optional[Set[Tuple[str, str]]] = None,
) -> None:
    """
    Render an internal node and its branches following Material Design 3 guidelines.

    Optimized for print publication with clear visual hierarchy and appropriate styling.

    Args:
        svg_group: SVG group element to render to
        node: The node to render
        coords: Coordinates data for this node
        node_coords: All node coordinates
        colors: Color palette
        highlight_edges: Set of highlighted edge IDs
        highlight_color: Color to use for highlighted elements
        highlight_stroke_width: Stroke width for highlighted elements
        split_depth: Depth at which to split the tree
        fade_opacity: Opacity to use for faded elements
        use_dash: Whether to use dashed lines for split elements
        use_waterbrush: Whether to use waterbrush effect for highlighted edges
        use_blobs: Whether to use blob effect for highlighted edges
        branch_roundness: Roundness of branch curves (0-10)
        use_elevation: Whether to use elevation effects
        blob_fill: Fill color for blob effects
        blob_opacity: Opacity for blob effects
        cut_edges: Set of (parent, child) tuples specifying edges to cut
    """
    node_id = coords["id"]

    # Create a group for this node and its branches
    node_group = create_node_group(svg_group, node_id)

    # Render branches to each child node
    render_node_branches(
        node_group,
        coords,
        node_coords,
        colors,
        highlight_edges,
        highlight_color,
        highlight_stroke_width,
        split_depth,
        fade_opacity,
        use_dash,
        use_waterbrush,
        use_blobs,
        branch_roundness,
        use_elevation,
        blob_fill,
        blob_opacity,
        cut_edges,
    )


def create_node_group(svg_group: ET.Element, node_id: str) -> ET.Element:
    """
    Create an SVG group element for a node.

    Args:
        svg_group: Parent SVG group element
        node_id: Identifier for the node

    Returns:
        The created SVG group element
    """
    return ET.SubElement(
        svg_group, "g", {"class": "md3-internal-node", "data-node-id": node_id}
    )


def render_node_branches(
    node_group: ET.Element,
    coords: Dict,
    node_coords: Dict[Node, Dict],
    colors: Dict,
    highlight_edges: Set,
    highlight_color: str,
    highlight_stroke_width: str,
    split_depth: float,
    fade_opacity: float,
    use_dash: bool,
    use_waterbrush: bool,
    use_blobs: bool,
    branch_roundness: float,
    use_elevation: bool,
    blob_fill: str,
    blob_opacity: str,
    cut_edges: Optional[Set[Tuple[str, str]]],
) -> None:
    """
    Render all branches from a node to its child nodes.

    Args:
        node_group: SVG group for the node
        coords: Coordinates data for this node
        node_coords: All node coordinates
        colors: Color palette
        highlight_edges: Set of highlighted edge IDs
        highlight_color: Color to use for highlighted elements
        highlight_stroke_width: Stroke width for highlighted elements
        split_depth: Depth at which to split the tree
        fade_opacity: Opacity to use for faded elements
        use_dash: Whether to use dashed lines for split elements
        use_waterbrush: Whether to use waterbrush effect for highlighted edges
        use_blobs: Whether to use blob effect for highlighted edges
        branch_roundness: Roundness of branch curves (0-10)
        use_elevation: Whether to use elevation effects
        blob_fill: Fill color for blob effects
        blob_opacity: Opacity for blob effects
        cut_edges: Set of (parent, child) tuples specifying edges to cut
    """
    x, y = coords["x"], coords["y"]
    node_id = coords["id"]

    # For each child, draw the branch
    for child_node in coords.get("child_nodes", []):
        child_coords = node_coords.get(child_node)
        if not child_coords:
            continue

        child_x = child_coords["x"]
        child_y = child_coords["y"]
        child_id = child_coords["id"]
        child_depth = child_coords["depth"]

        # Create branch group
        branch_group = create_branch_group(node_group, node_id, child_id)

        # Determine branch state
        branch_state = determine_branch_state(
            node_id, child_id, child_depth, split_depth, highlight_edges, cut_edges
        )

        # Render the branch with appropriate styling based on state
        render_branch(
            branch_group,
            x,
            y,
            child_x,
            child_y,
            branch_state,
            colors,
            highlight_color,
            highlight_stroke_width,
            fade_opacity,
            use_dash,
            use_waterbrush,
            use_blobs,
            branch_roundness,
            use_elevation,
            blob_fill,
            blob_opacity,
        )


def create_branch_group(
    parent_group: ET.Element, parent_id: str, child_id: str
) -> ET.Element:
    """
    Create an SVG group for a branch connecting two nodes.

    Args:
        parent_group: SVG group of the parent node
        parent_id: ID of the parent node
        child_id: ID of the child node

    Returns:
        The created SVG group for the branch
    """
    return ET.SubElement(
        parent_group,
        "g",
        {"class": "md3-branch", "data-edge-id": f"{parent_id}-{child_id}"},
    )


def determine_branch_state(
    parent_id: str,
    child_id: str,
    child_depth: float,
    split_depth: float,
    highlight_edges: Set,
    cut_edges: Optional[Set[Tuple[str, str]]],
) -> Dict:
    """
    Determine the state of a branch based on its properties.

    Args:
        parent_id: ID of the parent node
        child_id: ID of the child node
        child_depth: Depth of the child node
        split_depth: Depth at which to split the tree
        highlight_edges: Set of highlighted edge IDs
        cut_edges: Set of edges to cut

    Returns:
        A dictionary describing the state of the branch
    """
    edge_id = (parent_id, child_id)
    return {
        "is_highlighted": edge_id in highlight_edges,
        "is_cut": cut_edges and edge_id in cut_edges,
        "is_faded": child_depth >= split_depth,
    }


def render_branch(
    branch_group: ET.Element,
    x: float,
    y: float,
    child_x: float,
    child_y: float,
    branch_state: Dict,
    colors: Dict,
    highlight_color: str,
    highlight_stroke_width: str,
    fade_opacity: float,
    use_dash: bool,
    use_waterbrush: bool,
    use_blobs: bool,
    branch_roundness: float,
    use_elevation: bool,
    blob_fill: str,
    blob_opacity: str,
) -> None:
    """
    Render a branch between nodes with appropriate styling.

    Args:
        branch_group: SVG group for the branch
        x, y: Coordinates of the parent node
        child_x, child_y: Coordinates of the child node
        branch_state: State dictionary for the branch
        colors: Color palette
        highlight_color: Color for highlighted elements
        highlight_stroke_width: Stroke width for highlighted elements
        fade_opacity: Opacity for faded elements
        use_dash: Whether to use dashed lines
        use_waterbrush: Whether to use waterbrush effect
        use_blobs: Whether to use blob effect
        branch_roundness: Roundness of branch curves
        use_elevation: Whether to use elevation effects
        blob_fill: Fill color for blob effects
        blob_opacity: Opacity for blob effects
    """
    is_highlighted = branch_state["is_highlighted"]
    is_cut = branch_state["is_cut"]
    is_faded = branch_state["is_faded"]

    # Calculate opacity based on state
    opacity = str(fade_opacity) if is_faded else "1.0"

    # Select color based on state
    color = highlight_color if is_highlighted else colors["base_stroke"]

    # Use appropriate visual treatment based on state and settings
    if is_highlighted and use_blobs:
        render_blob_branch(
            branch_group, x, y, child_x, child_y, blob_fill, blob_opacity
        )
    elif is_highlighted and use_waterbrush:
        render_waterbrush_branch(
            branch_group, x, y, child_x, child_y, color, highlight_stroke_width, opacity
        )
    else:
        # Standard line with MD3 styling
        render_standard_branch(
            branch_group,
            x,
            y,
            child_x,
            child_y,
            is_highlighted,
            is_faded,
            is_cut,
            color,
            colors,
            highlight_stroke_width,
            opacity,
            use_dash,
            branch_roundness,
            use_elevation,
            use_waterbrush,
            use_blobs,
        )


def render_blob_branch(
    branch_group: ET.Element,
    x: float,
    y: float,
    child_x: float,
    child_y: float,
    blob_fill: str,
    blob_opacity: str,
) -> None:
    """
    Render a branch with blob effect.

    Args:
        branch_group: SVG group for the branch
        x, y: Coordinates of the parent node
        child_x, child_y: Coordinates of the child node
        blob_fill: Fill color for blob effects
        blob_opacity: Opacity for blob effects
    """
    # Convert blob_opacity to float if it's a string
    float_blob_opacity = (
        float(blob_opacity) if isinstance(blob_opacity, str) else blob_opacity
    )
    render_blob_edge(
        branch_group, x, y, child_x, child_y, blob_fill, str(float_blob_opacity)
    )


def render_waterbrush_branch(
    branch_group: ET.Element,
    x: float,
    y: float,
    child_x: float,
    child_y: float,
    color: str,
    stroke_width: str,
    opacity: str,
) -> None:
    """
    Render a branch with waterbrush effect.

    Args:
        branch_group: SVG group for the branch
        x, y: Coordinates of the parent node
        child_x, child_y: Coordinates of the child node
        color: Color for the branch
        stroke_width: Width of the stroke
        opacity: Opacity value
    """
    # For print, use a more defined waterbrush with better contrast
    render_waterbrush_edge(
        branch_group,
        x,
        y,
        child_x,
        child_y,
        color,
        stroke_width,
        min(float(opacity) * 1.2, 1.0),  # Increase opacity slightly for print
    )


def render_standard_branch(
    branch_group: ET.Element,
    x: float,
    y: float,
    child_x: float,
    child_y: float,
    is_highlighted: bool,
    is_faded: bool,
    is_cut: bool,
    color: str,
    colors: Dict,
    highlight_stroke_width: str,
    opacity: str,
    use_dash: bool,
    branch_roundness: float,
    use_elevation: bool,
    use_waterbrush: bool,
    use_blobs: bool,
) -> None:
    """
    Render a standard branch with Material Design 3 styling.

    Args:
        branch_group: SVG group for the branch
        x, y: Coordinates of the parent node
        child_x, child_y: Coordinates of the child node
        is_highlighted: Whether the branch is highlighted
        is_faded: Whether the branch is faded
        is_cut: Whether the branch is cut
        color: Color for the branch
        colors: Color palette
        highlight_stroke_width: Stroke width for highlighted elements
        opacity: Opacity value
        use_dash: Whether to use dashed lines
        branch_roundness: Roundness of branch curves
        use_elevation: Whether to use elevation effects
        use_waterbrush: Whether waterbrush effect is enabled
        use_blobs: Whether blob effect is enabled
    """
    # Calculate dash pattern
    dash_pattern = calculate_dash_pattern(is_faded, use_dash, branch_roundness)

    # Calculate bezier curve control points
    control_points = calculate_bezier_control_points(
        x, y, child_x, child_y, branch_roundness
    )

    # Create path data
    path_data = create_path_data(x, y, control_points, child_x, child_y)

    # Create line style
    line_style = create_line_style(
        color,
        highlight_stroke_width if is_highlighted else colors["default_stroke_width"],
        opacity,
        dash_pattern,
    )

    # Apply cut edge styling if needed
    if is_cut:
        apply_cut_edge_styling(branch_group, line_style, x, y, child_x, child_y, colors)

    # Apply elevation effect if needed
    if use_elevation and is_highlighted and not is_cut:
        apply_elevation_effect(
            branch_group, path_data, line_style, highlight_stroke_width, opacity
        )

    # Add the main branch path
    ET.SubElement(branch_group, "path", {**{"d": path_data}, **line_style})

    # Add pattern overlay for highlighted branches
    if is_highlighted and not use_waterbrush and not use_blobs:
        add_highlight_pattern(branch_group, path_data, highlight_stroke_width)


def calculate_dash_pattern(
    is_faded: bool, use_dash: bool, branch_roundness: float
) -> str:
    """
    Calculate the dash pattern for a branch.

    Args:
        is_faded: Whether the branch is faded
        use_dash: Whether to use dashed lines
        branch_roundness: Roundness of branch curves

    Returns:
        Dash pattern string
    """
    dash_pattern = "none"
    if is_faded and use_dash:
        # More print-friendly dash pattern
        dash_pattern = "3,3" if branch_roundness > 0 else "4,4"
    return dash_pattern


def calculate_bezier_control_points(
    x: float, y: float, child_x: float, child_y: float, branch_roundness: float
) -> Dict:
    """
    Calculate bezier curve control points for a branch.

    Args:
        x, y: Coordinates of the parent node
        child_x, child_y: Coordinates of the child node
        branch_roundness: Roundness of branch curves

    Returns:
        Dictionary with control point coordinates
    """
    dx = child_x - x
    dy = child_y - y

    # Calculate curve tension based on distance and roundness
    base_curve_tension = 0.4  # Higher values = more curved
    curve_tension = base_curve_tension

    if branch_roundness > 0:
        # Adjust tension based on branch_roundness (higher = more curved)
        curve_tension = min(base_curve_tension + (branch_roundness / 20), 0.8)

    # Calculate bezier curve control points
    # First control point is vertically down from parent
    cx1 = x
    cy1 = y + (dy * curve_tension)

    # Second control point is horizontally aligned with child
    cx2 = child_x
    cy2 = y + (dy * (1 - curve_tension))

    # Special case for horizontal alignment to avoid weird curves
    if abs(dx) < 10:
        # If nodes are nearly aligned vertically, use a simpler curve
        cx1 = x
        cy1 = y + (dy * 0.25)
        cx2 = x
        cy2 = y + (dy * 0.75)

    return {"cx1": cx1, "cy1": cy1, "cx2": cx2, "cy2": cy2}


def create_path_data(
    x: float, y: float, control_points: Dict, child_x: float, child_y: float
) -> str:
    """
    Create the SVG path data string for a bezier curve.

    Args:
        x, y: Coordinates of the parent node
        control_points: Bezier curve control points
        child_x, child_y: Coordinates of the child node

    Returns:
        SVG path data string
    """
    cx1, cy1 = control_points["cx1"], control_points["cy1"]
    cx2, cy2 = control_points["cx2"], control_points["cy2"]

    return (
        f"M {x:.2f},{y:.2f} "  # Start at parent node
        f"C {cx1:.2f},{cy1:.2f} {cx2:.2f},{cy2:.2f} {child_x:.2f},{child_y:.2f}"  # Cubic bezier to child
    )


def create_line_style(
    color: str, stroke_width: str, opacity: str, dash_pattern: str
) -> Dict:
    """
    Create the style attributes for a line.

    Args:
        color: Line color
        stroke_width: Line width
        opacity: Line opacity
        dash_pattern: Line dash pattern

    Returns:
        Dictionary of style attributes
    """
    return {
        "stroke": color,
        "stroke-width": stroke_width,
        "stroke-opacity": opacity,
        "stroke-dasharray": dash_pattern,
        "fill": "none",
        "stroke-linecap": "round",  # MD3 uses rounded caps for better print quality
        "stroke-linejoin": "round",  # MD3 uses rounded joins for better print quality
    }


def apply_cut_edge_styling(
    branch_group: ET.Element,
    line_style: Dict,
    x: float,
    y: float,
    child_x: float,
    child_y: float,
    colors: Dict,
) -> None:
    """
    Apply styling for cut edges.

    Args:
        branch_group: SVG group for the branch
        line_style: Style dictionary for the line
        x, y: Coordinates of the parent node
        child_x, child_y: Coordinates of the child node
        colors: Color palette
    """
    # Use dashed pattern for cut edges
    line_style["stroke-dasharray"] = "5,3"  # Dashed line pattern

    # Use specified cut edge color if available, otherwise use error/warning color
    cut_edge_color = colors.get(
        "cut_edge_color", "#B4261E"
    )  # Default to MD3 Error 40 (red)
    line_style["stroke"] = cut_edge_color

    # Calculate position for the tangential cut line (at the midpoint of the branch)
    mid_x = (x + child_x) / 2
    mid_y = (y + child_y) / 2
    
    # Calculate the direction vector of the branch
    dx = child_x - x
    dy = child_y - y
    
    # Calculate perpendicular (tangential) direction for the cut line
    # Normalize the vector and rotate 90 degrees
    length = (dx**2 + dy**2)**0.5
    if length > 0:
        perp_dx = -dy / length
        perp_dy = dx / length
        
        # Calculate cut line length (proportional to branch thickness)
        cut_length = float(line_style.get("stroke-width", "2")) * 3
        
        # Calculate endpoints of the cut line
        cut_start_x = mid_x + perp_dx * cut_length
        cut_start_y = mid_y + perp_dy * cut_length
        cut_end_x = mid_x - perp_dx * cut_length
        cut_end_y = mid_y - perp_dy * cut_length
        
        # Create cut line style (make it stand out)
        cut_style = {
            "stroke": colors.get("cut_line_color", "#000000"),
            "stroke-width": str(float(line_style.get("stroke-width", "2")) * 1.5),
            "stroke-linecap": "round"
        }
        
        # Add the tangential cut line
        ET.SubElement(branch_group, "line", {
            "x1": str(cut_start_x),
            "y1": str(cut_start_y),
            "x2": str(cut_end_x),
            "y2": str(cut_end_y),
            **cut_style
        })
    
    # Add scissors marker if available in the color palette
    if colors.get("use_cut_markers", False):
        add_cut_marker(branch_group, x, y, child_x, child_y, cut_edge_color, line_style)


def add_cut_marker(
    branch_group: ET.Element,
    x: float,
    y: float,
    child_x: float,
    child_y: float,
    cut_edge_color: str,
    line_style: Dict,
) -> None:
    """
    Add a cut marker (scissors icon) to a branch.

    Args:
        branch_group: SVG group for the branch
        x, y: Coordinates of the parent node
        child_x, child_y: Coordinates of the child node
        cut_edge_color: Color for the cut marker
        line_style: Style dictionary for the line
    """
    # Find midpoint of the path for marker placement
    mid_x = (x + child_x) / 2
    mid_y = (y + child_y) / 2

    # Add a small scissors icon or marker at the midpoint
    scissors_group = ET.SubElement(
        branch_group,
        "g",
        {"class": "cut-marker", "transform": f"translate({mid_x},{mid_y})"},
    )

    # Simple scissors symbol (can be replaced with more complex path)
    scissors_size = float(line_style["stroke-width"]) * 2.5

    # Create marker circle
    ET.SubElement(
        scissors_group,
        "circle",
        {
            "r": str(scissors_size),
            "fill": "#FFFFFF",
            "stroke": cut_edge_color,
            "stroke-width": "1",
        },
    )

    # Add an X inside the circle to represent cut
    line_size = scissors_size * 0.7
    ET.SubElement(
        scissors_group,
        "line",
        {
            "x1": str(-line_size),
            "y1": str(-line_size),
            "x2": str(line_size),
            "y2": str(line_size),
            "stroke": cut_edge_color,
            "stroke-width": "1.2",
            "stroke-linecap": "round",
        },
    )
    ET.SubElement(
        scissors_group,
        "line",
        {
            "x1": str(line_size),
            "y1": str(-line_size),
            "x2": str(-line_size),
            "y2": str(line_size),
            "stroke": cut_edge_color,
            "stroke-width": "1.2",
            "stroke-linecap": "round",
        },
    )


def apply_elevation_effect(
    branch_group: ET.Element,
    path_data: str,
    line_style: Dict,
    highlight_stroke_width: str,
    opacity: str,
) -> None:
    """
    Apply elevation effect to a branch.

    Args:
        branch_group: SVG group for the branch
        path_data: SVG path data string
        line_style: Style dictionary for the line
        highlight_stroke_width: Stroke width for highlighted elements
        opacity: Opacity value
    """
    # Add secondary decorative element for highlighted branches (MD3 style)
    # First draw a wider stroke with lower opacity as a "glow" effect
    glow_style = line_style.copy()
    glow_style["stroke-width"] = str(float(highlight_stroke_width) * 2.5)
    glow_style["stroke-opacity"] = str(float(opacity) * 0.15)
    ET.SubElement(branch_group, "path", {**{"d": path_data}, **glow_style})

    # Add subtle drop shadow for primary branch
    if float(highlight_stroke_width) > 1.5:
        line_style["filter"] = "url(#branchShadow)"


def add_highlight_pattern(
    branch_group: ET.Element, path_data: str, highlight_stroke_width: str
) -> None:
    """
    Add a pattern overlay to a highlighted branch.

    Args:
        branch_group: SVG group for the branch
        path_data: SVG path data string
        highlight_stroke_width: Stroke width for highlighted elements
    """
    pattern_style = {
        "stroke": "url(#highlightPattern)",
        "stroke-width": str(float(highlight_stroke_width) * 1.5),
        "stroke-opacity": "0.6",
        "fill": "none",
    }
    ET.SubElement(branch_group, "path", {**{"d": path_data}, **pattern_style})


# --- LEAF NODE RENDERING FUNCTIONS ---


def render_leaf_node(
    svg_group: ET.Element,
    node: Node,
    coords: Dict,
    label_y: float,
    colors: Dict,
    highlight_leaves: Set,
    highlight_color: str,
    highlight_stroke_width: str,
    split_depth: float,
    fade_opacity: float,
    use_dash: bool,
    node_marker_size: float = 0,
    leaf_font_size: Optional[float] = None,
    leaf_label_offset: float = 0.0,
) -> None:
    """
    Render a leaf node with its label according to Material Design 3 guidelines.

    Optimized for print publication with clear typography and appropriate contrast.

    Args:
        svg_group: SVG group element to render to
        node: The node to render
        coords: Coordinates data for this node
        label_y: Y-coordinate for the label
        colors: Color palette
        highlight_leaves: Set of highlighted leaf node IDs
        highlight_color: Color to use for highlighted elements
        highlight_stroke_width: Stroke width for highlighted elements
        split_depth: Depth at which to split the tree
        fade_opacity: Opacity to use for faded elements
        use_dash: Whether to use dashed lines for split elements
        node_marker_size: Size of node markers
        leaf_font_size: Font size for leaf labels
        leaf_label_offset: Vertical offset for leaf labels
    """
    x, y = coords["x"], coords["y"]
    node_id = coords["id"]
    depth = coords["depth"]

    # Create leaf node group
    leaf_group = create_leaf_group(svg_group, node_id)

    # Determine leaf state
    leaf_state = determine_leaf_state(node_id, depth, split_depth, highlight_leaves)

    # Get node label
    label_text = get_node_label(node)

    # Render connector line if needed
    render_leaf_connector_line(
        leaf_group,
        x,
        y,
        label_y,
        leaf_state,
        colors,
        highlight_color,
        highlight_stroke_width,
        use_dash,
    )

    # Render text label if present
    if label_text:
        render_leaf_label_text(
            leaf_group,
            x,
            label_y + leaf_label_offset,
            label_text,
            leaf_state,
            colors,
            highlight_color,
            leaf_font_size,
        )


def create_leaf_group(svg_group: ET.Element, node_id: str) -> ET.Element:
    """
    Create an SVG group for a leaf node.

    Args:
        svg_group: Parent SVG group
        node_id: ID of the leaf node

    Returns:
        The created SVG group
    """
    return ET.SubElement(
        svg_group, "g", {"class": "md3-leaf-node", "data-node-id": node_id}
    )


def determine_leaf_state(
    node_id: str,
    depth: float,
    split_depth: float,
    highlight_leaves: Set,
    fade_opacity: float = 0.5,
) -> Dict:
    """
    Determine the state of a leaf node.

    Args:
        node_id: ID of the leaf node
        depth: Depth of the leaf node
        split_depth: Depth at which to split the tree
        highlight_leaves: Set of highlighted leaf node IDs

    Returns:
        A dictionary describing the state of the leaf node
    """
    return {
        "is_highlighted": node_id in highlight_leaves,
        "is_faded": depth >= split_depth,
        "opacity": str(fade_opacity) if depth >= split_depth else "1.0",
    }


def render_leaf_connector_line(
    leaf_group: ET.Element,
    x: float,
    y: float,
    label_y: float,
    leaf_state: Dict,
    colors: Dict,
    highlight_color: str,
    highlight_stroke_width: str,
    use_dash: bool,
) -> None:
    """
    Render a connector line from a leaf node to its label.

    Args:
        leaf_group: SVG group for the leaf node
        x, y: Coordinates of the leaf node
        label_y: Y-coordinate for the label
        leaf_state: State dictionary for the leaf node
        colors: Color palette
        highlight_color: Color for highlighted elements
        highlight_stroke_width: Stroke width for highlighted elements
        use_dash: Whether to use dashed lines
    """
    is_highlighted = leaf_state["is_highlighted"]
    is_faded = leaf_state["is_faded"]
    opacity = leaf_state["opacity"]

    # Material Design 3 uses more subtle dash patterns for secondary elements
    dash = "3,3" if (is_faded and use_dash) else "none"

    # Material Design 3 uses state-based styling with coherent sizing
    connector_style = {
        "stroke": highlight_color if is_highlighted else colors["base_stroke"],
        "stroke-width": highlight_stroke_width
        if is_highlighted
        else colors["default_stroke_width"],
        "stroke-opacity": opacity,
        "stroke-dasharray": dash,
        "stroke-linecap": "round",  # MD3 uses rounded caps for better print quality
        "fill": "none",
    }

    # For print, we need to ensure the line is visible but not overwhelming
    path_data = f"M {x:.2f},{y:.2f} V {label_y - 10:.2f}"
    ET.SubElement(leaf_group, "path", {**{"d": path_data}, **connector_style})


def render_leaf_label_text(
    leaf_group: ET.Element,
    x: float,
    label_y: float,
    label_text: str,
    leaf_state: Dict,
    colors: Dict,
    highlight_color: str,
    leaf_font_size: Optional[float],
) -> None:
    """
    Render the text label for a leaf node.

    Args:
        leaf_group: SVG group for the leaf node
        x: X-coordinate for the label
        label_y: Y-coordinate for the label
        label_text: Text of the label
        leaf_state: State dictionary for the leaf node
        colors: Color palette
        highlight_color: Color for highlighted elements
        leaf_font_size: Font size for leaf labels
    """
    is_highlighted = leaf_state["is_highlighted"]
    opacity = leaf_state["opacity"]

    # Calculate font size
    font_size = calculate_font_size(leaf_font_size, colors)

    # For print, we use weights that reproduce well
    font_weight = "500" if is_highlighted else "400"

    # Material Design's type scale with appropriate contrast
    text_style = {
        "font-family": colors.get("font_family", FONT_SANS_SERIF),
        "font-size": str(font_size),
        "font-weight": font_weight,
        "text-anchor": "middle",
        "dominant-baseline": "central",  # Better vertical alignment
        "fill": highlight_color if is_highlighted else colors["base_text"],
        "opacity": opacity,
    }

    # Add text element
    text_el = ET.SubElement(
        leaf_group, "text", {**{"x": str(x), "y": str(label_y)}, **text_style}
    )
    text_el.text = label_text


def calculate_font_size(leaf_font_size: Optional[float], colors: Dict) -> float:
    """
    Calculate the font size for a leaf label.

    Args:
        leaf_font_size: Custom font size if provided
        colors: Color palette

    Returns:
        Calculated font size
    """
    if leaf_font_size is not None:
        # Use the custom leaf_font_size if provided
        return max(8.0, min(24.0, float(leaf_font_size)))
    else:
        # MD3 typography scale for labels in print context
        try:
            base_font_size = float(colors["font_size"])
            # For print, ensure font size is reasonable
            return max(8.0, min(14.0, base_font_size))
        except (ValueError, KeyError):
            return 10.0  # Fallback size for print


# --- NODE MARKER RENDERING FUNCTIONS ---


def render_node_markers(
    svg_group: ET.Element,
    node: Node,
    coords: Dict,
    node_coords: Dict[Node, Dict],
    colors: Dict,
    highlight_edges: Set,
    highlight_color: str,
    is_leaf: bool = False,
    filter_id: Optional[str] = None,
) -> None:
    """
    Render Material Design 3 style node markers optimized for print publication.

    Args:
        svg_group: SVG group element to render marker for
        node: The node to render marker for
        coords: Coordinates data for this node
        node_coords: All node coordinates
        colors: Color palette
        highlight_edges: Set of highlighted edge IDs
        highlight_color: Color to use for highlighted elements
        is_leaf: Whether this is a leaf node
        filter_id: Optional ID of a filter to apply to the marker (e.g., "url(#nodeGlow)")
    """
    # Get node marker size from colors or use default
    node_marker_size = get_node_marker_size(colors)

    # Skip if no markers needed
    if node_marker_size <= 0:
        return

    # Create marker group
    marker_group = create_marker_group(svg_group, coords["id"])

    # Determine if node is highlighted
    is_highlighted = is_node_highlighted(coords, node_coords, highlight_edges, is_leaf)

    # Render appropriate marker based on node type
    if is_leaf:
        render_leaf_marker(
            marker_group,
            coords["x"],
            coords["y"],
            is_highlighted,
            highlight_color,
            colors,
            node_marker_size,
            filter_id,
        )
    else:
        render_internal_marker(
            marker_group,
            coords["x"],
            coords["y"],
            is_highlighted,
            highlight_color,
            colors,
            node_marker_size,
            filter_id,
        )


def get_node_marker_size(colors: Dict) -> float:
    """
    Get the node marker size from colors or use default.

    Args:
        colors: Color palette

    Returns:
        Node marker size
    """
    # Increased default size from 3.0 to 4.5 for better visibility in print
    return float(colors.get("node_marker_size", 4.5))


def create_marker_group(svg_group: ET.Element, node_id: str) -> ET.Element:
    """
    Create an SVG group for a node marker.

    Args:
        svg_group: Parent SVG group
        node_id: ID of the node

    Returns:
        The created SVG group
    """
    return ET.SubElement(
        svg_group, "g", {"class": "md3-node-marker", "data-node-id": node_id}
    )


def is_node_highlighted(
    coords: Dict, node_coords: Dict[Node, Dict], highlight_edges: Set, is_leaf: bool
) -> bool:
    """
    Determine if a node should be highlighted.

    Args:
        coords: Coordinates data for this node
        node_coords: All node coordinates
        highlight_edges: Set of highlighted edge IDs
        is_leaf: Whether this is a leaf node

    Returns:
        True if the node should be highlighted, False otherwise
    """
    node_id = coords["id"]

    if is_leaf:
        # For leaf nodes, check if directly highlighted
        return node_id in highlight_edges
    else:
        # For internal nodes, check if any edge to children is highlighted
        return any(
            (node_id, node_coords[child_node]["id"]) in highlight_edges
            for child_node in coords.get("child_nodes", [])
            if child_node in node_coords
        )


def render_leaf_marker(
    marker_group: ET.Element,
    x: float,
    y: float,
    is_highlighted: bool,
    highlight_color: str,
    colors: Dict,
    node_marker_size: float,
    filter_id: Optional[str],
) -> None:
    """
    Render a marker for a leaf node.

    Args:
        marker_group: SVG group for the marker
        x, y: Coordinates of the node
        is_highlighted: Whether the node is highlighted
        highlight_color: Color for highlighted elements
        colors: Color palette
        node_marker_size: Size of the marker
        filter_id: Optional ID of a filter to apply
    """
    # Base fill color
    base_fill = highlight_color if is_highlighted else colors["base_stroke"]

    # Apply filter if provided and node is highlighted
    marker_attrs = {}
    if filter_id and is_highlighted:
        marker_attrs["filter"] = filter_id

    # Leaf nodes get a diamond shape for clear visual distinction
    marker_size = node_marker_size * 0.9  # Slightly smaller for leaf nodes
    diamond_size = marker_size * 1.2

    # Create a diamond (rotated square) for leaf nodes
    ET.SubElement(
        marker_group,
        "rect",
        {
            "x": str(x - diamond_size / 2),
            "y": str(y - diamond_size / 2),
            "width": str(diamond_size),
            "height": str(diamond_size),
            "transform": f"rotate(45, {x}, {y})",
            "fill": base_fill,
            "stroke": "white" if is_highlighted else "none",
            "stroke-width": "0.5",
            **marker_attrs,
        },
    )


def render_internal_marker(
    marker_group: ET.Element,
    x: float,
    y: float,
    is_highlighted: bool,
    highlight_color: str,
    colors: Dict,
    node_marker_size: float,
    filter_id: Optional[str],
) -> None:
    """
    Render a marker for an internal node.

    Args:
        marker_group: SVG group for the marker
        x, y: Coordinates of the node
        is_highlighted: Whether the node is highlighted
        highlight_color: Color for highlighted elements
        colors: Color palette
        node_marker_size: Size of the marker
        filter_id: Optional ID of a filter to apply
    """
    # Base fill color
    base_fill = highlight_color if is_highlighted else colors["base_stroke"]

    # Apply filter if provided and node is highlighted
    marker_attrs = {}
    if filter_id and is_highlighted:
        marker_attrs["filter"] = filter_id

    # Calculate marker sizes
    outer_r = node_marker_size
    inner_r = node_marker_size * 0.5

    if is_highlighted:
        render_highlighted_internal_marker(
            marker_group, x, y, base_fill, outer_r, inner_r, marker_attrs
        )
    else:
        render_normal_internal_marker(
            marker_group, x, y, base_fill, outer_r, inner_r, marker_attrs
        )


def render_highlighted_internal_marker(
    marker_group: ET.Element,
    x: float,
    y: float,
    base_fill: str,
    outer_r: float,
    inner_r: float,
    marker_attrs: Dict,
) -> None:
    """
    Render a highlighted marker for an internal node.

    Args:
        marker_group: SVG group for the marker
        x, y: Coordinates of the node
        base_fill: Fill color for the marker
        outer_r: Outer radius of the marker
        inner_r: Inner radius of the marker
        marker_attrs: Additional marker attributes
    """
    # Highlighted internal nodes get a filled circle with lighter border and increased size
    ET.SubElement(
        marker_group,
        "circle",
        {
            "cx": str(x),
            "cy": str(y),
            "r": str(outer_r * 1.2),  # Increase size for highlighted nodes
            "fill": base_fill,
            "stroke": "white",
            "stroke-width": "1.2",  # Thicker border for better visibility
            "stroke-opacity": "0.8",
            **marker_attrs,
        },
    )

    # Add a small white center for better visual interest
    ET.SubElement(
        marker_group,
        "circle",
        {
            "cx": str(x),
            "cy": str(y),
            "r": str(inner_r * 0.5),
            "fill": "white",
            "fill-opacity": "0.9",
            "stroke": "none",
        },
    )


def render_normal_internal_marker(
    marker_group: ET.Element,
    x: float,
    y: float,
    base_fill: str,
    outer_r: float,
    inner_r: float,
    marker_attrs: Dict,
) -> None:
    """
    Render a normal (non-highlighted) marker for an internal node.

    Args:
        marker_group: SVG group for the marker
        x, y: Coordinates of the node
        base_fill: Fill color for the marker
        outer_r: Outer radius of the marker
        inner_r: Inner radius of the marker
        marker_attrs: Additional marker attributes
    """
    # Non-highlighted internal nodes get double ring for more complex shape
    # Outer ring
    ET.SubElement(
        marker_group,
        "circle",
        {
            "cx": str(x),
            "cy": str(y),
            "r": str(outer_r),
            "fill": "none",
            "stroke": base_fill,
            "stroke-width": "1.4",  # Thicker for better visibility in print
            **marker_attrs,
        },
    )

    # Middle ring
    ET.SubElement(
        marker_group,
        "circle",
        {
            "cx": str(x),
            "cy": str(y),
            "r": str(outer_r * 0.7),  # Middle ring for visual complexity
            "fill": "none",
            "stroke": base_fill,
            "stroke-width": "0.7",
            "stroke-opacity": "0.6",
            **marker_attrs,
        },
    )

    # Add a center dot
    ET.SubElement(
        marker_group,
        "circle",
        {
            "cx": str(x),
            "cy": str(y),
            "r": str(inner_r * 0.7),
            "fill": base_fill,
            "fill-opacity": "0.8",
            "stroke": "none",
        },
    )


def render_leaf_nodes(
    svg_group: ET.Element,
    nodes_to_render: List[Node],
    node_coords: Dict[Node, Dict],
    label_y: float,
    leaf_label_offset: float,
    colors: Dict,
    style: Dict,
    leaf_font_size: Optional[float],
) -> None:
    """
    Render all leaf nodes of the tree.

    Args:
        svg_group: SVG group element to render to
        nodes_to_render: List of nodes to render
        node_coords: Dictionary mapping nodes to their coordinates
        label_y: Y-coordinate for leaf labels
        leaf_label_offset: Offset to apply to leaf label positions
        colors: Dictionary of colors and style information
        style: Dictionary of style settings
        leaf_font_size: Custom font size for leaf labels
    """
    for node in nodes_to_render:
        if not node.is_leaf():
            continue

        coords = node_coords[node]
        # inline validation
        if any(k not in coords for k in ("x", "y", "id", "depth")):
            continue

        render_leaf_node(
            svg_group=svg_group,
            node=node,
            coords=coords,
            label_y=label_y,
            colors=colors,
            highlight_leaves=style["highlight_leaves"],
            highlight_color=style["highlight_leaf_color"],
            highlight_stroke_width=style["highlight_stroke_width"],
            split_depth=style["split_depth"],
            fade_opacity=style["fade_opacity"],
            use_dash=style["use_dash"],
            node_marker_size=style["node_marker_size"],
            leaf_label_offset=leaf_label_offset,
            leaf_font_size=leaf_font_size
        )

def validate_node_coords(coords: Dict, required_keys: List[str]) -> bool:
    """
    Validate that node coordinates contain all required keys.

    Args:
        coords: Node coordinates dictionary
        required_keys: List of required keys

    Returns:
        True if all required keys are present, False otherwise
    """
    return all(k in coords for k in required_keys)
