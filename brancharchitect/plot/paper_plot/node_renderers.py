# --- node_rendering.py (Cleaned, Internal Markers Always Black Dot + Shadow) ---
"""
Node rendering utilities for BranchArchitect.

This module contains functions for rendering tree nodes and branches in SVG format.
Internal node markers are always rendered as black dots with shadows.
"""

import xml.etree.ElementTree as ET
from typing import Dict, Set, Optional, Tuple, List, Union
import uuid

# --- Assumed Correct Imports (Adjust Path If Necessary) ---
from brancharchitect.tree import Node
from brancharchitect.plot.tree_utils import get_node_label
from brancharchitect.plot.paper_plot.paper_plot_constants import FONT_SANS_SERIF
from brancharchitect.plot.paper_plot.branch_effects import (
    find_or_create_defs,
    render_blob_branch,
    render_waterbrush_branch,
    apply_elevation_effect,
    apply_cut_edge_styling,
    create_line_style,
)

# --- Helper Functions (Defined here if not imported from effects) ---
# These are needed by rendering functions below


def add_non_scaling(style: Dict[str, str]) -> Dict[str, str]:
    """PDF-optimized: Do not add vector-effect (not PDF-safe)."""
    return style


def calculate_bezier_control_points(
    x: float,
    y: float,
    cx: float,
    cy: float,
    roundness: float,
    angle_rad: float = 0.0,
    offset_x: float = 0.0,
) -> Dict[str, float]:
    dx, dy = cx - x, cy - y
    roundness = max(0.0, min(roundness, 10.0))
    t = min(0.4 + roundness / 25.0, 0.8)
    # For polytomies, offset control points horizontally by offset_x
    p1 = (x + offset_x, y + dy * t)
    p2 = (cx + offset_x, y + dy * (1 - t))
    if abs(dx) < 1e-6:  # vertical
        p1 = p2 = (x + offset_x, y + dy * 0.5)
    elif abs(dy) < 1e-6:  # horizontal
        p1 = p2 = (x + dx * 0.5 + offset_x, y)
    return {"cx1": p1[0], "cy1": p1[1], "cx2": p2[0], "cy2": p2[1]}


def create_path_data(
    x: float, y: float, cp: Dict[str, float], cx: float, cy: float
) -> str:
    def FMT(f: float) -> str:
        return f"{f:.3f}"  # 3‑decimal formatter (use everywhere)
    return (
        f"M {FMT(x)},{FMT(y)} C {FMT(cp['cx1'])},{FMT(cp['cy1'])} "
        f"{FMT(cp['cx2'])},{FMT(cp['cy2'])} {FMT(cx)},{FMT(cy)}"
    )


# --- Core Rendering Functions ---


def render_internal_node(
    svg_group: ET.Element,
    node: Node,
    coords: Dict,
    node_coords: Dict[Node, Dict],
    colors: Dict,
    highlight_edges: Union[Set[Tuple[str, str]], Dict[Tuple[str, str], Dict]],
    highlight_color: str,  # Base color for highlighted elements (e.g., gradient start)
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
    """Renders an internal node's branches."""
    node_id = coords.get("id")
    if not node_id:
        print(
            f"Warning: Node {node} missing 'id' in coords during internal node rendering."
        )
        return

    node_group = create_node_group(svg_group, node_id)
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
    """Creates an SVG group element for a node."""
    return ET.SubElement(
        svg_group, "g", {"class": "internal-node", "data-node-id": node_id}
    )


def render_node_branches(
    node_group: ET.Element,
    coords: Dict,
    node_coords: Dict[Node, Dict],
    colors: Dict,
    highlight_edges: Union[Set[Tuple[str, str]], Dict[Tuple[str, str], Dict]],
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
    """Renders all branches originating from a node, fanning out polytomies."""
    x, y = coords.get("x"), coords.get("y")
    node_id = coords.get("id")
    if x is None or y is None or node_id is None:
        print(
            f"Warning: Parent node {node_id or coords} missing coordinates or id for branch rendering."
        )
        return

    child_nodes = coords.get("child_nodes", [])
    n_children = len(child_nodes)
    # For polytomies, fan out branches using angles
    if n_children > 2:
        angle_range = 60  # degrees, total spread
        angle_start = -angle_range / 2
        angle_step = angle_range / (n_children - 1) if n_children > 1 else 0
        for i, child_node in enumerate(child_nodes):
            child_coords = node_coords.get(child_node)
            if not child_coords or not validate_node_coords(
                child_coords, ["x", "y", "id", "depth"]
            ):
                continue
            child_x, child_y = child_coords["x"], child_coords["y"]
            child_id, child_depth = child_coords["id"], child_coords["depth"]
            branch_group = create_branch_group(node_group, node_id, child_id)
            branch_state = determine_branch_state(
                node_id, child_id, child_depth, split_depth, highlight_edges, cut_edges
            )
            # Calculate angle for this child
            angle_deg = angle_start + i * angle_step
            angle_rad = angle_deg * 3.14159265 / 180.0
            # Offset control points horizontally by a factor of the branch length
            branch_len = ((child_x - x) ** 2 + (child_y - y) ** 2) ** 0.5
            offset = branch_len * 0.25 * (1 if n_children > 3 else 0.18)
            offset_x = offset * (angle_rad)
            # Pass angle and offset to render_branch
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
                angle_rad,
                offset_x,
            )
    else:
        for child_node in child_nodes:
            child_coords = node_coords.get(child_node)
            if not child_coords or not validate_node_coords(
                child_coords, ["x", "y", "id", "depth"]
            ):
                continue
            child_x, child_y = child_coords["x"], child_coords["y"]
            child_id, child_depth = child_coords["id"], child_coords["depth"]
            branch_group = create_branch_group(node_group, node_id, child_id)
            branch_state = determine_branch_state(
                node_id, child_id, child_depth, split_depth, highlight_edges, cut_edges
            )
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
    """Creates an SVG group for a branch."""
    return ET.SubElement(
        parent_group,
        "g",
        {"class": "branch", "data-edge-id": f"{parent_id}-{child_id}"},
    )


def determine_branch_state(
    parent_id: str,
    child_id: str,
    child_depth: float,
    split_depth: float,
    highlight_edges: Union[Set[Tuple[str, str]], Dict[Tuple[str, str], Dict]],
    cut_edges: Optional[Set[Tuple[str, str]]],
) -> Dict:
    """Determines the visual state (highlighted, cut, faded) and style overrides for a branch."""
    edge_id = (parent_id, child_id)
    edge_style = {}
    is_highlighted = False

    if isinstance(highlight_edges, dict):
        if edge_id in highlight_edges:
            is_highlighted = True
            edge_style = highlight_edges.get(edge_id) or {}  # Ensure dict
            if not isinstance(edge_style, dict):
                print(f"Warning: Invalid edge style for {edge_id}. Resetting.")
                edge_style = {}
    elif isinstance(highlight_edges, set):
        is_highlighted = edge_id in highlight_edges

    is_cut = bool(cut_edges and edge_id in cut_edges)
    # Faded only if deep enough AND not highlighted or cut
    is_faded = child_depth >= split_depth and not is_highlighted and not is_cut

    return {
        "is_highlighted": is_highlighted,
        "is_cut": is_cut,
        "is_faded": is_faded,
        "edge_style": edge_style,
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
    angle_rad: float = 0.0,
    offset_x: float = 0.0,
) -> None:
    """Renders a single branch with appropriate style based on its state."""
    is_highlighted = branch_state["is_highlighted"]
    is_cut = branch_state["is_cut"]
    is_faded = branch_state["is_faded"]
    edge_style = branch_state.get("edge_style", {})

    # Determine effective styles, allowing overrides from edge_style
    color = edge_style.get(
        "highlight_color",
        highlight_color if is_highlighted else colors.get("base_stroke", "#000000"),
    )
    stroke_width = str(
        edge_style.get(
            "stroke_width",
            (
                highlight_stroke_width
                if is_highlighted
                else colors.get("default_stroke_width", "1.5")
            ),
        )
    )
    opacity = str(edge_style.get("opacity", fade_opacity if is_faded else "1.0"))
    edge_use_waterbrush = edge_style.get("use_waterbrush", use_waterbrush)
    edge_use_blobs = edge_style.get("use_blobs", use_blobs)
    gradient_end_color = edge_style.get("gradient_end")  # Used by standard/waterbrush
    """PDF-optimized: Render as a normal or gradient line (no filter)."""
    control_points = calculate_bezier_control_points(x, y, child_x, child_y, 0)
    path_data = create_path_data(x, y, control_points, child_x, child_y)

    # --- Select Rendering Mode ---
    if is_highlighted and edge_use_blobs:
        eff_blob_fill = edge_style.get("blob_fill", blob_fill or color)
        eff_blob_opacity = edge_style.get("blob_opacity", blob_opacity)
        render_blob_branch(
            branch_group, x, y, child_x, child_y, eff_blob_fill, eff_blob_opacity
        )
    elif is_highlighted and edge_use_waterbrush:
        render_waterbrush_branch(
            branch_group,
            x,
            y,
            child_x,
            child_y,
            color,
            stroke_width,
            opacity,
            gradient_end=gradient_end_color,
            control_points=control_points,
            path_data=path_data,
        )
    else:
        # Standard rendering (handles normal, highlighted, faded, cut, gradients)
        control_points = calculate_bezier_control_points(
            x, y, child_x, child_y, branch_roundness, angle_rad, offset_x
        )
        render_standard_branch(
            branch_group,
            x,
            y,
            child_x,
            child_y,
            branch_state,
            is_highlighted,
            is_faded,
            is_cut,
            color,
            colors,
            stroke_width,
            opacity,
            use_dash,
            branch_roundness,
            use_elevation,
            gradient_end_color,
            control_points,
        )




def render_standard_branch(
    branch_group: ET.Element,
    x: float,
    y: float,
    child_x: float,
    child_y: float,
    branch_state: Dict,
    is_highlighted: bool,
    is_faded: bool,
    is_cut: bool,
    color: str,
    colors: Dict,
    stroke_width: str,
    opacity: str,
    use_dash: bool,
    branch_roundness: float,
    use_elevation: bool,
    gradient_end_color: Optional[str],
    control_points: Dict[str, float],
) -> None:
    """Renders a standard branch (line/curve) with optional effects (fade, dash, cut, gradient, elevation)."""
    edge_style = branch_state.get("edge_style", {})

    # --- Path Calculation ---
    path_data = create_path_data(x, y, control_points, child_x, child_y)

    # --- Base Style Calculation ---
    dash_pattern = "none"
    if is_faded and use_dash and not is_cut:
        dash_pattern = "3,3" if branch_roundness > 0 else "4,4"
    dash_pattern = edge_style.get("stroke-dasharray", dash_pattern)  # Allow override
    line_style = create_line_style(color, stroke_width, opacity, dash_pattern)
    line_style = add_non_scaling(line_style)  # Non-scaling stroke

    # --- Gradient Handling ---
    # MODIFIED BLOCK START
    if is_highlighted and not is_cut:
        gradient_url = _create_branch_gradient(
            branch_group, color, gradient_end_color, x, y, child_x, child_y
        )
        if gradient_url:
            line_style["stroke"] = gradient_url
    # MODIFIED BLOCK END

    # --- Cut Edge Handling ---
    if is_cut:
        apply_cut_edge_styling(branch_group, line_style, x, y, child_x, child_y, colors)
        ET.SubElement(branch_group, "path", {**{"d": path_data}, **line_style})
        return  # Cut edge rendering finished

    # --- Elevation Effect Handling ---
    if use_elevation and is_highlighted:
        # apply_elevation_effect draws the glow/shadow AND the main path itself
        apply_elevation_effect(
            branch_group, path_data, line_style, stroke_width, opacity, color
        )
    else:
        # Draw main path if not cut and elevation didn't draw it
        ET.SubElement(branch_group, "path", {**{"d": path_data}, **line_style})



def _create_branch_gradient(
    branch_group: ET.Element,
    start_color: str,
    end_color: str,
    x1_coord: float,
    y1_coord: float,
    x2_coord: float,
    y2_coord: float,
) -> Optional[str]:
    """
    Creates an SVG linear gradient definition within the branch_group's <defs>
    and returns the 'url(#gradient_id)' string for styling.

    Returns None if gradient creation fails or end_color is not provided.
    """
    if not end_color:
        return None

    try:
        defs = find_or_create_defs(branch_group)
        gradient_id = f"branchGradient_{uuid.uuid4().hex[:8]}"

        # Determine gradient direction (horizontal or vertical)
        is_horizontal = abs(x2_coord - x1_coord) > abs(y2_coord - y1_coord)
        g_x1, g_y1, g_x2, g_y2 = (
            ("0%", "0%", "100%", "0%")
            if is_horizontal
            else ("0%", "0%", "0%", "100%")
        )

        gradient = ET.SubElement(
            defs,
            "linearGradient",
            {"id": gradient_id, "x1": g_x1, "y1": g_y1, "x2": g_x2, "y2": g_y2},
        )
        ET.SubElement(gradient, "stop", {"offset": "0%", "stop-color": start_color})
        ET.SubElement(gradient, "stop", {"offset": "100%", "stop-color": end_color})

        return f"url(#{gradient_id})"
    except Exception as e:
        print(f"Error creating gradient: {e}")
        return None




# --- LEAF NODE RENDERING ---


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
    """Renders all leaf nodes in the provided list."""
    highlight_leaves = style.get("highlight_leaves", set())
    highlight_color = style.get(
        "highlight_leaf_color", style.get("highlight_color", "#D81B60")
    )
    highlight_stroke_width = style.get("highlight_stroke_width", "2.0")
    split_depth = style.get("split_depth", float("inf"))
    fade_opacity = style.get("fade_opacity", 0.5)
    use_dash = style.get("use_dash", True)

    for node in nodes_to_render:
        if not node.is_leaf():
            continue
        coords = node_coords.get(node)
        if not coords or not validate_node_coords(coords, ["x", "y", "id", "depth"]):
            print(
                f"Warning: Leaf node {getattr(node, 'id', node)} has incomplete/missing coords. Skipping."
            )
            continue

        render_leaf_node(
            svg_group,
            node,
            coords,
            label_y,
            colors,
            highlight_leaves,
            highlight_color,
            highlight_stroke_width,
            split_depth,
            fade_opacity,
            use_dash,
            leaf_font_size,
            leaf_label_offset,
        )

def render_leaf_node(
    svg_group: ET.Element,
    node: Node,
    coords: Dict,
    label_y: float,
    colors: Dict,
    highlight_leaves: Union[Set[str], Dict[str, Dict]],
    highlight_color: str,
    highlight_stroke_width: str,
    split_depth: float,
    fade_opacity: float,
    use_dash: bool,
    leaf_font_size: Optional[float],
    leaf_label_offset: float,
) -> None:
    """Renders a single leaf node (connector line and text label)."""
    x, y = coords["x"], coords["y"]
    node_id, depth = coords["id"], coords["depth"]

    leaf_group = create_leaf_group(svg_group, node_id)
    leaf_state = determine_leaf_state(
        node_id, depth, split_depth, highlight_leaves, fade_opacity
    )
    label_text = get_node_label(node)

    # Use default stroke width for non-highlighted leaves, highlight width if highlighted
    connector_stroke_width = (
        highlight_stroke_width
        if leaf_state["is_highlighted"]
        else colors.get("default_stroke_width", "1.5")
    )

    render_leaf_connector_line(
        leaf_group,
        x,
        y,
        label_y,
        leaf_state,
        colors,
        highlight_color,
        connector_stroke_width,
        use_dash,
    )

    if label_text:
        leaf_style = leaf_state.get("leaf_style", {})
        eff_highlight_color = leaf_style.get("highlight_color", highlight_color)
        eff_font_size = leaf_style.get("font_size", leaf_font_size)
        render_leaf_label_text(
            leaf_group,
            x,
            label_y + leaf_label_offset,
            label_text,
            leaf_state,
            colors,
            eff_highlight_color,
            eff_font_size,
        )


def create_leaf_group(svg_group: ET.Element, node_id: str) -> ET.Element:
    """Creates an SVG group for a leaf node."""
    return ET.SubElement(
        svg_group, "g", {"class": "leaf-node", "data-node-id": node_id}
    )


def determine_leaf_state(
    node_id: str,
    depth: float,
    split_depth: float,
    highlight_leaves: Union[Set[str], Dict[str, Dict]],
    fade_opacity: float = 0.5,
) -> Dict:
    """Determines the visual state for a leaf node."""
    is_highlighted = False
    leaf_style = {}
    if isinstance(highlight_leaves, dict):
        if node_id in highlight_leaves:
            is_highlighted = True
            leaf_style = highlight_leaves.get(node_id) or {}
            if not isinstance(leaf_style, dict):
                leaf_style = {}  # Reset if invalid
    elif isinstance(highlight_leaves, set):
        is_highlighted = node_id in highlight_leaves

    is_faded = depth >= split_depth and not is_highlighted
    return {
        "is_highlighted": is_highlighted,
        "is_faded": is_faded,
        "opacity": str(fade_opacity) if is_faded else "1.0",
        "leaf_style": leaf_style,
    }


def render_leaf_connector_line(
    leaf_group: ET.Element,
    x: float,
    y: float,
    label_y: float,
    leaf_state: Dict,
    colors: Dict,
    highlight_color: str,
    stroke_width: str,
    use_dash: bool,
) -> None:
    """Renders the vertical connector line for a leaf node."""
    is_highlighted = leaf_state["is_highlighted"]
    is_faded = leaf_state["is_faded"]
    opacity = leaf_state["opacity"]
    leaf_style = leaf_state.get("leaf_style", {})

    conn_color = leaf_style.get(
        "connector_color",
        highlight_color if is_highlighted else colors.get("base_stroke", "#AAAAAA"),
    )
    conn_stroke_width = str(leaf_style.get("connector_stroke_width", stroke_width))
    conn_dash = leaf_style.get(
        "connector_dasharray", "3,3" if (is_faded and use_dash) else "none"
    )
    conn_opacity = str(leaf_style.get("connector_opacity", opacity))

    connector_style = {
        "stroke": conn_color,
        "stroke-width": conn_stroke_width,
        "stroke-opacity": conn_opacity,
        "fill": "none",
        "stroke-linecap": "round",
    }
    if conn_dash != "none":
        connector_style["stroke-dasharray"] = conn_dash

    path_data = (
        f"M {x:.2f},{y:.2f} V {label_y - 5:.2f}"  # Line ends slightly above label Y
    )
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
    """Renders the text label for a leaf node."""
    is_highlighted = leaf_state["is_highlighted"]
    opacity = leaf_state["opacity"]
    leaf_style = leaf_state.get("leaf_style", {})

    font_size = calculate_font_size(leaf_font_size, colors)  # Base size
    font_size = leaf_style.get("font_size", font_size)  # Override
    default_weight = "500" if is_highlighted else "400"
    font_weight = str(leaf_style.get("font_weight", default_weight))
    text_color = leaf_style.get(
        "color",
        highlight_color if is_highlighted else colors.get("base_text", "#000000"),
    )
    text_opacity = str(leaf_style.get("opacity", opacity))

    text_style = {
        "font-family": leaf_style.get(
            "font_family", colors.get("font_family", FONT_SANS_SERIF)
        ),
        "font-size": str(font_size),
        "font-weight": font_weight,
        "text-anchor": "middle",
        "dominant-baseline": "central",
        "fill": text_color,
        "opacity": text_opacity,
    }

    text_el = ET.SubElement(
        leaf_group, "text", {**{"x": f"{x:.2f}", "y": f"{label_y:.2f}"}, **text_style}
    )
    text_el.text = label_text


def calculate_font_size(leaf_font_size: Optional[float], colors: Dict) -> float:
    """Calculates the font size for a leaf label, handling custom inputs and defaults."""
    if leaf_font_size is not None:
        try:
            return max(6.0, min(30.0, float(leaf_font_size)))
        except ValueError:
            print(
                f"Warning: Invalid custom leaf_font_size '{leaf_font_size}'. Using default."
            )
    try:
        base_font_size = float(colors.get("font_size", 10.0))
        return max(8.0, min(14.0, base_font_size))  # Clamp default for print
    except (ValueError, KeyError):
        return 10.0  # Fallback


# --- NODE MARKER RENDERING ---


def render_node_markers(
    svg_group: ET.Element,
    node: Node,
    coords: Dict,
    node_coords: Dict[Node, Dict],
    colors: Dict,
    is_leaf: bool = False,
    filter_id: Optional[str] = None,
) -> None:
    """Renders node markers (dots for internal nodes)."""
    node_marker_size = get_node_marker_size(colors)
    if node_marker_size <= 0:
        return  # Skip if size is zero or negative

    node_id = coords.get("id")
    x, y = coords.get("x"), coords.get("y")
    if not node_id or x is None or y is None:
        print(
            f"Warning: Cannot render marker for node {node_id or node}, missing data."
        )
        return

    marker_group = create_marker_group(svg_group, node_id)

    if is_leaf:
        # No leaf marker rendering implemented by default
        pass
    else:
        # Internal nodes: always render black dot with shadow
        render_internal_marker(
            marker_group,
            x,
            y,
            colors,
            node_marker_size,
            filter_id,  # Pass filter ID if provided (e.g., for glow)
            {},  # Pass overrides (e.g., for size factor, shadow toggle)
        )


# ...existing code...
def get_node_marker_size(colors: Dict) -> float:
    """Gets the node marker size from colors dict, ensuring it's non-negative."""
    try:
        size_val = colors.get("node_marker_size", 4.5) # Default size if not found
        return max(0.0, float(size_val))
    except ValueError:
        print("DEBUG [get_node_marker_size]: ValueError encountered. Returning default 4.5") # ADDED DEBUG
        return 4.5
# ...existing code...


def create_marker_group(svg_group: ET.Element, node_id: str) -> ET.Element:
    """Creates an SVG group for a node marker."""
    return ET.SubElement(
        svg_group, "g", {"class": "node-marker", "data-node-id": node_id}
    )


def render_internal_marker(
    marker_group: ET.Element,
    x: float,
    y: float,
    colors: Dict,
    node_marker_size: float,
    filter_id: Optional[str],
    style_override: Dict,
) -> None:
    """
    Renders a marker for an internal node.
    PDF-optimized: Only vector shapes, no SVG filters.
    """
    inner_r_factor = style_override.get("inner_radius_factor", 0.6)
    inner_r = max(1.0, node_marker_size * inner_r_factor)
    marker_fill = "#000000"  # Always black

    # --- Simulated shadow/glow as a lighter, larger circle (vector only) ---
    shadow_opacity = style_override.get("shadow_opacity", "0.18")
    shadow_color = style_override.get("shadow_color", "#333333")
    shadow_radius_factor = style_override.get("shadow_radius_factor", 1.5)
    
    ET.SubElement(
        marker_group,
        "circle",
        {
            "cx": str(x),
            "cy": str(y),
            "r": str(inner_r * shadow_radius_factor),
            "fill": shadow_color,
            "fill-opacity": str(shadow_opacity),
            "stroke": "none",
            "class": "marker-shadow",
        },
    )

    # --- Main Marker Circle (Always Black, no filter) ---
    marker_attrs = {
        "cx": str(x),
        "cy": str(y),
        "r": str(inner_r),
        "fill": marker_fill,
        "fill-opacity": style_override.get("opacity", "1.0"),
        "class": "marker-dot",
    }
    ET.SubElement(marker_group, "circle", marker_attrs)


# --- UTILITY FUNCTIONS ---


def validate_node_coords(coords: Optional[Dict], required_keys: List[str]) -> bool:
    """Validates that node coordinates dictionary contains all required keys."""
    if not isinstance(coords, dict):
        return False
    return all(k in coords and coords[k] is not None for k in required_keys)
