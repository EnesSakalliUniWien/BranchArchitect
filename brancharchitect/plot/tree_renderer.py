"""
Tree rendering utilities for BranchArchitect.

This module contains functions for rendering complete trees in SVG format.
The code has been modularized for better maintainability and clarity.
"""

import xml.etree.ElementTree as ET
from typing import Dict, Optional, Set, Tuple, List
from brancharchitect.tree import Node
from brancharchitect.plot.tree_utils import get_leaves
from brancharchitect.plot.paper_plot.node_renderers import (
    render_internal_node,
    render_node_markers,
    render_leaf_nodes,
    validate_node_coords,
)
from brancharchitect.plot.paper_plot.paper_plot_constants import (
    QUANTA_COLORS,
    LIGHT_COLORS,
    MATERIAL_LIGHT_COLORS,
    MATERIAL_DARK_COLORS,
    NATURE_MD3_COLORS,
    MD3_SCIENTIFIC_LIGHT,
    MD3_SCIENTIFIC_DARK,
    MD3_SCIENTIFIC_PRINT,
    FONT_SANS_SERIF,
    DEFAULT_SPLIT_DASH,
)

# -------------------- Main Rendering Function --------------------


def render_tree(
    svg_group: ET.Element,
    node_coords: Dict[Node, Dict],
    label_y: float,
    color_mode: str = "quanta",
    highlight_options: Optional[Dict] = None,
    split_options: Optional[Dict] = None,
    leaf_font_size: Optional[float] = None,
    cut_edges: Optional[Set[Tuple[str, str]]] = None,
    leaf_label_offset: float = 0.0,
) -> None:
    """
    Render a tree with the given coordinates and enhanced visual styling.

    Args:
        svg_group: SVG group element to render to
        node_coords: Dictionary mapping nodes to their coordinates
        label_y: Y-coordinate for leaf labels
        color_mode: Color mode ('quanta', 'light', 'material_light', 'material_dark',
                    'nature_md3', 'md3_scientific_light', 'md3_scientific_dark', 'md3_scientific_print')
        highlight_options: Dictionary of highlight options
        split_options: Dictionary of options for split styling
        leaf_font_size: Custom font size for leaf labels
        cut_edges: Set of (parent, child) tuples specifying edges to cut
        leaf_label_offset: Offset to apply to leaf label positions
    """
    # Initialize options dictionaries if not provided
    highlight_options = initialize_options(highlight_options)
    split_options = initialize_options(split_options)

    # Determine color scheme based on color mode
    colors = select_color_scheme(color_mode)

    # Extract style settings from highlight and split options
    style = extract_style_settings(highlight_options, split_options, colors)

    # Find root node and collect nodes to render
    root_node, nodes_to_render = identify_nodes_to_render(node_coords)
    if not root_node:
        return

    # Add caption if specified
    if "caption_text" in style and style["caption_text"]:
        add_caption(svg_group, style["caption_text"], colors)

    # Create SVG definitions for filters and effects
    create_svg_definitions(svg_group, style, colors)

    # Process highlight edges if needed to highlight all edges
    if style["highlight_all_edges"]:
        style["highlight_edges"] = process_all_edges(
            node_coords, style["highlight_edges"]
        )

    # Render the tree components
    render_tree_components(
        svg_group,
        nodes_to_render,
        node_coords,
        label_y,
        leaf_label_offset,
        colors,
        style,
        leaf_font_size,
        cut_edges,
    )


# -------------------- Initialization Functions --------------------


def initialize_options(options: Optional[Dict]) -> Dict:
    """
    Initialize options dictionary if not provided.

    Args:
        options: Dictionary of options or None

    Returns:
        Initialized options dictionary
    """
    return {} if options is None else options


def select_color_scheme(color_mode: str) -> Dict:
    """
    Select the appropriate color scheme based on the color mode.

    Args:
        color_mode: Color mode string or custom dictionary

    Returns:
        Dictionary of colors and style information
    """
    color_schemes = {
        "light": LIGHT_COLORS,
        "material_light": MATERIAL_LIGHT_COLORS,
        "material_dark": MATERIAL_DARK_COLORS,
        "nature_md3": NATURE_MD3_COLORS,
        "md3_scientific_light": MD3_SCIENTIFIC_LIGHT,
        "md3_scientific_dark": MD3_SCIENTIFIC_DARK,
        "md3_scientific_print": MD3_SCIENTIFIC_PRINT,
        "quanta": QUANTA_COLORS,  # Default fallback
    }

    if isinstance(color_mode, dict):
        # Allow passing a custom color dict directly
        return color_mode

    return color_schemes.get(color_mode, QUANTA_COLORS)


def extract_style_settings(
    highlight_options: Dict, split_options: Dict, colors: Dict
) -> Dict:
    """
    Extract and combine style settings from highlight and split options.

    Args:
        highlight_options: Dictionary of highlight options
        split_options: Dictionary of split options
        colors: Dictionary of colors and style information

    Returns:
        Combined dictionary of style settings
    """
    style = {}

    # Extract highlight settings
    style["highlight_leaves"] = highlight_options.get("leaves", set())
    style["highlight_edges"] = highlight_options.get("edges", set())
    style["highlight_leaf_color"] = highlight_options.get(
        "leaf_color", colors["highlight_leaf"]
    )
    style["highlight_branch_color"] = highlight_options.get(
        "branch_color", colors["highlight_branch"]
    )
    style["highlight_stroke_width"] = highlight_options.get(
        "stroke_width", colors["highlight_stroke_width"]
    )
    style["use_waterbrush"] = highlight_options.get("use_waterbrush", False)
    style["use_blobs"] = highlight_options.get(
        "use_blobs", False
    ) or highlight_options.get("use_blob", False)
    style["blob_fill"] = highlight_options.get(
        "blob_fill", style["highlight_branch_color"]
    )
    style["blob_opacity"] = highlight_options.get(
        "blob_opacity", colors.get("blob_opacity", "0.35")
    )
    style["node_glow"] = highlight_options.get("node_glow", True)
    style["branch_gradient"] = highlight_options.get("branch_gradient", True)
    style["caption_text"] = highlight_options.get("caption", None)

    # Check if we should highlight all edges
    style["highlight_all_edges"] = (
        style["highlight_edges"] == set() and "highlight_color" in highlight_options
    )

    # Extract split settings
    style["split_depth"] = split_options.get("depth", float("inf"))
    style["fade_opacity"] = split_options.get("fade_opacity", colors["faded_opacity"])
    style["use_dash"] = split_options.get("use_dash", DEFAULT_SPLIT_DASH)

    # Material Design 3 enhancements
    style["use_elevation"] = highlight_options.get("use_elevation", True)
    style["branch_roundness"] = colors.get("branch_roundness", 4)
    style["node_marker_size"] = colors.get("node_marker_size", 3.5)

    return style


def identify_nodes_to_render(
    node_coords: Dict[Node, Dict],
) -> Tuple[Optional[Node], List[Node]]:
    """
    Find the root node and identify all nodes to render.

    Args:
        node_coords: Dictionary mapping nodes to their coordinates

    Returns:
        Tuple containing the root node and list of nodes to render
    """
    # Find root node for traversal
    root_node = next((n for n, c in node_coords.items() if n.parent is None), None)
    if not root_node:
        root_node = next(iter(node_coords.keys()), None)
    if not root_node:
        return None, []

    # Get nodes to render (must be in node_coords)
    nodes_to_render = [n for n in get_leaves(root_node) if n in node_coords]
    for n in node_coords.keys():
        if n not in nodes_to_render:
            nodes_to_render.append(n)

    return root_node, nodes_to_render


# -------------------- Caption Functions --------------------


def add_caption(svg_group: ET.Element, caption_text: str, colors: Dict) -> None:
    """
    Add a caption to the tree visualization.

    Args:
        svg_group: SVG group element to render to
        caption_text: Text for the caption
        colors: Dictionary of colors and style information
    """
    caption_group = ET.SubElement(svg_group, "g", {"class": "highlight-caption"})
    # Position caption at the top of the tree
    caption_y = 10  # Small offset from top

    caption_el = ET.SubElement(
        caption_group,
        "text",
        {
            "x": "50%",  # Center horizontally
            "y": str(caption_y),
            "text-anchor": "middle",
            "font-family": colors.get("font_family", FONT_SANS_SERIF),
            "font-size": str(
                float(colors.get("font_size", "12")) * 1.0
            ),  # Slightly larger
            "fill": colors.get("caption", "#333333"),  # Darker for better contrast
            "font-weight": "500",  # Medium weight for better readability
        },
    )
    caption_el.text = caption_text


# -------------------- SVG Definition Functions --------------------


def create_svg_definitions(svg_group: ET.Element, style: Dict, colors: Dict) -> None:
    """
    Create SVG definitions for filters and effects.

    Args:
        svg_group: SVG group element to render to
        style: Dictionary of style settings
        colors: Dictionary of colors and style information
    """
    defs = ET.SubElement(svg_group, "defs")

    # Add various filter and effect definitions
    add_branch_shadow_filter(defs)
    add_blob_filter(defs)
    add_node_glow_filter(defs, style["highlight_branch_color"])

    # Add gradient definitions if branch gradient is enabled
    if style["branch_gradient"]:
        add_branch_gradients(defs, colors, style["highlight_branch_color"])

    # Add highlight pattern
    add_highlight_pattern(defs, style["highlight_branch_color"])


def add_branch_shadow_filter(defs: ET.Element) -> None:
    """
    Add a drop shadow filter for branches.

    Args:
        defs: SVG defs element to add the filter to
    """
    # Improved drop shadow with softer edges
    filter_el = ET.SubElement(
        defs, "filter", {"id": "branchShadow", "height": "150%", "width": "150%"}
    )
    ET.SubElement(
        filter_el,
        "feGaussianBlur",
        {"in": "SourceAlpha", "stdDeviation": "1.2", "result": "blur"},
    )
    ET.SubElement(
        filter_el,
        "feOffset",
        {"in": "blur", "dx": "0.8", "dy": "1.0", "result": "offsetBlur"},
    )
    # Add subtle color to the shadow
    ET.SubElement(
        filter_el,
        "feColorMatrix",
        {
            "in": "offsetBlur",
            "type": "matrix",
            "values": "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.35 0",
            "result": "matrixOut",
        },
    )
    fe_merge = ET.SubElement(filter_el, "feMerge")
    ET.SubElement(fe_merge, "feMergeNode", {"in": "matrixOut"})
    ET.SubElement(fe_merge, "feMergeNode", {"in": "SourceGraphic"})


def add_blob_filter(defs: ET.Element) -> None:
    """
    Add a blob effect filter.

    Args:
        defs: SVG defs element to add the filter to
    """
    # Enhanced blob effect with smoother edges
    blob_filter = ET.SubElement(
        defs,
        "filter",
        {
            "id": "enhancedBlob",
            "height": "300%",
            "width": "300%",
            "x": "-100%",
            "y": "-100%",
        },
    )
    ET.SubElement(
        blob_filter,
        "feGaussianBlur",
        {"in": "SourceGraphic", "stdDeviation": "15", "result": "blur"},
    )
    ET.SubElement(
        blob_filter,
        "feColorMatrix",
        {
            "in": "blur",
            "type": "matrix",
            "values": "1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 18 -7",
            "result": "glow",
        },
    )
    ET.SubElement(
        blob_filter,
        "feComposite",
        {"in": "SourceGraphic", "in2": "glow", "operator": "over"},
    )


def add_node_glow_filter(defs: ET.Element, highlight_color: str) -> None:
    """
    Add a glow filter for highlighted nodes.

    Args:
        defs: SVG defs element to add the filter to
        highlight_color: Color for the highlight glow
    """
    # Node glow effect for highlighted nodes
    node_glow_filter = ET.SubElement(
        defs,
        "filter",
        {"id": "nodeGlow", "height": "200%", "width": "200%", "x": "-50%", "y": "-50%"},
    )
    ET.SubElement(
        node_glow_filter,
        "feGaussianBlur",
        {"in": "SourceAlpha", "stdDeviation": "2.5", "result": "blur"},
    )
    ET.SubElement(
        node_glow_filter,
        "feFlood",
        {"flood-color": highlight_color, "flood-opacity": "0.6", "result": "color"},
    )
    ET.SubElement(
        node_glow_filter,
        "feComposite",
        {"in": "color", "in2": "blur", "operator": "in", "result": "glow"},
    )
    node_glow_merge = ET.SubElement(node_glow_filter, "feMerge")
    ET.SubElement(node_glow_merge, "feMergeNode", {"in": "glow"})
    ET.SubElement(node_glow_merge, "feMergeNode", {"in": "SourceGraphic"})


def add_branch_gradients(defs: ET.Element, colors: Dict, highlight_color: str) -> None:
    """
    Add gradient definitions for branches.

    Args:
        defs: SVG defs element to add the gradients to
        colors: Dictionary of colors and style information
        highlight_color: Color for highlighted branches
    """
    # Create a linear gradient for branches
    branch_gradient = ET.SubElement(
        defs,
        "linearGradient",
        {"id": "branchGradient", "x1": "0%", "y1": "0%", "x2": "100%", "y2": "0%"},
    )
    ET.SubElement(
        branch_gradient,
        "stop",
        {
            "offset": "0%",
            "stop-color": colors.get(
                "branch_gradient_start", colors.get("branch_color", "#555555")
            ),
            "stop-opacity": "1",
        },
    )
    ET.SubElement(
        branch_gradient,
        "stop",
        {
            "offset": "100%",
            "stop-color": colors.get("branch_gradient_end", highlight_color),
            "stop-opacity": "1",
        },
    )

    # Highlighted branch gradient
    highlight_gradient = ET.SubElement(
        defs,
        "linearGradient",
        {"id": "highlightGradient", "x1": "0%", "y1": "0%", "x2": "100%", "y2": "0%"},
    )
    ET.SubElement(
        highlight_gradient,
        "stop",
        {"offset": "0%", "stop-color": highlight_color, "stop-opacity": "0.7"},
    )
    ET.SubElement(
        highlight_gradient,
        "stop",
        {"offset": "100%", "stop-color": highlight_color, "stop-opacity": "1"},
    )


def add_highlight_pattern(defs: ET.Element, highlight_color: str) -> None:
    """
    Add a pattern for highlighted areas.

    Args:
        defs: SVG defs element to add the pattern to
        highlight_color: Color for highlights
    """
    # Add a subtle pattern for highlighted areas
    pattern_el = ET.SubElement(
        defs,
        "pattern",
        {
            "id": "highlightPattern",
            "patternUnits": "userSpaceOnUse",
            "width": "10",
            "height": "10",
            "patternTransform": "rotate(45)",
        },
    )
    ET.SubElement(
        pattern_el,
        "line",
        {
            "x1": "0",
            "y1": "0",
            "x2": "0",
            "y2": "10",
            "stroke": highlight_color,
            "stroke-width": "1.2",
            "stroke-opacity": "0.15",
        },
    )


# -------------------- Edge Processing Functions --------------------


def process_all_edges(
    node_coords: Dict[Node, Dict], highlight_edges: Set[Tuple[str, str]]
) -> Set[Tuple[str, str]]:
    """
    Process all edges to highlight when highlight_all_edges is True.

    Args:
        node_coords: Dictionary mapping nodes to their coordinates
        highlight_edges: Current set of edges to highlight

    Returns:
        Updated set of edges to highlight
    """
    processed_highlight_edges = highlight_edges.copy()

    for node, coords in node_coords.items():
        if node.is_leaf() or "child_nodes" not in coords:
            continue

        for child_node in coords.get("child_nodes", []):
            if child_node in node_coords:
                edge = (coords["id"], node_coords[child_node]["id"])
                processed_highlight_edges.add(edge)

    return processed_highlight_edges


# -------------------- Tree Component Rendering Functions --------------------


def render_tree_components(
    svg_group: ET.Element,
    nodes_to_render: List[Node],
    node_coords: Dict[Node, Dict],
    label_y: float,
    leaf_label_offset: float,
    colors: Dict,
    style: Dict,
    leaf_font_size: Optional[float],
    cut_edges: Optional[Set[Tuple[str, str]]],
) -> None:
    """
    Render all components of the tree.

    Args:
        svg_group: SVG group element to render to
        nodes_to_render: List of nodes to render
        node_coords: Dictionary mapping nodes to their coordinates
        label_y: Y-coordinate for leaf labels
        leaf_label_offset: Offset to apply to leaf label positions
        colors: Dictionary of colors and style information
        style: Dictionary of style settings
        leaf_font_size: Custom font size for leaf labels
        cut_edges: Set of edges to cut
    """
    # First render all internal branches
    render_internal_branches(
        svg_group, nodes_to_render, node_coords, colors, style, cut_edges
    )

    # Then render all leaf nodes and labels
    render_leaf_nodes(
        svg_group,
        nodes_to_render,
        node_coords,
        label_y,
        leaf_label_offset,
        colors,
        style,
        leaf_font_size,
    )

    # Render node markers for all nodes
    render_node_markers_for_all(svg_group, nodes_to_render, node_coords, colors, style)


def render_internal_branches(
    svg_group: ET.Element,
    nodes_to_render: List[Node],
    node_coords: Dict[Node, Dict],
    colors: Dict,
    style: Dict,
    cut_edges: Optional[Set[Tuple[str, str]]],
) -> None:
    """
    Render all internal branches of the tree.

    Args:
        svg_group: SVG group element to render to
        nodes_to_render: List of nodes to render
        node_coords: Dictionary mapping nodes to their coordinates
        colors: Dictionary of colors and style information
        style: Dictionary of style settings
        cut_edges: Set of edges to cut
    """
    for node in nodes_to_render:
        if node.is_leaf():
            continue

        coords = node_coords[node]
        if not validate_node_coords(coords, ["x", "y", "id", "depth", "child_nodes"]):
            continue

        render_internal_node(
            svg_group,
            node,
            coords,
            node_coords,
            colors,
            style["highlight_edges"],
            style["highlight_branch_color"],
            style["highlight_stroke_width"],
            style["split_depth"],
            style["fade_opacity"],
            style["use_dash"],
            style["use_waterbrush"],
            style["use_blobs"],
            style["branch_roundness"],
            style["use_elevation"],
            style["blob_fill"],
            style["blob_opacity"],
            cut_edges,
        )


def render_node_markers_for_all(
    svg_group: ET.Element,
    nodes_to_render: List[Node],
    node_coords: Dict[Node, Dict],
    colors: Dict,
    style: Dict,
) -> None:
    """
    Render markers for all nodes in the tree.

    Args:
        svg_group: SVG group element to render to
        nodes_to_render: List of nodes to render
        node_coords: Dictionary mapping nodes to their coordinates
        colors: Dictionary of colors and style information
        style: Dictionary of style settings
    """
    if style["node_marker_size"] <= 0:
        return

    for node in nodes_to_render:
        coords = node_coords[node]

        # Check if this node is part of a highlighted edge
        is_highlighted = is_node_highlighted(coords["id"], style["highlight_edges"])

        # Apply glow filter for highlighted nodes
        filter_id = "url(#nodeGlow)" if is_highlighted and style["node_glow"] else None

        render_node_markers(
            svg_group,
            node,
            coords,
            node_coords,
            colors,
            style["highlight_edges"],
            style["highlight_branch_color"],
            is_leaf=node.is_leaf(),
            filter_id=filter_id,
        )


# -------------------- Helper Functions --------------------



def is_node_highlighted(node_id: str, highlight_edges: Set[Tuple[str, str]]) -> bool:
    """
    Check if a node is part of a highlighted edge.

    Args:
        node_id: ID of the node to check
        highlight_edges: Set of highlighted edges

    Returns:
        True if the node is part of a highlighted edge, False otherwise
    """
    return any((node_id == edge[0] or node_id == edge[1]) for edge in highlight_edges)
