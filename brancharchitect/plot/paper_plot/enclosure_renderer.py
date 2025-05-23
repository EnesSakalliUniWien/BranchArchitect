"""
Enclosure rendering utilities for BranchArchitect.

This module contains functions for rendering enclosures around subtrees.
The code has been modularized for better maintainability and clarity.
"""

import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

from brancharchitect.tree import Node
from brancharchitect.plot.tree_utils import find_node, get_leaves
from brancharchitect.plot.paper_plot.paper_plot_constants import (
    DEFAULT_ENCLOSE_PADDING,
    DEFAULT_ENCLOSE_STROKE_WIDTH,
    DEFAULT_ENCLOSE_DASHARRAY,
    FONT_SANS_SERIF,
)

# --- MAIN ENCLOSURE RENDERING FUNCTION ---


def render_enclosures(
    svg_group: ET.Element,
    layout: Dict,
    enclose_options: Dict,
    colors: Dict,
    leaf_label_offset: float = 0.0,
) -> None:
    """
    Render enclosures around subtrees following Material Design 3 container principles.

    Args:
        svg_group: SVG group element to render to
        layout: Layout information
        enclose_options: Options for enclosing subtrees
        colors: Color palette
        leaf_label_offset: Offset for leaf labels
    """
    if not enclose_options:
        return

    # Extract necessary data from layout
    processed_root = layout["processed_root"]
    node_coords = layout["coords"]
    label_y = layout["label_y"]

    # Create a dedicated background group for containers
    container_background_group = ET.SubElement(svg_group, "g", {
        "class": "container-background-layer", 
    })

    # Create container definitions
    create_container_definitions(container_background_group)

    # Process each subtree enclosure
    for subtree_id, style in enclose_options.items():
        # Find the subtree root node
        subtree_root = find_subtree_root(processed_root, subtree_id)
        if not subtree_root:
            continue

        # Collect all nodes in the subtree
        subtree_nodes, leaf_nodes = collect_subtree_nodes(subtree_root)

        # Get coordinates for nodes in the subtree
        all_node_coords, leaf_coords = get_node_coordinates(
            subtree_nodes, leaf_nodes, node_coords
        )

        # Make sure we have coordinates to work with
        if not all_node_coords:
            continue

        # Calculate boundaries of the enclosure
        boundaries = calculate_enclosure_boundaries(
            all_node_coords,
            leaf_coords,
            label_y,
            style,
            colors,
            leaf_label_offset=leaf_label_offset,
        )

        # Adjust boundaries for parent connection if needed
        boundaries = adjust_for_parent_connection(boundaries, subtree_root, node_coords)

        # Calculate corner radius based on container size
        corner_radius = calculate_corner_radius(
            boundaries["width"], boundaries["height"], style
        )

        # Create container group
        container_group = create_container_group(container_background_group, subtree_id)

        # Render the container rectangle
        render_container_rectangle(
            container_group, boundaries, corner_radius, style, colors
        )

        # Add container label if specified
        add_container_label(container_group, boundaries, style, colors)


# --- HELPER FUNCTIONS ---


def create_container_definitions(svg_group: ET.Element) -> None:
    """y
    Create SVG definitions for container styling.

    Args:
        svg_group: SVG group element to add definitions to
    """
    defs_id = "md3_container_defs"
    existing_defs = svg_group.find(f"./defs[@id='{defs_id}']")

    if existing_defs is None:
        defs = ET.SubElement(svg_group, "defs", {"id": defs_id})

        # For print publication we'll use subtle patterns instead of shadows
        # Add subtle container pattern for MD3 style containers
        pattern_el = ET.SubElement(
            defs,
            "pattern",
            {
                "id": "md3ContainerPattern",
                "patternUnits": "userSpaceOnUse",
                "width": "8",
                "height": "8",
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
                "y2": "8",
                "stroke": "#000000",
                "stroke-width": "0.5",
                "stroke-opacity": "0.05",
            },
        )


def find_subtree_root(processed_root: Node, subtree_id: str) -> Optional[Node]:
    """
    Find the root node of a subtree by ID or name.

    Args:
        processed_root: Root node of the entire tree
        subtree_id: ID or name of the subtree root

    Returns:
        Root node of the subtree if found, None otherwise
    """
    # First try finding by exact ID
    subtree_root = find_node(processed_root, subtree_id)

    # If not found, try finding by name
    if not subtree_root:
        all_nodes = []

        def gather_all_nodes(node: Node) -> None:
            if node:
                all_nodes.append(node)
                for child in getattr(node, "children", []):
                    gather_all_nodes(child)

        gather_all_nodes(processed_root)

        # Try to find a node with matching name
        for node in all_nodes:
            if node.name == subtree_id:
                subtree_root = node
                break

    return subtree_root


def collect_subtree_nodes(subtree_root: Node) -> Tuple[List[Node], List[Node]]:
    """
    Collect all nodes and leaf nodes in a subtree.

    Args:
        subtree_root: Root node of the subtree

    Returns:
        Tuple containing (all_nodes, leaf_nodes) in the subtree
    """
    subtree_nodes = []

    # Helper function to collect all nodes in a subtree
    def collect_nodes(node: Node) -> None:
        if node:
            subtree_nodes.append(node)
            for child in getattr(node, "children", []):
                collect_nodes(child)

    # Collect all nodes starting from subtree root
    collect_nodes(subtree_root)

    # Get all leaves specifically
    leaf_nodes = get_leaves(subtree_root)

    return subtree_nodes, leaf_nodes


def get_node_coordinates(
    subtree_nodes: List[Node], leaf_nodes: List[Node], node_coords: Dict[Node, Dict]
) -> Tuple[List[Dict], List[Dict]]:
    """
    Get coordinates for nodes in a subtree.

    Args:
        subtree_nodes: All nodes in the subtree
        leaf_nodes: Leaf nodes in the subtree
        node_coords: Dictionary mapping nodes to their coordinates

    Returns:
        Tuple containing (all_node_coords, leaf_coords)
    """
    # Get coordinates for all nodes that are in the node_coords dictionary
    all_node_coords = [
        node_coords[node] for node in subtree_nodes if node in node_coords
    ]

    # Get coordinates for leaf nodes
    leaf_coords = [node_coords[leaf] for leaf in leaf_nodes if leaf in node_coords]

    return all_node_coords, leaf_coords


def calculate_enclosure_boundaries(
    all_node_coords: List[Dict],
    leaf_coords: List[Dict],
    label_y: float,
    style: Dict,
    colors: Dict,
    leaf_label_offset: float = 0.0,
) -> Dict:
    """
    Calculate the boundaries of an enclosure.

    Args:
        all_node_coords: Coordinates of all nodes in the subtree
        leaf_coords: Coordinates of leaf nodes in the subtree
        label_y: Y-coordinate of labels
        style: Style options for the enclosure
        colors: Color palette
        leaf_label_offset: Offset for leaf labels

    Returns:
        Dictionary containing enclosure boundaries (x, y, width, height)
    """
    # Calculate enclosure boundaries for all nodes in the subtree
    min_x = min(c["x"] for c in all_node_coords)
    max_x = max(c["x"] for c in all_node_coords)
    min_y = min(c["y"] for c in all_node_coords)
    max_node_y = max(c["y"] for c in all_node_coords)

    # For leaf labels, we need to extend to label_y
    # This ensures leaf labels are fully enclosed
    if leaf_coords:
        # For leaf nodes, extend max_y to include labels and offset
        max_y = label_y + leaf_label_offset + 5  # Add a little extra space below labels
    else:
        max_y = max_node_y

    # Add padding (MD3 uses consistent spacing)
    boundaries = calculate_padding(min_x, min_y, max_x, max_y, style, colors)

    return boundaries


def calculate_padding(
    min_x: float, min_y: float, max_x: float, max_y: float, style: Dict, colors: Dict
) -> Dict:
    """
    Calculate padding for enclosure boundaries.

    Args:
        min_x: Minimum x-coordinate of enclosed nodes.
        min_y: Minimum y-coordinate of enclosed nodes.
        max_x: Maximum x-coordinate of enclosed nodes.
        max_y: Maximum y-coordinate of enclosed nodes.
        style: Dictionary containing styling options.
        colors: Dictionary containing color options.

    Returns:
        Dictionary containing calculated rectangle properties (x, y, width, height)
        and original bounds (min_x, min_y, max_x, max_y) and the effective padding used.
    """
    # Use the padding specified in the style directly
    effective_padding = float(style.get("padding", DEFAULT_ENCLOSE_PADDING))

    # Optional: Add a small fixed buffer if needed, instead of using max()
    # fixed_buffer = 2 # Example: Add 2 units always
    # effective_padding += fixed_buffer

    # Calculate final rectangle position and dimensions
    x = min_x - effective_padding
    y = min_y - effective_padding
    width = max(0, max_x - min_x + 2 * effective_padding)
    # Adjust height calculation slightly to ensure bottom padding is consistent
    height = max(0, max_y - y + effective_padding) # Changed from max_y - y + effective_padding

    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "min_x": min_x,
        "min_y": min_y,
        "max_x": max_x,
        "max_y": max_y,
        "padding": effective_padding, # Return the actual padding used
    }


def adjust_for_parent_connection(
    boundaries: Dict, subtree_root: Node, node_coords: Dict[Node, Dict]
) -> Dict:
    """
    Adjust enclosure boundaries to connect to parent node.

    Args:
        boundaries: Current enclosure boundaries
        subtree_root: Root node of the subtree
        node_coords: Dictionary mapping nodes to their coordinates

    Returns:
        Updated boundaries dictionary
    """
    x = boundaries["x"]
    y = boundaries["y"]
    width = boundaries["width"]
    height = boundaries["height"]
    effective_padding = boundaries["padding"]
    max_x = boundaries["max_x"]
    max_y = boundaries["max_y"]

    # Special case: if this subtree has a parent node outside the subtree
    # We need to extend the enclosure to include half of the branch to the parent
    # This prevents "floating" enclosures disconnected from the tree structure
    if subtree_root.parent and subtree_root.parent in node_coords:
        parent_coords = node_coords[subtree_root.parent]
        subtree_root_coords = node_coords[subtree_root]

        # Check if parent is outside our current bounds
        if (
            parent_coords["x"] < x
            or parent_coords["x"] > x + width
            or parent_coords["y"] < y
            or parent_coords["y"] > y + height
        ):

            # Calculate midpoint of the branch to parent
            mid_x = (parent_coords["x"] + subtree_root_coords["x"]) / 2
            mid_y = (parent_coords["y"] + subtree_root_coords["y"]) / 2

            # Extend enclosure to include half of the branch
            x = min(x, mid_x - effective_padding / 2)
            y = min(y, mid_y - effective_padding / 2)
            width = max(width, max_x - x + effective_padding)
            height = max(height, max_y - y + effective_padding)

    # Update boundaries with adjusted values
    boundaries.update({"x": x, "y": y, "width": width, "height": height})

    return boundaries


def calculate_corner_radius(width: float, height: float, style: Dict) -> str:
    """
    Calculate corner radius based on container size.

    Args:
        width: Width of the container
        height: Height of the container
        style: Style options for the enclosure

    Returns:
        Corner radius as a string
    """
    # Get corner radius (MD3 scales radius based on container size)
    # For print, use more subtle rounding
    if width > 300 or height > 200:
        corner_radius = 12
    elif width > 200 or height > 150:
        corner_radius = 8
    elif width > 100 or height > 80:
        corner_radius = 4
    else:
        corner_radius = 2

    # Use custom rx from style if provided
    return style.get("rx", str(corner_radius))


def create_container_group(svg_group: ET.Element, subtree_id: str) -> ET.Element:
    """
    Create an SVG group for a container.

    Args:
        svg_group: Parent SVG group
        subtree_id: ID of the subtree

    Returns:
        Created SVG group element
    """
    return ET.SubElement(
        svg_group, "g", {
            "class": "md3-container", 
            "data-subtree": subtree_id,
            "style": "pointer-events: none;"  # Make containers non-interfering with mouse events
        }
    )


def render_container_rectangle(
    container_group: ET.Element,
    boundaries: Dict,
    corner_radius: str,
    style: Dict,
    colors: Dict,
) -> None:
    """
    Render the container rectangle with appropriate styling.
    Uses highlight_color and background_opacity if present in style.
    """
    x = boundaries["x"]
    y = boundaries["y"]
    width = boundaries["width"]
    height = boundaries["height"]

    # Use highlight_color and background_opacity if present
    fill_color = style.get("highlight_color", style.get("fill", "none"))
    fill_opacity = style.get("background_opacity", style.get("fill_opacity", 0.15))
    # If highlight is not set, default to outline only
    is_filled = style.get("highlight", False) or style.get("fill", "none") != "none"

    # Get container colors - for print we use more subtle colors
    stroke_color = style.get("stroke", colors.get("highlight_enclose", "#417505"))
    stroke_width = style.get("stroke_width", DEFAULT_ENCLOSE_STROKE_WIDTH)
    stroke_dasharray = style.get("stroke_dasharray", DEFAULT_ENCLOSE_DASHARRAY)
    opacity = style.get("opacity", "0.8" if is_filled else "1.0")

    # For filled containers in print, use subtle pattern fill (optional, not for per-leaf highlight)
    # Only use pattern if not using highlight_color
    if is_filled and fill_color == "none":
        add_container_texture(container_group, x, y, width, height, corner_radius)

    # Add the main container rectangle
    ET.SubElement(
        container_group,
        "rect",
        {
            "x": f"{x:.2f}",
            "y": f"{y:.2f}",
            "width": f"{width:.2f}",
            "height": f"{height:.2f}",
            "rx": corner_radius,
            "stroke": stroke_color,
            "stroke-width": str(stroke_width),
            "stroke-dasharray": stroke_dasharray,
            "fill": fill_color,
            "fill-opacity": str(fill_opacity),
            "opacity": str(opacity),
        },
    )


def add_container_texture(
    container_group: ET.Element,
    x: float,
    y: float,
    width: float,
    height: float,
    corner_radius: str,
) -> None:
    """
    Add texture to a container using a pattern.

    Args:
        container_group: SVG group for the container
        x, y: Position of the container
        width, height: Dimensions of the container
        corner_radius: Corner radius for the container
    """
    ET.SubElement(
        container_group,
        "rect",
        {
            "x": f"{x:.2f}",
            "y": f"{y:.2f}",
            "width": f"{width:.2f}",
            "height": f"{height:.2f}",
            "rx": corner_radius,
            "fill": "url(#md3ContainerPattern)",
            "opacity": "0.1",
            "pointer-events": "none",
        },
    )


def add_container_label(
    container_group: ET.Element, boundaries: Dict, style: Dict, colors: Dict
) -> None:
    """
    Add a label to a container.

    Args:
        container_group: SVG group for the container
        boundaries: Boundaries of the container
        style: Style options for the enclosure
        colors: Color palette
    """
    container_label = style.get("label", None)
    if not container_label:
        return

    # Calculate label styling
    label_style = calculate_label_style(style, colors)

    # Calculate text position
    text_position = calculate_text_position(boundaries, style, label_style)

    # Create text element
    text_el = ET.SubElement(
        container_group,
        "text",
        {
            "x": f"{text_position['x']:.2f}",
            "y": f"{text_position['y']:.2f}",
            "font-family": label_style["font_family"],
            "font-size": label_style["font_size"],
            "font-weight": label_style["font_weight"],
            "fill": label_style["text_color"],
            "opacity": label_style["opacity"],
        },
    )
    text_el.text = container_label


def calculate_label_style(style: Dict, colors: Dict) -> Dict:
    """
    Calculate the style for a container label.

    Args:
        style: Style options for the enclosure
        colors: Color palette

    Returns:
        Dictionary of label style attributes
    """
    # Check for custom font size
    custom_font_size = style.get("font_size")
    if custom_font_size:
        font_size = str(custom_font_size)
    else:
        # MD3 uses slightly smaller text for container labels, but not too small for print
        try:
            base_font_size = float(colors.get("font_size", 11))
        except (ValueError, TypeError):
            base_font_size = 11

        font_size = str(int(base_font_size * 0.9))

    # Use text_color from style if provided
    text_color = style.get(
        "text_color", style.get("stroke", colors["highlight_enclose"])
    )
    opacity = style.get("opacity", "1.0")
    
    # Process font weight with comprehensive handling
    raw_weight = style.get("font_weight")
    if raw_weight is not None:
        # Convert text values like "bold" to their SVG equivalents
        if isinstance(raw_weight, str) and raw_weight.lower() == "bold":
            font_weight = "700"
        elif isinstance(raw_weight, str) and raw_weight.lower() == "normal":
            font_weight = "400"
        else:
            # Ensure numeric values are properly converted to strings
            font_weight = str(raw_weight)
    else:
        # Default to medium weight if not specified
        font_weight = "500"  # MD3 default is medium (500)

    return {
        "font_family": colors.get("font_family", FONT_SANS_SERIF),
        "font_size": font_size,
        "font_weight": font_weight,
        "text_color": text_color,
        "opacity": opacity,
    }


def calculate_text_position(boundaries: Dict, style: Dict, label_style: Dict) -> Dict:
    """
    Calculate the position for a container label.

    Args:
        boundaries: Boundaries of the container
        style: Style options for the enclosure
        label_style: Style options for the label

    Returns:
        Dictionary with x and y coordinates for the label
    """
    x = boundaries["x"]
    y = boundaries["y"]

    # Get custom text padding if provided, otherwise use defaults
    text_padding_left = float(style.get("text_padding_left", 8))
    text_padding_top = float(style.get("text_padding_top", 14))

    # Scale text padding with font size for better visual balance
    custom_font_size = style.get("font_size")
    if custom_font_size and float(custom_font_size) > 16:
        # Increase padding proportionally for larger fonts
        font_size_ratio = float(custom_font_size) / 16
        text_padding_left = max(text_padding_left, 8 * font_size_ratio)
        text_padding_top = max(text_padding_top, 14 * font_size_ratio)

    return {"x": x + text_padding_left, "y": y + text_padding_top}


def update_layout_for_enclosures(layout: Dict, enclose_options: Dict) -> None:
    """
    Update layout dimensions to account for enclosures.

    Args:
        layout: Layout information
        enclose_options: Options for enclosing subtrees
    """
    # Enclosures might extend beyond the basic dimensions
    # Check if any enclosure styles specify padding
    enclosure_padding = calculate_enclosure_padding(enclose_options)

    # Add enclosure padding to the effective dimensions
    if enclosure_padding > 0:
        layout["effective_width"] += enclosure_padding * 2
        layout["effective_height"] += enclosure_padding * 2


def calculate_enclosure_padding(enclose_options: Dict) -> float:
    """
    Calculate maximum padding from enclosure options.

    Args:
        enclose_options: Options for enclosing subtrees

    Returns:
        Maximum padding value
    """
    enclosure_padding = 0
    for style in enclose_options.values():
        if isinstance(style, dict):
            padding = style.get("padding", 0)
            try:
                padding = float(padding)
                enclosure_padding = max(enclosure_padding, padding)
            except (ValueError, TypeError):
                pass

    return enclosure_padding
