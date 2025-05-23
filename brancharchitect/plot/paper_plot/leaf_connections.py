import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional, Any
from svgpathtools import Path, Line, QuadraticBezier

"""
Leaf connection utilities for BranchArchitect using svgpath2mpath and svgpathtools.
"""


def render_leaf_connections(
    container: ET.Element,
    layouts: List[Dict],
    leaf_connections: List[Dict[str, Any]],
    inter_tree_paddings: Optional[List[float]] = None,
) -> None:
    """
    Render arrows connecting leaf labels between different trees using advanced SVG path libraries.

    Args:
        container: Parent SVG element
        layouts: List of layout information dictionaries
        leaf_connections: List of connection specifications
        inter_tree_paddings: Optional list of paddings between trees
    """
    if not leaf_connections:
        return

    # Calculate tree positions
    tree_positions = []
    curr_pos = 0
    for i, layout in enumerate(layouts):
        tree_positions.append(curr_pos)
        if inter_tree_paddings and i < len(inter_tree_paddings):
            curr_pos += layout["width"] + inter_tree_paddings[i]
        else:
            # Use constant from paper_plot_constants.py
            curr_pos += layout["width"] + 60  # INTER_TREE_SPACING assumed to be 60

    # Create connections group
    connections_group = ET.SubElement(container, "g", {"class": "leaf-connections"})

    # Create defs section for arrowheads if it doesn't exist
    defs = container.find(".//defs")
    if defs is None:
        defs = ET.SubElement(container, "defs")

    # Create advanced arrowhead marker
    create_advanced_arrowhead(defs)

    # Draw each connection using advanced path tools
    for conn_idx, conn in enumerate(leaf_connections):
        source_tree_idx = conn["source_tree"]
        target_tree_idx = conn["target_tree"]
        source_leaf = conn["source_leaf"]
        target_leaf = conn["target_leaf"]
        style = conn.get("style", {})

        if source_tree_idx >= len(layouts) or target_tree_idx >= len(layouts):
            print(f"Warning: Tree index out of range in leaf connection")
            continue

        # Find leaf coordinates
        source_coords, target_coords = get_leaf_coordinates(
            layouts,
            tree_positions,
            source_tree_idx,
            target_tree_idx,
            source_leaf,
            target_leaf,
        )

        if source_coords is None or target_coords is None:
            print(
                f"Warning: Could not find leaf '{source_leaf}' or '{target_leaf}' for connection"
            )
            continue

        # Create advanced path with proper arrow connection
        create_advanced_path_connection(
            connections_group, source_coords, target_coords, style, conn_idx
        )


def create_advanced_arrowhead(defs: ET.Element) -> None:
    """
    Create a reusable advanced arrowhead marker.
    """
    # Create a sophisticated arrowhead marker
    marker_attrs = {
        "id": "advanced-arrowhead",
        "viewBox": "0 0 10 10",
        "refX": "1",  # Reference point at the tip
        "refY": "5",
        "markerWidth": "6",
        "markerHeight": "6",
        "orient": "auto",
        "markerUnits": "strokeWidth",
    }
    marker = ET.SubElement(defs, "marker", marker_attrs)

    # Create arrowhead path
    arrow_path = ET.SubElement(
        marker,
        "path",
        {
            "d": "M 0,0 L 10,5 L 0,10 z",  # Triangle arrowhead
            "fill": "context-stroke",  # Use the same color as the stroke
        },
    )


def get_leaf_coordinates(
    layouts: List[Dict],
    tree_positions: List[float],
    source_tree_idx: int,
    target_tree_idx: int,
    source_leaf: str,
    target_leaf: str,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """
    Get coordinates for source and target leaf labels.
    """
    source_layout = layouts[source_tree_idx]
    target_layout = layouts[target_tree_idx]

    # Find leaf coordinates
    source_coords = find_leaf_label_coords(
        source_layout, source_leaf, tree_positions[source_tree_idx]
    )
    target_coords = find_leaf_label_coords(
        target_layout, target_leaf, tree_positions[target_tree_idx]
    )

    return source_coords, target_coords


def find_leaf_label_coords(
    layout: Dict, leaf_name: str, tree_offset: float
) -> Optional[Tuple[float, float]]:
    """
    Find precise coordinates of a leaf label endpoint.
    """
    for node, coords in layout["coords"].items():
        node_label = getattr(node, "label", None) or getattr(node, "name", None)

        if node_label == leaf_name:
            leaf_x = coords["x"]
            leaf_y = coords["y"]

            # Add leaf label offset and tree position offset
            label_x = leaf_x + layout.get("leaf_label_offset", 8)

            # Calculate text width based on leaf name and font size
            leaf_font_size = layout.get("leaf_font_size", 20)
            text_width = len(leaf_name) * (leaf_font_size * 0.55)

            # Return position after text for source, before text for target
            # We'll determine which is which in the path creation
            return (tree_offset + label_x, leaf_y, text_width)

    return None


def create_advanced_path_connection(
    container: ET.Element,
    source_coords: Tuple[float, float, float],
    target_coords: Tuple[float, float, float],
    style: Dict,
    conn_idx: int,
) -> None:
    """
    Create an advanced SVG path connection between leaves.
    """
    source_x, source_y, source_width = source_coords
    target_x, target_y, target_width = target_coords

    # Position source after the text, target before the text
    source_point = complex(source_x + source_width + 5, source_y)
    target_point = complex(target_x - 5, target_y)

    # Create path based on style
    path_type = style.get("path-type", "curve")

    if path_type == "straight":
        path = Path(Line(source_point, target_point))

    elif path_type == "angled":
        mid_x = (source_point.real + target_point.real) / 2
        path = Path(
            Line(source_point, complex(mid_x, source_y)),
            Line(complex(mid_x, source_y), complex(mid_x, target_y)),
            Line(complex(mid_x, target_y), target_point),
        )

    else:  # curve
        # Create a nice quadratic bezier curve
        curve_factor = float(style.get("curve-factor", 0.3))
        control_x = (source_point.real + target_point.real) / 2
        control_y = ((source_y + target_y) / 2) - (
            abs(target_point.real - source_point.real) * curve_factor
        )
        control_point = complex(control_x, control_y)

        path = Path(QuadraticBezier(source_point, control_point, target_point))

    # Create SVG path element with proper styling
    path_attrs = {
        "d": path.d(),
        "fill": "none",
        "stroke": style.get("stroke", "#555555"),
        "stroke-width": str(style.get("stroke-width", 2.0)),
        "marker-end": "url(#advanced-arrowhead)",  # Use our advanced arrowhead
        "class": f"leaf-connection-path connection-{conn_idx}",
    }

    # Add path to container
    ET.SubElement(container, "path", path_attrs)
