"""
Special visual effects for tree rendering.

This module contains functions for creating advanced visual effects for tree visualization,
including waterbrush-style edges and glowing blobs.
"""

import random
import math
import xml.etree.ElementTree as ET

# Effect Constants
WATERBRUSH_VARIATIONS = 5  # Increased from 4 for more detailed strokes
WATERBRUSH_AMPLITUDE = 2.5  # Reduced slightly from 3 for more controlled variations
WATERBRUSH_FREQUENCY = 0.4  # Slightly reduced for smoother lines
WATERBRUSH_OPACITY = (
    0.5  # Slightly increased for better visibility of branch differences
)
WATERBRUSH_BLUR = 2  # Reduced blur for sharper edges to better show length differences


# Blob effect constants
BLOB_SIZE_MIN = 10  # Minimum blob radius
BLOB_SIZE_MAX = 30  # Maximum blob radius
BLOB_BLUR = 15  # Blur amount for blobs (more reasonable)
BLOB_COUNT = 12  # Fewer, larger blobs for better effect


def find_or_create_defs(svg_element: ET.Element) -> ET.Element:
    """Find existing defs element or create a new one."""
    # Try to find existing defs
    defs = svg_element.find(".//defs")

    # If not found, try to find root SVG and add defs there
    if defs is None:
        # First check if we're already at the SVG root
        if svg_element.tag == "svg":
            root = svg_element
        else:
            # Try to find the SVG root
            root = svg_element.find("./svg")
            if root is None:
                # If no SVG root found, use the provided element
                root = svg_element

        # Look for defs in the root
        defs = root.find("./defs")
        if defs is None:
            # Create new defs element if not found
            defs = ET.SubElement(root, "defs")

    return defs


def generate_waterbrush_path(start_x, start_y, end_x, end_y):
    """
    Generate a waterbrush-style path between two points using cubic bezier curves
    similar to those used in node_renderers.py.

    Args:
        start_x, start_y: Starting coordinates
        end_x, end_y: Ending coordinates

    Returns:
        Path data string for SVG path element
    """
    # Calculate the horizontal and vertical distances for curve control
    dx = end_x - start_x
    dy = end_y - start_y

    # Add slight randomization to control points for waterbrush effect
    # Base curve tension controls how pronounced the curve is
    base_curve_tension = 0.4  # Higher values = more curved
    curve_tension = base_curve_tension + random.uniform(-0.1, 0.1)

    # Limit tension to reasonable values
    curve_tension = min(max(curve_tension, 0.2), 0.6)

    # Calculate bezier curve control points with slight randomization
    # First control point is vertically down from parent with some randomization
    cx1 = start_x + random.uniform(-WATERBRUSH_AMPLITUDE, WATERBRUSH_AMPLITUDE)
    cy1 = (
        start_y
        + (dy * curve_tension)
        + random.uniform(-WATERBRUSH_AMPLITUDE, WATERBRUSH_AMPLITUDE)
    )

    # Second control point is horizontally aligned with child with some randomization
    cx2 = end_x + random.uniform(-WATERBRUSH_AMPLITUDE, WATERBRUSH_AMPLITUDE)
    cy2 = (
        start_y
        + (dy * (1 - curve_tension))
        + random.uniform(-WATERBRUSH_AMPLITUDE, WATERBRUSH_AMPLITUDE)
    )

    # Special case for horizontal alignment to avoid weird curves
    if abs(dx) < 10:
        # If nodes are nearly aligned vertically, use a simpler curve with randomization
        cx1 = start_x + random.uniform(-WATERBRUSH_AMPLITUDE, WATERBRUSH_AMPLITUDE)
        cy1 = (
            start_y
            + (dy * 0.25)
            + random.uniform(-WATERBRUSH_AMPLITUDE / 2, WATERBRUSH_AMPLITUDE / 2)
        )
        cx2 = start_x + random.uniform(-WATERBRUSH_AMPLITUDE, WATERBRUSH_AMPLITUDE)
        cy2 = (
            start_y
            + (dy * 0.75)
            + random.uniform(-WATERBRUSH_AMPLITUDE / 2, WATERBRUSH_AMPLITUDE / 2)
        )

    # Create path using cubic bezier curve (C command) with the randomized control points
    path_data = (
        f"M {start_x:.2f},{start_y:.2f} "  # Start at parent node
        f"C {cx1:.2f},{cy1:.2f} {cx2:.2f},{cy2:.2f} {end_x:.2f},{end_y:.2f}"  # Cubic bezier to child
    )

    return path_data


def render_waterbrush_edge(
    svg_group,
    start_x,
    start_y,
    end_x,
    end_y,
    color,
    stroke_width,
    opacity,
    horizontal=False,
):
    """
    Render an edge with waterbrush effect using spline-based paths.

    Args:
        svg_group: SVG group element to render to
        start_x, start_y: Starting coordinates
        end_x, end_y: Ending coordinates
        color: Stroke color
        stroke_width: Width of the stroke
        opacity: Base opacity
        horizontal: Parameter kept for backward compatibility but no longer used
    """
    # Create a group for this edge
    edge_group_id = f"waterbrush_{random.randint(1000, 9999)}"
    edge_group = ET.SubElement(svg_group, "g", {"id": edge_group_id})

    # Create multiple path variations for water brush effect
    base_stroke_width = float(stroke_width)

    # First draw a wider stroke with lower opacity as a base "glow" effect
    glow_path_data = generate_waterbrush_path(start_x, start_y, end_x, end_y)
    glow_style = {
        "stroke": color,
        "stroke-width": str(base_stroke_width * 2.0),
        "stroke-opacity": str(opacity * 0.25),
        "fill": "none",
        "filter": "url(#waterbrushBlur)",
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
    }
    ET.SubElement(edge_group, "path", {**{"d": glow_path_data}, **glow_style})

    # Create multiple varied paths for the waterbrush effect
    for i in range(WATERBRUSH_VARIATIONS):
        # Generate unique path data for each variation
        path_data = generate_waterbrush_path(start_x, start_y, end_x, end_y)

        # Vary opacity and width slightly for each path to create texture
        path_opacity = opacity * random.uniform(0.7, 1.0)
        path_width = base_stroke_width * random.uniform(0.5, 1.1)

        # Create path with water brush styling
        path_style = {
            "stroke": color,
            "stroke-width": str(path_width),
            "stroke-opacity": str(path_opacity),
            "fill": "none",
            "filter": "url(#waterbrushBlur)",
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
        }

        ET.SubElement(edge_group, "path", {**{"d": path_data}, **path_style})


def render_blob_edge(
    svg_group: ET.Element,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    color: str,
    opacity: float = 1.0,
) -> None:
    """
    Render a single glowing blob in the middle of an edge with a gradient
    that becomes less dense toward the edges.

    Args:
        svg_group: SVG group element to render to
        start_x, start_y: Starting coordinates
        end_x, end_y: Ending coordinates
        color: Base color for the blob
        opacity: Base opacity (0.0 to 1.0)
    """
    # Ensure opacity is a float (handle string inputs)
    opacity = float(opacity) if isinstance(opacity, str) else opacity

    # Create a group for this edge
    edge_group_id = f"blobEdge_{random.randint(1000, 9999)}"
    edge_group = ET.SubElement(svg_group, "g", {"id": edge_group_id})

    # For tree branches with an "L" shape, we need to find the corner point
    has_corner = abs(end_x - start_x) > 5 and abs(end_y - start_y) > 5

    if has_corner:
        # For L-shaped tree branches, place the blob at the corner
        corner_x = start_x
        corner_y = end_y

        # Calculate total path length for scaling
        distance_v = abs(corner_y - start_y)  # Vertical segment
        distance_h = abs(end_x - corner_x)  # Horizontal segment
        total_distance = distance_v + distance_h

        # Size the blob based on the total distance
        blob_size = min(80, max(30, total_distance * 0.4))

        # Create a single large blob at the corner
        create_centered_blob(edge_group, corner_x, start_y, blob_size, color, opacity)
    else:
        # For straight edges, place the blob at the midpoint
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2

        # Calculate distance for proper scaling
        dx = end_x - start_x
        dy = end_y - start_y
        distance = math.sqrt(dx * dx + dy * dy)

        # Size the blob based on the distance
        blob_size = min(80, max(30, distance * 0.6))

        # Create a single large blob at the midpoint
        create_centered_blob(edge_group, mid_x, mid_y, blob_size, color, opacity)


def create_centered_blob(
    parent: ET.Element,
    x: float,
    y: float,
    radius: float,
    color: str,
    opacity: float = 1.0,
) -> None:
    """
    Create a single glowing blob with highest density in the center,
    gradually fading out toward the edges.

    Args:
        parent: Parent SVG element
        x, y: Center coordinates
        radius: Radius of the blob
        color: Base color for the blob
        opacity: Base opacity (0.0 to 1.0)
    """
    # Create unique ID for this gradient
    gradient_id = f"blobGradient_{random.randint(1000, 9999)}"

    # Add radial gradient definition to defs
    defs = find_or_create_defs(parent)
    gradient = ET.SubElement(
        defs,
        "radialGradient",
        {
            "id": gradient_id,
            "cx": "50%",
            "cy": "50%",
            "r": "50%",
            "fx": "50%",
            "fy": "50%",
            "spreadMethod": "pad",
        },
    )

    # Add more gradient stops for smoother transition from dense center to transparent edge
    ET.SubElement(
        gradient,
        "stop",
        {"offset": "0%", "stop-color": color, "stop-opacity": str(opacity)},
    )
    ET.SubElement(
        gradient,
        "stop",
        {"offset": "40%", "stop-color": color, "stop-opacity": str(opacity * 0.9)},
    )
    ET.SubElement(
        gradient,
        "stop",
        {"offset": "70%", "stop-color": color, "stop-opacity": str(opacity * 0.5)},
    )
    ET.SubElement(
        gradient,
        "stop",
        {"offset": "85%", "stop-color": color, "stop-opacity": str(opacity * 0.2)},
    )
    ET.SubElement(
        gradient, "stop", {"offset": "100%", "stop-color": color, "stop-opacity": "0"}
    )

    # Create the blob circle
    ET.SubElement(
        parent,
        "circle",
        {
            "cx": str(x),
            "cy": str(y),
            "r": str(radius),
            "fill": f"url(#{gradient_id})",
            "filter": "url(#blobBlur)",
        },
    )
