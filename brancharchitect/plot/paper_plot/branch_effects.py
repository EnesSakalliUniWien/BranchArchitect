"""
Special visual effects for tree rendering.

This module contains functions for creating advanced visual effects for tree visualization,
including waterbrush-style edges and glowing blobs.
"""
import xml.etree.ElementTree as ET
from typing import Dict, Optional, Tuple, List
import uuid


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

def create_line_style(
    color: str, stroke_width: str, opacity: str, dash_pattern: str
) -> Dict:
    """Creates the basic style attribute dictionary for an SVG path."""
    style = {
        "stroke": color,
        "stroke-width": stroke_width,
        "stroke-opacity": opacity,
        "fill": "none",
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
    }
    if dash_pattern != "none":
        style["stroke-dasharray"] = dash_pattern
    return style

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


def render_waterbrush_branch(
    branch_group: ET.Element,
    x: float,
    y: float,
    child_x: float,
    child_y: float,
    color: str,
    stroke_width: str,
    opacity: str,
    gradient_end: Optional[str] = None,
    control_points: Optional[List[Tuple[float, float]]] = None,
    path_data: Optional[str] = None,
) -> None:
    """PDF-optimized: Render as a normal or gradient line (no filter)."""
    style = {
        "stroke": color,
        "stroke-width": stroke_width,
        "stroke-opacity": opacity,
        "fill": "none",
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
    }
    # Optionally add a gradient if requested
    if gradient_end:
        defs = find_or_create_defs(branch_group)
        gradient_id = f"branchGradient_{uuid.uuid4().hex[:8]}"
        is_horizontal = abs(child_x - x) > abs(child_y - y)
        x1, y1, x2, y2 = (
            ("0%", "0%", "100%", "0%") if is_horizontal else ("0%", "0%", "0%", "100%")
        )
        gradient = ET.SubElement(
            defs,
            "linearGradient",
            {"id": gradient_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2},
        )
        ET.SubElement(gradient, "stop", {"offset": "0%", "stop-color": color})
        ET.SubElement(gradient, "stop", {"offset": "100%", "stop-color": gradient_end})
        style["stroke"] = f"url(#{gradient_id})"
    ET.SubElement(branch_group, "path", {**{"d": path_data}, **style})

def apply_cut_edge_styling(
    branch_group: ET.Element,
    line_style: Dict,
    x: float,
    y: float,
    child_x: float,
    child_y: float,
    colors: Dict,
) -> None:
    """Applies visual style for cut edges (color, dash, marker). Modifies line_style."""
    cut_color = colors.get("cut_edge_color", "#B4261E")  # Default Red
    line_style["stroke"] = cut_color
    line_style["stroke-dasharray"] = "5,3"
    line_style["stroke-opacity"] = "1.0"  # Ensure visibility

    # Add tangential cut line marker
    mid_x, mid_y = (x + child_x) / 2, (y + child_y) / 2
    dx, dy = child_x - x, child_y - y
    length = (dx**2 + dy**2) ** 0.5
    if length > 1e-6:
        perp_dx, perp_dy = -dy / length, dx / length
        base_width = float(line_style.get("stroke-width", "1.5"))
        cut_len = max(2.0, base_width * 1.5)
        cut_start_x, cut_start_y = mid_x + perp_dx * cut_len, mid_y + perp_dy * cut_len
        cut_end_x, cut_end_y = mid_x - perp_dx * cut_len, mid_y - perp_dy * cut_len
        cut_style = {
            "stroke": colors.get("cut_line_color", cut_color),
            "stroke-width": str(max(1.0, base_width * 0.75)),
            "stroke-linecap": "round",
            "stroke-opacity": line_style["stroke-opacity"],
        }
        ET.SubElement(
            branch_group,
            "line",
            {
                "x1": f"{cut_start_x:.2f}",
                "y1": f"{cut_start_y:.2f}",
                "x2": f"{cut_end_x:.2f}",
                "y2": f"{cut_end_y:.2f}",
                **cut_style,
            },
        )


def apply_elevation_effect(
    branch_group: ET.Element,
    path_data: str,
    line_style: Dict,  # Style for the main line (might include gradient URL)
    base_stroke_width: str,  # Base width (numeric string) for calculations
    base_opacity: str,  # Base opacity (numeric string) for calculations
    base_color: str,  # Base color used for highlighting (e.g., glow, or shadow base)
) -> None:
    """
    Applies a PDF-compatible elevation effect using vector elements:
    1. A dark, offset path for a subtle shadow.
    2. A wider, semi-transparent path in the base color for a soft glow.
    3. The main path drawn on top.
    """
    try:
        width_num = float(base_stroke_width)
        opacity_num = float(base_opacity)
    except ValueError:
        print(
            f"Warning: Invalid base stroke width ('{base_stroke_width}') or opacity ('{base_opacity}') for elevation."
        )
        width_num, opacity_num = 1.5, 1.0  # Fallback defaults

    # --- 1. Simulated Drop Shadow (Offset Path) ---
    # Calculate offset based slightly on stroke width for scaling effect
    shadow_offset_x = max(0.5, width_num * 0.2)
    shadow_offset_y = max(0.75, width_num * 0.3)
    shadow_opacity = 0.3  # Adjust opacity as needed (0.2 - 0.5 range)
    shadow_color = "#333333"  # Dark grey shadow color

    shadow_style = {
        "stroke": shadow_color,
        "stroke-width": base_stroke_width,  # Same width as main line for simple shadow
        "stroke-opacity": str(shadow_opacity),
        "fill": "none",
        "stroke-linecap": "round",  # Match main style
        "stroke-linejoin": "round",
    }
    # Add the path with a transform attribute for the offset
    ET.SubElement(
        branch_group,
        "path",
        {
            "d": path_data,
            "transform": f"translate({shadow_offset_x:.2f}, {shadow_offset_y:.2f})",
            **shadow_style,
        },
    )

    # --- 2. Refined Glow Path (Wider, Semi-Transparent Base Color) ---
    glow_width_factor = (
        2.5  # How much wider the glow is than the main line (e.g., 2.5x)
    )
    glow_opacity_factor = 0.20  # Opacity relative to base_opacity (e.g., 20%)

    glow_style = {
        "stroke": base_color,  # Glow uses the base highlight color
        "stroke-width": str(width_num * glow_width_factor),
        "stroke-opacity": str(opacity_num * glow_opacity_factor),
        "fill": "none",
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
    }
    ET.SubElement(branch_group, "path", {**{"d": path_data}, **glow_style})

    # --- 3. Main Path (On Top) ---
    # Ensure no filter is applied for PDF compatibility
    main_style = line_style.copy()
    if "filter" in main_style:
        del main_style["filter"]

    ET.SubElement(branch_group, "path", {**{"d": path_data}, **main_style})


def add_highlight_pattern(
    branch_group: ET.Element, path_data: str, stroke_width: str
) -> None:
    """Adds a pattern overlay (requires pattern definition in <defs>)."""
    try:
        width_num = float(stroke_width)
    except ValueError:
        width_num = 1.5
    pattern_style = {
        "stroke": "url(#highlightPattern)",  # Assumes definition exists
        "stroke-width": str(width_num * 1.5),
        "stroke-opacity": "0.6",
        "fill": "none",
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
    }
    ET.SubElement(branch_group, "path", {**{"d": path_data}, **pattern_style})
    
    
def render_blob_branch(
    branch_group: ET.Element,
    x: float,
    y: float,
    child_x: float,
    child_y: float,
    blob_fill: str,
    blob_opacity: str,
    control_points: Optional[List[Tuple[float, float]]] = None,
    path_data: Optional[str] = None,
) -> None:
    """PDF-optimized: Render as a thick, semi-transparent line (no filter)."""
    try:
        float_opacity = float(blob_opacity)
    except ValueError:
        float_opacity = 0.35
    style = {
        "stroke": blob_fill,
        "stroke-width": "8.0",
        "stroke-opacity": str(float_opacity),
        "fill": "none",
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
    }
    ET.SubElement(branch_group, "path", {**{"d": path_data}, **style})
