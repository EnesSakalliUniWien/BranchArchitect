"""
SVG utility functions for creating and manipulating SVG elements.

This module provides helper functions for working with SVG elements,
including creating SVG roots, converting to strings, and color utilities.
"""

import xml.etree.ElementTree as ET
from typing import Tuple

# SVG filter constants
WATERBRUSH_BLUR = 3
BLOB_BLUR = 15

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert hex color string to RGB tuple.
    
    Args:
        hex_color: Hex color string (with or without leading #)
        
    Returns:
        Tuple of (r, g, b) values (0-255)
    """
    # Remove hash symbol if present
    hex_color = hex_color.lstrip("#")

    # Handle both 3-digit and 6-digit hex
    if len(hex_color) == 3:
        r = int(hex_color[0] + hex_color[0], 16)
        g = int(hex_color[1] + hex_color[1], 16)
        b = int(hex_color[2] + hex_color[2], 16)
    else:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

    return (r, g, b)


def create_svg_root(width: float, height: float) -> ET.Element:
    """
    Create the root SVG element with the specified dimensions.
    
    Args:
        width: Width of the SVG in pixels
        height: Height of the SVG in pixels
        
    Returns:
        The root SVG element
    """
    width = max(width, 100)
    height = max(height, 100)

    attrs = {
        "width": str(width),
        "height": str(height),
        "viewBox": f"0 0 {width} {height}",
        "version": "1.1",
        "xmlns": "http://www.w3.org/2000/svg",
        "xmlns:xlink": "http://www.w3.org/1999/xlink",
    }
    root = ET.Element("svg", attrs)

    # Add defs section with filters
    defs = ET.SubElement(root, "defs")

    # Add waterbrush blur filter
    filter_el = ET.SubElement(
        defs,
        "filter",
        {
            "id": "waterbrushBlur",
            "x": "-20%",
            "y": "-20%",
            "width": "140%",
            "height": "140%",
        },
    )
    ET.SubElement(
        filter_el,
        "feGaussianBlur",
        {"in": "SourceGraphic", "stdDeviation": str(WATERBRUSH_BLUR)},
    )

    # Add blob blur filter (stronger blur)
    blob_filter = ET.SubElement(
        defs,
        "filter",
        {
            "id": "blobBlur",
            "x": "-50%",
            "y": "-50%",
            "width": "200%",
            "height": "200%",
        },
    )
    ET.SubElement(
        blob_filter,
        "feGaussianBlur",
        {"in": "SourceGraphic", "stdDeviation": str(BLOB_BLUR)},
    )

    return root


def svg_to_string(svg_root: ET.Element) -> str:
    """
    Convert an SVG element to a string.
    
    Args:
        svg_root: The root SVG element
        
    Returns:
        String representation of the SVG
    """
    try:
        return ET.tostring(svg_root, encoding="unicode")
    except Exception as e:
        print(f"Error serializing SVG: {e}")
        return f"<svg><text fill='red'>Error creating SVG: {e}</text></svg>"


def setup_svg_root(svg_width, svg_height, colors, enable_latex, use_mathjax):
    """
    Set up the SVG root element with specified parameters.
    
    Args:
        svg_width: Width of the SVG
        svg_height: Height of the SVG
        colors: Color configuration
        enable_latex: Whether LaTeX is enabled
        use_mathjax: Whether MathJax is used
    
    Returns:
        The root SVG element
    """
    # Implementation details here
    pass


def add_background_to_svg(svg_root, colors):
    """
    Add a background to the SVG root element.
    
    Args:
        svg_root: The root SVG element
        colors: Color configuration
    
    Returns:
        None
    """
    # Implementation details here
    pass


def create_container_group(svg_root, margin_left, margin_top):
    """
    Create a container group within the SVG root element.
    
    Args:
        svg_root: The root SVG element
        margin_left: Left margin
        margin_top: Top margin
    
    Returns:
        The container group element
    """
    # Implementation details here
    pass