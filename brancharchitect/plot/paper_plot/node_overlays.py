"""
Overlay label utilities for BranchArchitect.

This module contains functions for rendering overlay labels, footers, and captions.
"""

import xml.etree.ElementTree as ET
import re
from typing import Dict, Any, List
from brancharchitect.plot.paper_plot.paper_plot_constants import (
    FONT_SANS_SERIF,
)
from brancharchitect.plot.paper_plot.folded_latex2svg.folded_latex2svg import (
    latex2svg_modular,
)


def render_direct_latex_svg(
    footer_group: ET.Element,
    line: str,
    x: float,
    y: float,
    font_size_num: float,
    font_color: str,
    font_family: str,
    line_height: float,
    latex_scale: float,
) -> float:
    """
    Renders a line of text as SVG using latex2svg_modular.

    Args:
        footer_group: The footer group element.
        line: Text line to render.
        x: X-coordinate for the text.
        y: Y-coordinate for the text.
        font_size_num: Font size.
        font_color: Font color.
        font_family: Font family.
        line_height: Line height.
        latex_scale: Scaling factor for LaTeX elements.

    Returns:
        Updated y-coordinate after rendering.
    """
    try:
        # Use latex2svg_modular instead of folded_latex2svg
        svg_dict = latex2svg_modular(
            line,
            params={
                "fontsize": int(font_size_num),
            },
        )

        svg_content = svg_dict["svg"]

        # Parse the SVG snippet
        # Remove XML declarations and DOCTYPE
        svg_content = re.sub(r"<\?xml.*?\?>", "", svg_content)
        svg_content = re.sub(r"<!DOCTYPE.*?>", "", svg_content)

        # Extract the width, height, and viewBox from the SVG
        latex_svg_root = ET.fromstring(svg_content)

        # Get dimensions from the SVG dict
        latex_width = float(svg_dict["width"]) * latex_scale
        latex_height = float(svg_dict["height"]) * latex_scale

        # Calculate centered position
        latex_x = x - (latex_width / 2)
        latex_y = y

        # Create a group to hold and properly scale/position the LaTeX SVG
        latex_group = ET.SubElement(
            footer_group,
            "g",
            {
                "transform": f"translate({latex_x:.2f}, {latex_y:.2f}) scale({latex_scale})",
                "class": "latex-svg-group",
            },
        )

        # Extract and append each child from the SVG
        for child in latex_svg_root:
            # Skip the defs section if present
            if child.tag == "{http://www.w3.org/2000/svg}defs":
                continue

            # Apply the theme's text color to all paths and text
            if child.tag in [
                "{http://www.w3.org/2000/svg}path",
                "{http://www.w3.org/2000/svg}text",
            ]:
                child.set("fill", font_color)

            # If it's a group, apply fill to all descendants
            if child.tag == "{http://www.w3.org/2000/svg}g":
                for path in child.findall(".//{http://www.w3.org/2000/svg}path"):
                    path.set("fill", font_color)
                for text in child.findall(".//{http://www.w3.org/2000/svg}text"):
                    text.set("fill", font_color)

            # Add the child to our group
            latex_group.append(child)

        # Return updated y position
        return y + latex_height + (line_height * 0.5)  # Add appropriate spacing

    except Exception as e:
        # Fallback to plain text rendering
        print(f"Warning: Failed to render LaTeX: '{line}' - {e}. Falling back to text.")
        return render_plain_text_line(
            footer_group,
            line,
            x,
            y,
            font_size_num,
            font_family,
            font_color,
            line_height,
        )


def process_latex_line(
    footer_group: ET.Element,
    line: str,
    x: float,
    y: float,
    font_size_num: float,
    font_family: str,
    font_color: str,
    line_height: float,
    latex_scale: float,
) -> float:
    """
    Processes a line that might contain LaTeX.

    Args:
        footer_group: The footer group element.
        line: Text line to process.
        x: X-coordinate for the text.
        y: Y-coordinate for the text.
        font_size_num: Font size.
        font_family: Font family.
        font_color: Font color.
        line_height: Line height.
        latex_scale: Scaling factor for LaTeX elements.

    Returns:
        Updated y-coordinate after processing.
    """
    # Skip empty lines but advance y
    if not line.strip():
        return y + line_height

    # Basic check for LaTeX delimiters
    if "$" in line and line.count("$") >= 2:
        return render_direct_latex_svg(
            footer_group,
            line,
            x,
            y,
            font_size_num,
            font_color,
            font_family,
            line_height,
            latex_scale,
        )
    else:
        # Render plain text line
        return render_plain_text_line(
            footer_group,
            line,
            x,
            y,
            font_size_num,
            font_family,
            font_color,
            line_height,
        )


def render_plain_text_line(
    footer_group: ET.Element,
    line: str,
    x: float,
    y: float,
    font_size_num: float,
    font_family: str,
    font_color: str,
    line_height: float,
) -> float:
    """
    Renders a line of plain text.

    Args:
        footer_group: The footer group element.
        line: Text line to render.
        x: X-coordinate for the text.
        y: Y-coordinate for the text.
        font_size_num: Font size.
        font_family: Font family.
        font_color: Font color.
        line_height: Line height.

    Returns:
        Updated y-coordinate after rendering.
    """
    text_element = ET.SubElement(
        footer_group,
        "text",
        {
            "x": str(x),
            "y": str(y + font_size_num / 2),
            "font-family": str(font_family),
            "font-size": str(font_size_num),
            "text-anchor": "middle",
            "fill": str(font_color),
            "dominant-baseline": "central",
        },
    )
    text_element.text = line
    return y + line_height


def render_plain_text_content(
    footer_group: ET.Element,
    lines: List[str],
    x: float,
    y: float,
    font_size_num: float,
    font_family: str,
    font_color: str,
    line_height: float,
) -> float:
    """
    Renders content as plain text.

    Args:
        footer_group: The footer group element.
        lines: List of text lines to render.
        x: X-coordinate for the text.
        y: Y-coordinate for the text.
        font_size_num: Font size.
        font_family: Font family.
        font_color: Font color.
        line_height: Line height.

    Returns:
        Updated y-coordinate after rendering all lines.
    """
    for line in lines:
        y = render_plain_text_line(
            footer_group,
            line.strip(),
            x,
            y,
            font_size_num,
            font_family,
            font_color,
            line_height,
        )
    return y


def escape_xml(text: str) -> str:
    """
    Escape special characters in XML text, except for those potentially used by LaTeX/MathJax.
    This is a simplified approach; robust handling might require more context.
    """
    # Avoid escaping characters commonly used in LaTeX math mode
    text = str(text)
    text = text.replace("&", "&amp;")  # Must be first
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    # Do not escape \, {, }, $, ^, _ if LaTeX might be used.
    # Quotes might still need escaping depending on context.
    text = text.replace('"', "&quot;")
    text = text.replace("'", "&apos;")
    return text


def render_node_labels(
    svg_group: ET.Element,
    node_coords: Dict,
    colors: Dict,
    node_labels: Dict[str, Dict[str, Any]],
) -> None:
    """
    Render labels for specific nodes with custom positioning and styling.

    Args:
        svg_group: SVG group element to render to
        node_coords: Dictionary mapping nodes to their coordinates
        colors: Color palette
        node_labels: Dictionary mapping node IDs to label information
    """
    if not node_labels:
        return

    # Create a group for all node labels
    labels_group = ET.SubElement(svg_group, "g", {"class": "md3-node-labels"})

    # Process each node label
    for node_id, label_info in node_labels.items():
        # Find the node coordinates
        node_coord = None
        for node, coords in node_coords.items():
            if coords.get("id") == node_id:
                node_coord = coords
                break

        if not node_coord:
            continue

        # Extract node position
        node_x = node_coord["x"]
        node_y = node_coord["y"]

        # Extract label properties with defaults
        text = label_info.get("text", "")
        if not text:
            continue

        position = label_info.get("position", "right")
        offset = float(label_info.get("offset", 20))
        highlight = label_info.get("highlight", False)
        highlight_color = label_info.get(
            "highlight_color", colors.get("highlight_label", "#6750A4")
        )
        background_opacity = label_info.get("background_opacity", "0.1")
        if not isinstance(background_opacity, str):
            background_opacity = str(background_opacity)

        # Font styling
        try:
            font_size = float(
                label_info.get("font_size", colors.get("font_size", "12"))
            )
        except (ValueError, TypeError):
            font_size = 12

        font_weight = label_info.get("font_weight", "400")

        # Calculate label position based on specified position
        if position == "right":
            label_x = node_x + offset
            label_y = node_y
            anchor = "start"
        elif position == "left":
            label_x = node_x - offset
            label_y = node_y
            anchor = "end"
        elif position == "above":
            label_x = node_x
            label_y = node_y - offset
            anchor = "middle"
        elif position == "below":
            label_x = node_x
            label_y = node_y + offset
            anchor = "middle"
        else:
            # Default to right if position is not recognized
            label_x = node_x + offset
            label_y = node_y
            anchor = "start"

        # Create a group for this label
        label_group = ET.SubElement(
            labels_group, "g", {"class": "md3-node-label", "data-node-id": node_id}
        )

        # If highlighted, add a background
        if highlight:
            # Calculate background dimensions based on text length and font size
            # This is an approximation
            text_width = len(text) * font_size * 0.6
            bg_width = text_width + 10  # Add padding
            bg_height = font_size * 1.5

            # Adjust background position based on text anchor
            if anchor == "start":
                bg_x = label_x - 5
            elif anchor == "end":
                bg_x = label_x - bg_width + 5
            else:  # middle
                bg_x = label_x - (bg_width / 2)

            bg_y = label_y - (bg_height / 2)

            # Add background rectangle
            ET.SubElement(
                label_group,
                "rect",
                {
                    "x": str(f"{bg_x:.2f}"),
                    "y": str(f"{bg_y:.2f}"),
                    "width": str(f"{bg_width:.2f}"),
                    "height": str(f"{bg_height:.2f}"),
                    "rx": str(f"{3}"),
                    "fill": str(highlight_color),
                    "fill-opacity": str(background_opacity),
                    "stroke": "none",
                },
            )

        # Add the text element
        text_el = ET.SubElement(
            label_group,
            "text",
            {
                "x": str(f"{label_x:.2f}"),
                "y": str(f"{label_y:.2f}"),
                "font-family": str(colors.get("font_family", FONT_SANS_SERIF)),
                "font-size": str(font_size),
                "font-weight": str(font_weight),
                "text-anchor": str(anchor),
                "dominant-baseline": "central",
                "fill": str(highlight_color if highlight else colors["base_text"]),
            },
        )
        text_el.text = text