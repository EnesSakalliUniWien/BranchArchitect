"""
Overlay label utilities for BranchArchitect.

This module contains functions for rendering overlay labels, footers, and captions.
"""

import xml.etree.ElementTree as ET
import re
from typing import Dict, Any, Optional, Tuple, List

from brancharchitect.plot.paper_plot.paper_plot_constants import (
    FOOTER_PADDING,
    FOOTER_FONT_SIZE_ADD,
    FONT_SANS_SERIF,
    MATHJAX_SCRIPT,
    MATHJAX_SRC
)

from brancharchitect.plot.paper_plot.folded_latex2svg.folded_latex2svg import (
    latex2svg_modular,
)


def create_footer_group(parent_svg_element: ET.Element) -> ET.Element:
    """
    Creates and returns a group element for the footer.

    Args:
        parent_svg_element: The SVG group element for the specific tree.

    Returns:
        The created footer group element.
    """
    return ET.SubElement(parent_svg_element, "g", {"class": "md3-footer"})


def calculate_footer_position(
    tree_height: float, tree_width: float
) -> Tuple[float, float]:
    """
    Calculates the starting position for the footer text.

    Args:
        tree_height: The calculated height of the tree drawing area.
        tree_width: The calculated width of the tree drawing area.

    Returns:
        Tuple containing x and y coordinates for the footer.
    """
    x = tree_width / 2  # Center horizontally
    y = tree_height + FOOTER_PADDING  # Position below the tree
    return x, y


def get_font_properties(
    colors: Dict, footer_font_size: Optional[float] = None
) -> Tuple[float, str, str, float]:
    """
    Extracts font properties from the colors dictionary.

    Args:
        colors: Dictionary containing color and style information.
        footer_font_size: Custom font size for footer text. If provided, overrides the default size.

    Returns:
        Tuple containing font size, font family, font color, and line height.
    """
    try:
        font_size_num = (
            footer_font_size
            if footer_font_size is not None
            else float(colors.get("font_size", 10)) + FOOTER_FONT_SIZE_ADD
        )
    except ValueError:
        font_size_num = 10.0

    font_family = colors.get(
        "font_family", "'Roboto', 'Arial', 'Helvetica', sans-serif"
    )
    font_color = colors.get("text_color", "#1A1A1A")
    line_height = font_size_num * 1.2  # Standard line height factor

    return font_size_num, font_family, font_color, line_height


def add_mathjax_script(parent_svg_element: ET.Element) -> None:
    """
    Adds MathJax script to the parent SVG element if not already added.

    Args:
        parent_svg_element: The SVG group element for the specific tree.
    """
    if not any(
        "MathJax-script" in child.get("id", "")
        for child in parent_svg_element.findall(".//script")
    ):
        script_element = ET.SubElement(
            parent_svg_element, "script", {"type": "text/javascript"}
        )
        script_element.text = MATHJAX_SCRIPT


def render_mathjax_text(
    footer_group: ET.Element,
    lines: List[str],
    x: float,
    y: float,
    font_size_num: float,
    font_family: str,
    font_color: str,
    line_height: float,
    parent_svg_element: ET.Element,
) -> float:
    """
    Renders text using MathJax for browser-based LaTeX rendering.

    Args:
        footer_group: The footer group element.
        lines: List of text lines to render.
        x: X-coordinate for the text.
        y: Y-coordinate for the text.
        font_size_num: Font size.
        font_family: Font family.
        font_color: Font color.
        line_height: Line height.
        parent_svg_element: The parent SVG element.

    Returns:
        Updated y-coordinate after rendering all lines.
    """
    # Add MathJax script if needed
    add_mathjax_script(parent_svg_element)

    # Process each line
    for i, line in enumerate(lines):
        if not line.strip():  # Skip empty lines
            y += line_height
            continue

        text_element = ET.SubElement(
            footer_group,
            "text",
            {
                "x": str(x),
                "y": str(y + font_size_num / 2),
                "font-family": font_family,
                "font-size": str(font_size_num),
                "text-anchor": "middle",
                "fill": font_color,
                "dominant-baseline": "central",
                "class": "mathjax-text",  # Add class for potential styling
            },
        )
        text_element.text = line
        y += line_height

    return y


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
                "fontsize": f"{font_size_num}pt",
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
            "font-family": font_family,
            "font-size": str(font_size_num),
            "text-anchor": "middle",
            "fill": font_color,
            "dominant-baseline": "central",
        },
    )
    text_element.text = line
    return y + line_height


def add_info_text(
    footer_group: ET.Element,
    x: float,
    y: float,
    message: str,
    font_size_num: float,
    font_family: str,
    colors: Dict,
    line_height: float,
) -> float:
    """
    Adds information text below the footer content.

    Args:
        footer_group: The footer group element.
        x: X-coordinate for the text.
        y: Y-coordinate for the text.
        message: The message to display.
        font_size_num: Font size.
        font_family: Font family.
        colors: Dictionary containing color and style information.
        line_height: Line height.

    Returns:
        Updated y-coordinate after adding the info text.
    """
    info_text = ET.SubElement(
        footer_group,
        "text",
        {
            "x": str(x),
            "y": str(y + font_size_num / 2),
            "font-family": font_family,
            "font-size": str(font_size_num * 0.7),  # Smaller font
            "text-anchor": "middle",
            "fill": colors.get("subtle_text", "#777777"),
            "dominant-baseline": "central",
            "class": "latex-info",
        },
    )
    info_text.text = message
    return y + line_height * 0.8  # Add smaller line


def render_latex_svg_content(
    footer_group: ET.Element,
    lines: List[str],
    x: float,
    y: float,
    font_size_num: float,
    font_family: str,
    font_color: str,
    line_height: float,
    colors: Dict,
    latex_scale: float,
) -> float:
    """
    Renders content using direct LaTeX to SVG conversion.

    Args:
        footer_group: The footer group element.
        lines: List of text lines to render.
        x: X-coordinate for the text.
        y: Y-coordinate for the text.
        font_size_num: Font size.
        font_family: Font family.
        font_color: Font color.
        line_height: Line height.
        colors: Dictionary containing color and style information.
        latex_scale: Scaling factor for LaTeX elements.

    Returns:
        Updated y-coordinate after rendering all lines.
    """
    try:
        # Process each line
        for line in lines:
            y = process_latex_line(
                footer_group,
                line.strip(),
                x,
                y,
                font_size_num,
                font_family,
                font_color,
                line_height,
                latex_scale,
            )

        # Add info text about rendering method
        return add_info_text(
            footer_group,
            x,
            y,
            #     "[LaTeX rendered as SVG paths]",
            font_size_num,
            font_family,
            colors,
            line_height,
        )

    except ImportError:
        print("Warning: latex2svg not available. Rendering as plain text.")
        # Fall back to plain text rendering
        y = render_plain_text_content(
            footer_group,
            lines,
            x,
            y,
            font_size_num,
            font_family,
            font_color,
            line_height,
        )

        # Add info about fallback
        return add_info_text(
            footer_group,
            x,
            y,
            #     "[LaTeX rendering not available - plain text fallback]",
            font_size_num,
            font_family,
            colors,
            line_height,
        )


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


def add_footer_text(
    parent_svg_element: ET.Element,
    footer_text: str,
    tree_height: float,
    tree_width: float,
    colors: Dict,
    enable_latex: bool = False,
    use_mathjax: bool = False,
    latex_scale: float = 1.0,
    footer_font_size: Optional[float] = None,
):
    """
    Adds footer text below the tree, handling multi-line text and optional LaTeX rendering.

    Args:
        parent_svg_element: The SVG group element for the specific tree.
        footer_text: The text content for the footer.
        tree_height: The calculated height of the tree drawing area.
        tree_width: The calculated width of the tree drawing area.
        colors: Dictionary containing color and style information.
        enable_latex: Flag to enable LaTeX rendering via latex2svg_modular or MathJax.
        use_mathjax: Flag to enable MathJax-based browser rendering instead of direct SVG conversion.
        latex_scale: Scaling factor for LaTeX elements (only applies to direct SVG conversion).
        footer_font_size: Custom font size for footer text. If provided, overrides the default size.
    """
    if not footer_text:
        return

    footer_group = ET.SubElement(parent_svg_element, "g", {"class": "md3-footer"})

    # Calculate starting position
    x = tree_width / 2  # Center horizontally
    y = tree_height + FOOTER_PADDING  # Position below the tree

    # Font properties
    try:
        font_size_num = (
            footer_font_size
            if footer_font_size is not None
            else float(colors.get("font_size", 10)) + FOOTER_FONT_SIZE_ADD
        )
    except ValueError:
        font_size_num = 10.0
    font_family = colors.get(
        "font_family", "'Roboto', 'Arial', 'Helvetica', sans-serif"
    )
    font_color = colors.get("text_color", "#1A1A1A")
    line_height = font_size_num * 1.2  # Standard line height factor

    lines = footer_text.split("\n")

    # Handle different LaTeX rendering methods
    if enable_latex:
        if use_mathjax:
            # Browser-based MathJax rendering
            # Add MathJax script if needed
            if not any(
                "MathJax-script" in child.get("id", "")
                for child in parent_svg_element.findall(".//script")
            ):
                # Add MathJax configuration script
                config_script_element = ET.SubElement(
                    parent_svg_element, "script", {"type": "text/javascript"}
                )
                config_script_element.text = MATHJAX_SCRIPT

            # Render text with MathJax
            for line in lines:
                if not line.strip():  # Skip empty lines
                    y += line_height
                    continue

                # Add the raw text with LaTeX delimiters for MathJax to process
                text_element = ET.SubElement(
                    footer_group,
                    "text",
                    {
                        "x": str(x),
                        "y": str(y + font_size_num / 2),
                        "font-family": font_family,
                        "font-size": str(font_size_num),
                        "text-anchor": "middle",
                        "fill": font_color,
                        "dominant-baseline": "central",
                        "class": "mathjax-text",  # Add class for potential styling
                    },
                )
                text_element.text = line
                y += line_height

            # Add info about rendering method
            info_text = ET.SubElement(
                footer_group,
                "text",
                {
                    "x": str(x),
                    "y": str(y + font_size_num / 2),
                    "font-family": font_family,
                    "font-size": str(font_size_num * 0.7),  # Smaller font
                    "text-anchor": "middle",
                    "fill": colors.get("subtle_text", "#777777"),
                    "dominant-baseline": "central",
                    "class": "latex-info",
                },
            )
            # info_text.text = "[LaTeX rendered via MathJax (browser)]"
        else:
            # Direct SVG conversion using latex2svg_modular
            for line in lines:
                line = line.strip()
                if not line:  # Skip empty lines
                    y += line_height
                    continue

                # Check if this line contains LaTeX
                if "$" in line and line.count("$") >= 2:
                    try:
                        svg_dict = latex2svg_modular(
                            line,
                            params={
                                "fontsize": f"{font_size_num}pt",
                            },
                        )

                        svg_content = svg_dict["svg"]

                        # Remove XML declarations and DOCTYPE
                        svg_content = re.sub(r"<\?xml.*?\?>", "", svg_content)
                        svg_content = re.sub(r"<!DOCTYPE.*?>", "", svg_content)

                        # Parse the SVG content
                        latex_svg_root = ET.fromstring(svg_content)

                        # Get dimensions and calculate position
                        latex_width = float(svg_dict["width"]) * latex_scale
                        latex_height = float(svg_dict["height"]) * latex_scale
                        latex_x = x - (latex_width / 2)
                        latex_y = y

                        # Create a group for the LaTeX SVG content
                        latex_group = ET.SubElement(
                            footer_group,
                            "g",
                            {
                                "transform": f"translate({latex_x:.2f}, {latex_y:.2f}) scale({latex_scale})",
                                "class": "latex-svg-group",
                            },
                        )

                        # Extract and append SVG elements
                        for child in latex_svg_root:
                            # Skip the defs section
                            if child.tag == "{http://www.w3.org/2000/svg}defs":
                                continue

                            # Apply text color to paths and text
                            if child.tag in [
                                "{http://www.w3.org/2000/svg}path",
                                "{http://www.w3.org/2000/svg}text",
                            ]:
                                child.set("fill", font_color)

                            # Process nested groups
                            if child.tag == "{http://www.w3.org/2000/svg}g":
                                for path in child.findall(
                                    ".//{http://www.w3.org/2000/svg}path"
                                ):
                                    path.set("fill", font_color)
                                for text in child.findall(
                                    ".//{http://www.w3.org/2000/svg}text"
                                ):
                                    text.set("fill", font_color)

                            # Add the child to our group
                            latex_group.append(child)

                        # Update y position
                        y += latex_height + (line_height * 0.5)

                    except Exception as e:
                        print(
                            f"Warning: Failed to render LaTeX: '{line}' - {e}. Falling back to text."
                        )
                        # Fallback to plain text
                        text_element = ET.SubElement(
                            footer_group,
                            "text",
                            {
                                "x": str(x),
                                "y": str(y + font_size_num / 2),
                                "font-family": font_family,
                                "font-size": str(font_size_num),
                                "text-anchor": "middle",
                                "fill": font_color,
                                "dominant-baseline": "central",
                            },
                        )
                        text_element.text = line
                        y += line_height
                else:
                    # Plain text line
                    text_element = ET.SubElement(
                        footer_group,
                        "text",
                        {
                            "x": str(x),
                            "y": str(y + font_size_num / 2),
                            "font-family": font_family,
                            "font-size": str(font_size_num),
                            "text-anchor": "middle",
                            "fill": font_color,
                            "dominant-baseline": "central",
                        },
                    )
                    text_element.text = line
                    y += line_height

            # Add info about rendering method
            info_text = ET.SubElement(
                footer_group,
                "text",
                {
                    "x": str(x),
                    "y": str(y + font_size_num / 2),
                    "font-family": font_family,
                    "font-size": str(font_size_num * 0.7),
                    "text-anchor": "middle",
                    "fill": colors.get("subtle_text", "#777777"),
                    "dominant-baseline": "central",
                    "class": "latex-info",
                },
            )
            # info_text.text = "[LaTeX rendered as SVG paths]"
    else:
        # Standard plain text rendering (no LaTeX)
        for line in lines:
            line = line.strip()
            if not line:  # Skip empty lines
                y += line_height
                continue
            text_element = ET.SubElement(
                footer_group,
                "text",
                {
                    "x": str(x),
                    "y": str(y + font_size_num / 2),
                    "font-family": font_family,
                    "font-size": str(font_size_num),
                    "text-anchor": "middle",
                    "fill": font_color,
                    "dominant-baseline": "central",
                },
            )
            text_element.text = line
            y += line_height


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
                    "x": f"{bg_x:.2f}",
                    "y": f"{bg_y:.2f}",
                    "width": f"{bg_width:.2f}",
                    "height": f"{bg_height:.2f}",
                    "rx": "3",
                    "fill": highlight_color,
                    "fill-opacity": background_opacity,
                    "stroke": "none",
                },
            )

        # Add the text element
        text_el = ET.SubElement(
            label_group,
            "text",
            {
                "x": f"{label_x:.2f}",
                "y": f"{label_y:.2f}",
                "font-family": colors.get("font_family", FONT_SANS_SERIF),
                "font-size": str(font_size),
                "font-weight": font_weight,
                "text-anchor": anchor,
                "dominant-baseline": "central",
                "fill": highlight_color if highlight else colors["base_text"],
            },
        )
        text_el.text = text


def add_caption(
    container: ET.Element,
    caption: str,
    caption_y: float,
    total_width: float,
    colors: Dict,
    font_size: Optional[float] = None,  # Add this parameter
) -> None:
    """
    Add a caption to the SVG.

    Args:
        container: Parent SVG element
        caption: Caption text
        caption_y: Y position for the caption
        total_width: Total width available
        colors: Color and style information
        font_size: Optional custom font size for the caption
    """
    caption_group = ET.SubElement(container, "g", {"class": "md3-caption"})

    # Get default font size from colors or use a reasonable default
    default_font_size = colors.get("caption_font_size", 14)

    # Use custom font size if provided, otherwise use default
    caption_font_size = font_size if font_size is not None else default_font_size

    # Create text element with appropriate styling
    caption_text = ET.SubElement(
        caption_group,
        "text",
        {
            "x": str(total_width / 2),
            "y": str(caption_y),
            "text-anchor": "middle",
            "dominant-baseline": "middle",
            "font-family": colors.get("font_family", "'Roboto', sans-serif"),
            "font-size": str(caption_font_size),  # Use the determined font size
            "font-weight": "500",
            "fill": colors.get("caption_color", colors.get("text_color", "#3C3C3C")),
        },
    )
    caption_text.text = caption


def add_tree_label(
    tree_group: ET.Element,
    tree_label: str,
    tree_label_font_size: Optional[float],
    leaf_font_size: Optional[float],
    tree_label_style: Optional[Dict],
    enable_latex: bool = False,
    use_mathjax: bool = False,
    latex_scale: float = 1.0,
) -> None:
    """
    Add a label to a tree with optional LaTeX rendering.
    """
    # Determine appropriate font size for the label
    label_font_size = determine_tree_label_font_size(
        tree_label_font_size, leaf_font_size
    )

    # Create label style
    label_style = create_tree_label_style(label_font_size, tree_label_style)

    # Create the label group and text element
    label_group = ET.SubElement(tree_group, "g", {"class": "md3-tree-label"})

    # Offset the label slightly from the top-left corner
    label_x = 10  # 10px from left edge
    label_y = -15  # 15px above the tree (adjust as needed)

    # Check for LaTeX rendering
    if enable_latex:
        # Use latex2svg_modular instead of folded_latex2svg
        latex_result = latex2svg_modular(tree_label)
        if latex_result and "svg" in latex_result and latex_result["svg"]:
            # Create a group for the rendered LaTeX
            y_offset = label_y - (label_font_size * 0.5)  # Approximate vertical center
            latex_group = ET.SubElement(
                label_group,
                "g",
                {"transform": f"translate({label_x}, {y_offset}) scale({latex_scale})"},
            )

            # Parse the SVG content from latex2svg_modular
            svg_content = f"<g>{latex_result['svg']}</g>"
            parsed_svg = ET.fromstring(svg_content)

            # Append children of the parsed SVG group
            for child in parsed_svg:
                if "fill" in child.attrib and child.attrib["fill"] != "none":
                    child.set("fill", label_style.get("fill", "currentColor"))
                if "stroke" in child.attrib and child.attrib["stroke"] != "none":
                    child.set("stroke", label_style.get("fill", "currentColor"))
                latex_group.append(child)
        else:
            raise ValueError("latex2svg_modular did not return valid SVG content.")
    else:
        # No LaTeX, use plain text
        label_text = ET.SubElement(
            label_group, "text", {"x": str(label_x), "y": str(label_y), **label_style}
        )
        label_text.text = str(tree_label)


def determine_tree_label_font_size(
    tree_label_font_size: Optional[float], leaf_font_size: Optional[float]
) -> float:
    """
    Determine the font size for a tree label.

    Args:
        tree_label_font_size: Specified font size for the label
        leaf_font_size: Font size for leaf labels

    Returns:
        Determined font size
    """
    if tree_label_font_size:
        return float(tree_label_font_size)
    elif leaf_font_size:
        return float(leaf_font_size) * 1.2  # 20% larger than leaf font
    else:
        return 18  # Default size


def create_tree_label_style(
    label_font_size: float, tree_label_style: Optional[Dict]
) -> Dict:
    """
    Create style for a tree label.

    Args:
        label_font_size: Font size for the label
        tree_label_style: Custom style for the label

    Returns:
        Label style dictionary
    """
    # Default styles for tree labels
    label_style = {
        "font-family": "'Roboto', 'Arial', 'Helvetica', sans-serif",
        "font-size": str(label_font_size),
        "font-weight": "500",  # Semi-bold
        "text-anchor": "start",  # Left-aligned
        "dominant-baseline": "hanging",  # Top-aligned
        "fill": "#1A1A1A",  # Near-black for good contrast
    }

    # Apply any custom styles provided
    if tree_label_style and isinstance(tree_label_style, dict):
        label_style.update(tree_label_style)

    return label_style


def add_mathjax_to_svg(svg_root: ET.Element) -> None:
    """
    Add MathJax scripts to the SVG root element.

    Args:
        svg_root: SVG root element
    """
    # Check if MathJax script is already in the SVG
    if mathjax_already_in_svg(svg_root):
        return

    # Add MathJax configuration script
    add_mathjax_config_script(svg_root)

    # Add MathJax source script
    add_mathjax_source_script(svg_root)


def mathjax_already_in_svg(svg_root: ET.Element) -> bool:
    """
    Check if MathJax script is already in the SVG.

    Args:
        svg_root: SVG root element

    Returns:
        True if MathJax script is already present, False otherwise
    """
    return any(
        "MathJax-script" in child.get("id", "")
        for child in svg_root.findall(".//script")
    )


def add_mathjax_config_script(svg_root: ET.Element) -> None:
    """
    Add MathJax configuration script to SVG.

    Args:
        svg_root: SVG root element
    """
    config_script_element = ET.SubElement(
        svg_root, "script", {"type": "text/javascript"}
    )
    config_script_element.text = MATHJAX_SCRIPT


def add_mathjax_source_script(svg_root: ET.Element) -> None:
    """
    Add MathJax source script to SVG.

    Args:
        svg_root: SVG root element
    """
    ET.SubElement(
        svg_root,
        "script",
        {"id": "MathJax-script", "async": "true", "src": MATHJAX_SRC},
    )

