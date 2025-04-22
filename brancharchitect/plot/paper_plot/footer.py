"""
Overlay label utilities for BranchArchitect.

This module contains functions for rendering overlay labels, footers, and captions.
"""

import re
import xml.etree.ElementTree as ET
from typing import Dict, Optional, Tuple


from brancharchitect.plot.paper_plot.paper_plot_constants import (
    FOOTER_PADDING,
    FOOTER_FONT_SIZE_ADD,
    MATHJAX_SCRIPT,
)

from brancharchitect.plot.paper_plot.folded_latex2svg.folded_latex2svg import (
    latex2svg_modular,
)


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


def create_footer_group(parent_svg_element: ET.Element) -> ET.Element:
    """
    Creates and returns a group element for the footer.

    Args:
        parent_svg_element: The SVG group element for the specific tree.

    Returns:
        The created footer group element.
    """
    return ET.SubElement(parent_svg_element, "g", {"class": "md3-footer"})


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
