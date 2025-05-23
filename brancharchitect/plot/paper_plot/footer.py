from typing import List, Optional
import xml.etree.ElementTree as ET
import re
from typing import Dict, Tuple
from brancharchitect.plot.paper_plot.paper_plot_constants import (
    FOOTER_PADDING,  # Keep for reference, but use smaller values derived from it
    FOOTER_FONT_SIZE_ADD,
)

from brancharchitect.plot.paper_plot.folded_latex2svg.folded_latex2svg import (
    latex2svg_modular,
)

# --- Constants for Reduced Padding ---
INITIAL_FOOTER_OFFSET_FACTOR = 0.4  # Reduce initial space above footer (40% of FOOTER_PADDING)
LINE_HEIGHT_MULTIPLIER = 1.1  # Reduce space between lines (from 1.2)
LATEX_LINE_HEIGHT_MULTIPLIER = 1.15 # Slightly more for LaTeX lines
MINIMAL_PADDING = 2 # Minimal padding used in height calculation


def calculate_footer_height(
    tree_footers: List[Optional[str]],
    footer_font_size: Optional[float],
    enable_latex: bool,
) -> float:
    """
    Calculate the total footer height needed based on footer content (Optimized for less padding).

    Args:
        tree_footers: List of footer texts for each tree
        footer_font_size: Font size for the footer text (independent parameter)
        enable_latex: Whether LaTeX rendering is enabled

    Returns:
        Calculated footer height
    """
    if not any(tree_footers):
        return 0

    # Use the provided footer_font_size, default to 10 if not set
    font_size_num = float(footer_font_size) if footer_font_size is not None else 10.0
    # Use reduced line height multiplier
    base_line_height = font_size_num * LINE_HEIGHT_MULTIPLIER

    # Count maximum number of lines in footers
    max_footer_lines = count_max_footer_lines(tree_footers)

    if max_footer_lines > 0:
        return calculate_footer_height_from_lines(
            max_footer_lines, base_line_height, enable_latex
        )

    return 0


def count_max_footer_lines(tree_footers: List[Optional[str]]) -> int:
    """
    Count the maximum number of lines across all footers.

    Args:
        tree_footers: List of footer texts

    Returns:
        Maximum number of lines
    """
    max_footer_lines = 0
    for footer in tree_footers:
        if footer:
            lines = footer.split("\n")
            max_footer_lines = max(max_footer_lines, len(lines))

    return max_footer_lines


def calculate_footer_height_from_lines(
    max_footer_lines: int, base_line_height: float, enable_latex: bool
) -> float:
    """
    Calculate footer height based on number of lines (Optimized for less padding).

    Args:
        max_footer_lines: Maximum number of lines in footers
        base_line_height: Base height of each line (using reduced multiplier)
        enable_latex: Whether LaTeX rendering is enabled

    Returns:
        Calculated footer height
    """
    # Use slightly larger multiplier for LaTeX lines if enabled
    estimated_line_height = (
        base_line_height * (LATEX_LINE_HEIGHT_MULTIPLIER / LINE_HEIGHT_MULTIPLIER)
        if enable_latex
        else base_line_height
    )
    # Use minimal padding for height calculation
    footer_height = MINIMAL_PADDING + (estimated_line_height * max_footer_lines) + MINIMAL_PADDING

    # Add minimal space for rendering method info text when using LaTeX
    if enable_latex:
        # Use a smaller addition, relative to font size
        footer_height += base_line_height * 0.4

    return footer_height


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
    Calculates the starting position for the footer text (Optimized for less padding).

    Args:
        tree_height: The calculated height of the tree drawing area.
        tree_width: The calculated width of the tree drawing area.

    Returns:
        Tuple containing x and y coordinates for the footer.
    """
    x = tree_width / 2  # Center horizontally
    # Use reduced initial offset based on FOOTER_PADDING
    y = tree_height + (FOOTER_PADDING * INITIAL_FOOTER_OFFSET_FACTOR)
    return x, y


def get_font_properties(
    colors: Dict, footer_font_size: Optional[float] = None
) -> Tuple[float, str, str, float]:
    """
    Extracts font properties from the colors dictionary (Optimized for less padding).

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
    # Use reduced line height multiplier
    line_height = font_size_num * LINE_HEIGHT_MULTIPLIER

    return font_size_num, font_family, font_color, line_height

def insert_latex_svg(
    parent_group: ET.Element,
    latex_string: str,
    x: float,
    y: float,
    font_size_num: float = 12,
    font_color: str = "#000000",
    font_family: str = "Arial",
    latex_scale: float = 1.0,
    align: str = "middle",
) -> Optional[ET.Element]:
    """
    Insert LaTeX rendering in SVG format. (No changes needed for padding here)

    Args:
        parent_group: Parent SVG element
        latex_string: LaTeX string to render
        x: X position
        y: Y position
        font_size_num: Font size
        font_color: Font color
        font_family: Font family
        latex_scale: Scaling factor for LaTeX elements
        align: Text alignment - 'middle', 'left', or 'right'

    Returns:
        Created SVG element or None if failed
    """
    try:
        svg_dict = latex2svg_modular(
            latex_string,
            params={"fontsize": int(font_size_num)},
        )
        # print(f"SVG dict: {svg_dict}") # Keep commented unless debugging
        svg_content = svg_dict["svg"]
        svg_content = re.sub(r"<\?xml.*?\?>", "", svg_content)
        svg_content = re.sub(r"<!DOCTYPE.*?>", "", svg_content)
        latex_svg_root = ET.fromstring(svg_content)
        latex_width = float(svg_dict["width"]) * latex_scale
        latex_height = float(svg_dict["height"]) * latex_scale # Keep height calculation

        # Adjust x position based on alignment
        if align == "middle":
            latex_x = x - (latex_width / 2)
        elif align == "right":
            latex_x = x - latex_width
        else:  # left or default
            latex_x = x

        # Adjust y position to align baseline (approximate)
        # latex2svg usually aligns baseline near the bottom
        # We want to align it similarly to text's dominant-baseline: central
        # Shifting up by roughly half the height seems reasonable
        latex_y = y - (latex_height / 2) # Adjust y for vertical centering

        latex_group = ET.SubElement(
            parent_group,
            "g",
            {
                "transform": f"translate({latex_x:.2f}, {latex_y:.2f}) scale({latex_scale})",
                "class": "latex-svg-group",
            },
        )

        # --- Merge <defs> --- (Code unchanged)
        main_svg_root = parent_group
        if main_svg_root.tag != "{http://www.w3.org/2000/svg}svg":
            root_candidate = parent_group
            while hasattr(root_candidate, 'tag') and root_candidate.tag != "{http://www.w3.org/2000/svg}svg":
                break
            main_svg_root = root_candidate
        main_defs = None
        for elem in main_svg_root:
            if elem.tag == "{http://www.w3.org/2000/svg}defs":
                main_defs = elem
                break
        if main_defs is None:
            main_defs = ET.SubElement(main_svg_root, "defs")
        for child in latex_svg_root:
            if child.tag == "{http://www.w3.org/2000/svg}defs":
                for defs_child in child:
                    main_defs.append(defs_child)

        # --- Add visible children --- (Code unchanged, but ensure fill is applied)
        for child in latex_svg_root:
            if child.tag == "{http://www.w3.org/2000/svg}defs":
                continue
            # Apply fill color recursively
            elements_to_color = [child]
            if child.tag == "{http://www.w3.org/2000/svg}g":
                elements_to_color.extend(child.findall(".//*")) # Find all descendants

            for elem in elements_to_color:
                 # Only apply fill to elements that typically have one
                if elem.tag in [
                    "{http://www.w3.org/2000/svg}path",
                    "{http://www.w3.org/2000/svg}text",
                    "{http://www.w3.org/2000/svg}rect", # Add other shapes if needed
                    "{http://www.w3.org/2000/svg}circle",
                    "{http://www.w3.org/2000/svg}ellipse",
                ]:
                    # Avoid overwriting existing fills if they are meaningful (e.g., 'none')
                    if 'fill' not in elem.attrib or elem.attrib.get('fill') != 'none':
                         elem.set("fill", font_color)
                if elem.tag == "{http://www.w3.org/2000/svg}text":
                     elem.set('class',"latex-text") # Keep class assignment

            latex_group.append(child)

        return latex_group
    except Exception as e:
        print(f"Warning: Failed to render LaTeX: '{latex_string}' - {e}. Falling back to text.")
        # Fallback text rendering (unchanged)
        text_element = ET.SubElement(
            parent_group,
            "text",
            {
                "x": str(x),
                "y": str(y), # Use y directly for central baseline
                "font-family": font_family,
                "font-size": str(font_size_num),
                "text-anchor": align,
                "fill": font_color,
                "dominant-baseline": "central", # Vertically center fallback text
            },
        )
        text_element.text = latex_string
        return None


def add_footer_text(
    parent_svg_element: ET.Element,
    footer_text: str,
    tree_height: float,
    tree_width: float,
    colors: Dict,
    enable_latex: bool = False,
    latex_scale: float = 1.0,
    footer_font_size: Optional[float] = None,
):
    """
    Adds footer text below the tree (Optimized for less padding).

    Args:
        parent_svg_element: The SVG group element for the specific tree.
        footer_text: The text content for the footer.
        tree_height: The calculated height of the tree drawing area.
        tree_width: The calculated width of the tree drawing area.
        colors: Dictionary containing color and style information.
        enable_latex: Flag to enable LaTeX rendering.
        latex_scale: Scaling factor for LaTeX elements.
        footer_font_size: Custom font size for footer text.
    """
    if not footer_text:
        return

    footer_group = create_footer_group(parent_svg_element)

    # Calculate starting position using optimized function
    x, current_y = calculate_footer_position(tree_height, tree_width)

    # Font properties using optimized function
    font_size_num, font_family, font_color, line_height = get_font_properties(
        colors, footer_font_size
    )

    lines = footer_text.split("\n")
    first_line = True

    for line in lines:
        line = line.strip()
        if not line:  # Skip empty lines but advance y
            current_y += line_height
            continue

        # Adjust y for vertical centering (dominant-baseline="central")
        # Add half line height initially, then full line height for subsequent lines
        if first_line:
            current_y += line_height / 2
            first_line = False
        else:
            current_y += line_height

        # Check if this line contains LaTeX
        if enable_latex and "$" in line and line.count("$") >= 2:
            insert_latex_svg(
                footer_group,
                line,
                x,
                current_y, # Pass current_y for vertical centering
                font_size_num,
                font_color,
                font_family,
                latex_scale,
                align="middle", # Assuming centered alignment for footer
            )
            # Note: y is already advanced by line_height before this loop iteration
        else:
            # Plain text line
            text_element = ET.SubElement(
                footer_group,
                "text",
                {
                    "x": str(x),
                    "y": str(current_y), # Use current_y directly
                    "font-family": font_family,
                    "font-size": str(font_size_num),
                    "text-anchor": "middle",
                    "fill": font_color,
                    "dominant-baseline": "central", # Vertically center text
                },
            )
            text_element.text = line
            # Note: y is already advanced by line_height before this loop iteration
