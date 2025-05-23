import xml.etree.ElementTree as ET
# Assuming folded_latex2svg is accessible relative to this path
from .folded_latex2svg.folded_latex2svg import latex2svg_modular

def _apply_fill_recursively(element: ET.Element, color: str):
    """Recursively applies fill color to visible SVG elements."""
    # Elements that typically have fill/stroke that should be colored
    colorable_tags = [
        "{http://www.w3.org/2000/svg}path",
        "{http://www.w3.org/2000/svg}text",
        "{http://www.w3.org/2000/svg}rect",
        "{http://www.w3.org/2000/svg}circle",
        "{http://www.w3.org/2000/svg}ellipse",
        "{http://www.w3.org/2000/svg}polygon",
        "{http://www.w3.org/2000/svg}polyline",
        "{http://www.w3.org/2000/svg}line", # Lines use stroke, but let's color it too
        "{http://www.w3.org/2000/svg}g", # Recurse into groups
    ]

    if element.tag in colorable_tags:
        # Apply fill, avoid overwriting 'none'
        if 'fill' not in element.attrib or element.attrib.get('fill') != 'none':
            element.set("fill", color)
        # Apply stroke for lines, potentially others if desired
        if element.tag == "{http://www.w3.org/2000/svg}line":
             if 'stroke' not in element.attrib or element.attrib.get('stroke') != 'none':
                element.set("stroke", color)

        # Recurse for groups
        if element.tag == "{http://www.w3.org/2000/svg}g":
            for child in element:
                _apply_fill_recursively(child, color)


def render_and_insert_latex_svg(
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
    Renders a LaTeX string to SVG and inserts it into a parent group.

    Handles positioning, scaling, coloring, and fallback to text.

    Args:
        parent_group: The parent SVG element (e.g., a <g>) to insert into.
        latex_string: The LaTeX string to render.
        x: Target x-coordinate for the alignment point.
        y: Target y-coordinate for the alignment point (usually baseline or center).
        font_size_num: Base font size in pixels (used for scaling em units).
        font_color: The desired fill/stroke color for the LaTeX elements.
        font_family: Font family for fallback text rendering.
        latex_scale: Additional scaling factor applied to the LaTeX SVG.
        align: Horizontal alignment ('left', 'middle', 'right') relative to x.

    Returns:
        The created SVG <g> element containing the LaTeX, or the fallback <text> element,
        or None if the input string was empty or only whitespace.
    """
    stripped_latex = latex_string.strip()
    if not stripped_latex:
        return None

    fallback_rendered = False
    try:
        # 1. Render LaTeX to SVG string
        latex_result = latex2svg_modular(
            stripped_latex,
            params={
                "fontsize": int(font_size_num), # Pass font size to converter
                "scale": latex_scale, # Pass scale
            },
        )

        # Check for errors or empty SVG from the converter
        if latex_result.get("error") or not latex_result.get("svg"):
            print(f"Warning: latex2svg failed for '{stripped_latex}'. Error: {latex_result.get('error', 'Empty SVG')}")
            raise ValueError("LaTeX rendering failed") # Trigger fallback

        # 2. Parse the resulting SVG string
        parser = ET.XMLParser()  # Removed recover=True for compatibility
        latex_svg_root = ET.fromstring(latex_result["svg"], parser=parser)

        # 3. Calculate Dimensions and Position
        # Convert em units from latex2svg result to px using font_size_num
        # Apply the overall latex_scale as well
        latex_w_px = latex_result.get("width", 0) * font_size_num * latex_scale
        latex_h_px = latex_result.get("height", 0) * font_size_num * latex_scale
        # valign is the offset from the baseline (0) to the bottom of the SVG box
        latex_valign_px = latex_result.get("valign", 0) * font_size_num * latex_scale

        # Adjust x based on alignment
        if align == "middle":
            final_x = x - (latex_w_px / 2)
        elif align == "right":
            final_x = x - latex_w_px
        else:  # 'left' or default
            final_x = x

        # Adjust y: Aim to vertically center the visual height (h) around the target y
        # The SVG's coordinate system starts at its top-left.
        # Its baseline is effectively at y=0 within its own system.
        # valign tells us how far the bottom edge is below the baseline.
        # The top edge is then baseline - height - valign = -h - valign
        # We want the center (top_edge + bottom_edge)/2 = (-h - valign + (-valign))/2 = (-h - 2*valign)/2
        # to align with the target y in the parent system.
        # So, the translation should be y - center_offset = y - (-h/2 - valign) = y + h/2 + valign
        # Let's try a simpler approach: align the baseline (y=0 in SVG) with the target y,
        # then shift up by half the height for approximate centering.
        # final_y = y - (latex_h_px / 2) # Approximate centering based on height only
        # Let's try aligning the baseline (y=0 in SVG) with the target y, adjusted by valign
        final_y = y + latex_valign_px # Align baseline, considering valign offset

        # 4. Create Insertion Group with Transform
        latex_group = ET.SubElement(
            parent_group,
            "g",
            {
                # Apply calculated translation and the overall scale
                "transform": f"translate({final_x:.2f}, {final_y:.2f}) scale({latex_scale})",
                "class": "latex-svg-inline",
            },
        )

        # 5. Add Styled Visible Elements (NO DEFS MERGING)
        for child in latex_svg_root:
            if child.tag == "{http://www.w3.org/2000/svg}defs":
                continue # Skip defs
            # Apply fill color recursively
            _apply_fill_recursively(child, font_color)
            latex_group.append(child)

        return latex_group # Success

    except Exception as e:
        # 6. Handle Fallback (if any exception occurred)
        if not fallback_rendered: # Avoid rendering fallback twice if latex2svg failed first
            print(f"Warning: Rendering '{stripped_latex}' as plain text due to error: {e}")
            fallback_text = ET.SubElement(
                parent_group,
                "text",
                {
                    "x": str(x),
                    "y": str(y),
                    "font-family": font_family,
                    "font-size": str(font_size_num),
                    "text-anchor": align, # Use alignment for fallback too
                    "fill": font_color,
                    "dominant-baseline": "central", # Good baseline for fallback
                },
            )
            fallback_text.text = stripped_latex
            fallback_rendered = True
            return fallback_text
        return None # Failed completely