"""
Overlay label utilities for BranchArchitect.

This module contains functions for rendering overlay labels, footers, and captions.
"""

import xml.etree.ElementTree as ET
from typing import Dict, Optional
from brancharchitect.plot.paper_plot.footer import insert_latex_svg


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
    # Default styles for tree labels - position on left side
    label_style = {
        "font-family": "'Roboto', 'Arial', 'Helvetica', sans-serif",
        "font-size": str(label_font_size),
        "font-weight": "500",  # Semi-bold
        "text-anchor": "start",  # Left-aligned (default for left positioning)
        "dominant-baseline": "hanging",  # Top-aligned
        "fill": "#1A1A1A",  # Near-black for good contrast
    }

    # Apply any custom styles provided
    if tree_label_style and isinstance(tree_label_style, dict):
        label_style.update(tree_label_style)

    return label_style


def add_tree_label(
    tree_group: ET.Element,
    tree_label: str,
    tree_label_font_size: Optional[float],
    leaf_font_size: Optional[float],
    tree_label_style: Optional[Dict],
    enable_latex: bool = False,
    latex_scale: float = 1.0,
) -> None:
    print(
        f"[DEBUG] add_tree_label called with label: {tree_label!r}, enable_latex={enable_latex}, latex_scale={latex_scale}"
    )
    label_font_size = determine_tree_label_font_size(
        tree_label_font_size, leaf_font_size
    )
    label_style = create_tree_label_style(label_font_size, tree_label_style)

    # Position the label on the left side of the tree
    # Use a small offset from the left edge for better spacing
    label_x = 5  # Small offset from left edge
    label_y = 5  # Small offset from top for better visibility
    
    # Ensure left alignment
    label_style["text-anchor"] = "start"  # Left-aligned
    
    contains_latex = "$" in str(tree_label) or "\\" in str(tree_label)

    if enable_latex or contains_latex:
        try:
            result = insert_latex_svg(
                parent_group=tree_group,
                latex_string=tree_label,
                x=label_x,
                y=label_y,
                font_size_num=label_font_size,
                font_color=label_style.get("fill", "#1A1A1A"),
                font_family=label_style.get(
                    "font-family", "'Roboto', 'Arial', 'Helvetica', sans-serif"
                ),
                latex_scale=latex_scale,
                align="left",  # Ensure left alignment for LaTeX
            )
            if result is None:
                raise ValueError("LaTeX SVG output was None")
        except Exception as e:
            print(
                f"Warning: Failed to render LaTeX: '{tree_label}' - {e}. Falling back to text."
            )
            label_text = ET.SubElement(
                tree_group,
                "text",
                {"x": str(label_x), "y": str(label_y), **label_style},
            )
            label_text.text = str(tree_label)
    else:
        label_text = ET.SubElement(
            tree_group, "text", {"x": str(label_x), "y": str(label_y), **label_style}
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
        return float(leaf_font_size) * 1.2  # 20% larger than leaf font (fixed the 11000 error)
    else:
        return 18  # Default size
