from brancharchitect.plot.paper_plot.paper_plot_constants import (
    QUANTA_COLORS,
    LIGHT_COLORS,
    MATERIAL_LIGHT_COLORS,
    MATERIAL_DARK_COLORS,
    NATURE_MD3_COLORS,
    MD3_SCIENTIFIC_LIGHT,
    MD3_SCIENTIFIC_DARK,
    MD3_SCIENTIFIC_PRINT,
)  # Utility functions for color and stroke-width handling, moved from paper_plots.py

from typing import Dict, Optional


def apply_manual_stroke_width(stroke_width, colors_dict, highlight_options):
    """
    Applies manual stroke width to the colors dictionary based on highlight options.
    """
    for key, value in colors_dict.items():
        if key in highlight_options:
            colors_dict[key]["stroke_width"] = stroke_width
    return colors_dict


def update_highlight_stroke_widths(
    highlight_options, tree_index, highlight_stroke_width
):
    """
    Updates the stroke widths for highlighted options in the tree index.
    """
    for option in highlight_options:
        if option in tree_index:
            tree_index[option]["stroke_width"] = highlight_stroke_width


# -----------------------------------------------------------------------------
# B. Color & Stroke-Width Functions
# -----------------------------------------------------------------------------
def select_color_scheme(color_mode: str, custom_colors: Optional[Dict] = None) -> Dict:
    """
    Select the appropriate color scheme based on the color mode.

    Args:
        color_mode: Color scheme name
        custom_colors: Custom color palette

    Returns:
        Dictionary containing color and style information
    """
    color_schemes = {
        "custom": custom_colors if custom_colors is not None else QUANTA_COLORS,
        "light": LIGHT_COLORS,
        "material_light": MATERIAL_LIGHT_COLORS,
        "material_dark": MATERIAL_DARK_COLORS,
        "nature_md3": NATURE_MD3_COLORS,
        "md3_scientific_light": MD3_SCIENTIFIC_LIGHT,
        "md3_scientific_dark": MD3_SCIENTIFIC_DARK,
        "md3_scientific_print": MD3_SCIENTIFIC_PRINT,
        # Default to quanta
        "quanta": QUANTA_COLORS,
    }

    return color_schemes.get(color_mode, QUANTA_COLORS)
