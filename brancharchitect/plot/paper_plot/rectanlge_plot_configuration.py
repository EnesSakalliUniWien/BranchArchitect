import matplotlib.colors as mcolors
import numpy as np


# =============================================================================
# Configuration & Constants
# =============================================================================


def _darken_color(color_hex: str, factor: float = 0.7) -> str:
    """Darkens a hex color. Returns hex."""
    try:
        rgb = mcolors.hex2color(color_hex)
        hsv = mcolors.rgb_to_hsv(rgb)
        hsv[2] = np.clip(hsv[2] * factor, 0, 1)
        return mcolors.to_hex(mcolors.hsv_to_rgb(hsv))
    except ValueError:
        return color_hex  # Return

# Define primary colors (adjust as needed)
UNIQUE_SPLIT_LEFT_COLOUR = "#007e5c"  # Teal/Greenish
UNIQUE_SPLIT_RIGHT_COLOR = "#D55E00"  # Orange/Vermillion

# Pre-calculate darker versions for borders/text
LEFT_DARK = _darken_color(UNIQUE_SPLIT_LEFT_COLOUR, 0.6)
RIGHT_DARK = _darken_color(UNIQUE_SPLIT_RIGHT_COLOR, 0.6)

DEFAULT_STYLE_CONFIG = {
    # General Figure
    "figure_size": (12, 8),  # Default size for 2x2 layout
    "dpi": 300,
    "face_color": "white",
    "axes_face_color": "#FFFFFF",
    # Typography
    "font_family": "sans-serif",
    "font_sans_serif": ["Roboto", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font_weight": "normal",
    "title_fontsize": 16,
    "subtitle_fontsize": 12,
    "group_summary_fontsize": 10,  # For Pair labels
    "atom_label_fontsize": 9,
    "atom_index_fontsize": 8,
    "matrix_font_size": 8,
    "solution_font_size": 9,
    # Layout & Spacing
    "y_position": 0.6,  # Base y for drawing elements in panels
    "box_height": 0.25,
    "box_width": 0.7,
    "element_spacing": 1.0,
    "group_bg_padding_x": 0.3,
    "group_bg_padding_y": 0.25,
    "summary_y_offset": 0.5,  # Offset for Pair label above group
    "connecting_line_y_offset": 0.15,
    # Colors & Opacity
    "group_bg_alpha": 0.20,
    "atom_box_alpha": 0.85,
    "connecting_line_alpha": 0.7,
    "matrix_header_color": "#E0E0E0",
    "matrix_grid_color": "#AAAAAA",
    "matrix_cell_color": "white",
    # Borders & Styles
    "group_bg_linewidth": 1.0,
    "atom_box_linewidth": 1.0,
    "connecting_line_linewidth": 1.5,
    "group_bg_corner_radius": 0.1,
    "atom_box_corner_radius": 0.1,
    "atom_box_padding": 0.1,
    # Atom Index Position
    "index_label_x_offset": 0.05,
    "index_label_y_offset": 0.05,
    # Color Mapping for Pairs (Extend if more pairs are common)
    "colors": {
        0: {"fill": UNIQUE_SPLIT_LEFT_COLOUR, "stroke": LEFT_DARK, "name": "Pair 1"},
        1: {"fill": UNIQUE_SPLIT_RIGHT_COLOR, "stroke": RIGHT_DARK, "name": "Pair 2"},
        # Add more based on a chosen palette if needed
        2: {
            "fill": "#56B4E9",
            "stroke": _darken_color("#56B4E9"),
            "name": "Pair 3",
        },  # Sky Blue
        3: {
            "fill": "#CC79A7",
            "stroke": _darken_color("#CC79A7"),
            "name": "Pair 4",
        },  # Purplish Red
    },
}