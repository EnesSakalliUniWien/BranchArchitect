# Utility functions for color and stroke-width handling, moved from paper_plots.py

def select_color_scheme(color_mode, custom_colors=None):
    """
    Selects a color scheme based on the given mode and optional custom colors.
    """
    if color_mode == 'default':
        return ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    elif color_mode == 'custom' and custom_colors:
        return custom_colors
    else:
        raise ValueError("Invalid color mode or missing custom colors.")

def apply_manual_stroke_width(stroke_width, colors_dict, highlight_options):
    """
    Applies manual stroke width to the colors dictionary based on highlight options.
    """
    for key, value in colors_dict.items():
        if key in highlight_options:
            colors_dict[key]['stroke_width'] = stroke_width
    return colors_dict

def update_highlight_stroke_widths(highlight_options, tree_index, highlight_stroke_width):
    """
    Updates the stroke widths for highlighted options in the tree index.
    """
    for option in highlight_options:
        if option in tree_index:
            tree_index[option]['stroke_width'] = highlight_stroke_width