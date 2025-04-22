"""
TreeViz: A library for visualizing phylogenetic trees as SVG.

This module provides the main API for generating SVG visualizations
of phylogenetic trees with various layout options.
The code has been modularized for better maintainability and clarity.
"""

import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional, Set, Union, Any
from brancharchitect.tree import Node
from brancharchitect.plot.tree_utils import (
    collapse_zero_length_branches,
    tree_depth,
    calculate_node_depths,
    prepare_highlight_edges,
)
from brancharchitect.plot.svg_utils import create_svg_root, svg_to_string
from brancharchitect.plot.layout import calculate_layout
from brancharchitect.plot.paper_plot.paper_plot_constants import (
    QUANTA_COLORS,
    LIGHT_COLORS,
    MATERIAL_LIGHT_COLORS,
    MATERIAL_DARK_COLORS,
    NATURE_MD3_COLORS,
    MD3_SCIENTIFIC_LIGHT,
    MD3_SCIENTIFIC_DARK,
    MD3_SCIENTIFIC_PRINT,
    DEFAULT_H_SPACING_PER_LEAF,
    DEFAULT_V_SPACING_PER_DEPTH,
    DEFAULT_LABEL_AREA_HEIGHT,
    CAPTION_HEIGHT,
    INTER_TREE_SPACING,
    MARGIN_X,
    MARGIN_Y,
    FOOTER_PADDING,
    FOOTER_FONT_SIZE_ADD,
)
from brancharchitect.plot.tree_renderer import render_tree
from brancharchitect.plot.paper_plot.enclosure_renderer import render_enclosures
from brancharchitect.plot.paper_plot.overlay_labels import add_footer_text
from brancharchitect.plot.paper_plot.save_utils import save_to_pdf
from brancharchitect.plot.paper_plot.overlay_labels import (
    render_node_labels,
    add_caption,
)
from brancharchitect.plot.paper_plot.overlay_labels import (
    add_tree_label,
    add_mathjax_to_svg
)


# -----------------------------------------------------------------------------
# A. LaTeX Handling Functions
# -----------------------------------------------------------------------------
def check_latex_rendering_mode(
    enable_latex: bool, use_mathjax: bool, output_pdf: Optional[str]
) -> None:
    """
    Check and notify about LaTeX rendering mode.

    Args:
        enable_latex: Whether LaTeX rendering is enabled
        use_mathjax: Whether to use MathJax for browser-based rendering
        output_pdf: Path to output PDF if applicable
    """
    if not enable_latex:
        return

    if use_mathjax:
        print(
            "LaTeX rendering mode: MathJax (browser-based) - Formulas will be rendered in your browser."
        )
        if output_pdf:
            print(
                "Warning: PDF saved to",
                output_pdf,
                "will not have rendered LaTeX, as browser-based MathJax rendering is selected.",
            )
    else:
        print(
            "LaTeX rendering mode: Server-side - Formulas will be rendered during SVG generation."
        )
        if output_pdf:
            print(
                "Note: PDF saved to",
                output_pdf,
                "will include pre-rendered LaTeX formulas.",
            )



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


def apply_manual_stroke_width(
    stroke_width: Optional[float],
    colors_dict: Dict[str, Any],
    highlight_options: Optional[Union[List[Optional[Dict]], Dict]],
) -> Tuple[Dict[str, Any], Optional[Union[List[Optional[Dict]], Dict]]]:
    """
    Apply manually specified stroke width to colors and highlight options.

    Args:
        stroke_width: Manually specified stroke width
        colors_dict: Dictionary of colors
        highlight_options: Options for highlighting

    Returns:
        Tuple of updated colors dictionary and updated highlight options
    """
    if stroke_width is None:
        return colors_dict, highlight_options

    # Apply manual stroke width
    colors_dict["default_stroke_width"] = str(stroke_width)

    # Also apply to all highlight options if they don't explicitly define a stroke width
    if highlight_options:
        if isinstance(highlight_options, list):
            for opt in highlight_options:
                if opt and "stroke_width" not in opt:
                    opt["stroke_width"] = str(stroke_width)
        elif (
            isinstance(highlight_options, dict)
            and "stroke_width" not in highlight_options
        ):
            highlight_options["stroke_width"] = str(stroke_width)

    # Return updated dict and options
    return colors_dict, highlight_options


def update_highlight_stroke_widths(
    highlight_options: Optional[Union[List[Optional[Dict]], Dict]],
    tree_index: int,
    highlight_stroke_width: float,
) -> Optional[Union[List[Optional[Dict]], Dict]]:
    """
    Update highlight options with calculated stroke widths.

    Args:
        highlight_options: Options for highlighting
        tree_index: Index of the current tree
        highlight_stroke_width: Calculated stroke width for highlighting

    Returns:
        Updated highlight options
    """
    if not highlight_options:
        return highlight_options

    if isinstance(highlight_options, list) and tree_index < len(highlight_options):
        if highlight_options[tree_index]:
            # Update stroke width in this tree's highlight options
            highlight_options[tree_index]["stroke_width"] = str(highlight_stroke_width)
    elif isinstance(highlight_options, dict):
        highlight_options["stroke_width"] = str(highlight_stroke_width)

    return highlight_options


# -----------------------------------------------------------------------------
# C. Input Processing Functions
# -----------------------------------------------------------------------------
def normalize_input_options(
    num_trees: int,
    highlight_options: Optional[Union[List[Optional[Dict]], Dict]] = None,
    enclose_subtrees: Optional[Union[List[Optional[Dict]], Dict]] = None,
    footer_texts: Optional[Union[str, List[Optional[str]]]] = None,
    node_labels: Optional[
        Union[List[Optional[Dict[str, Dict]]], Dict[str, Dict]]
    ] = None,
    cut_edges: Optional[
        Union[List[Optional[Set[Tuple[str, str]]]], Set[Tuple[str, str]]]
    ] = None,
) -> Tuple[
    List[Optional[Dict]],
    List[Optional[Dict]],
    List[Optional[str]],
    List[Optional[Dict[str, Dict]]],
    List[Optional[Set[Tuple[str, str]]]],
]:
    """
    Normalize input options for multiple trees.

    Args:
        num_trees: Number of trees to process
        highlight_options: Options for highlighting branches and leaves
        enclose_subtrees: Options for enclosing subtrees
        footer_texts: Footer texts for trees
        node_labels: Node labels for trees
        cut_edges: Set of edges to cut in each tree

    Returns:
        Tuple of normalized lists for each option type
    """
    tree_highlights = normalize_highlight_options(num_trees, highlight_options)
    tree_enclosures = normalize_enclosure_options(num_trees, enclose_subtrees)
    tree_footers = normalize_footer_texts(num_trees, footer_texts)
    tree_node_labels = normalize_node_labels(num_trees, node_labels)
    tree_cut_edges = normalize_cut_edges(num_trees, cut_edges)

    return (
        tree_highlights,
        tree_enclosures,
        tree_footers,
        tree_node_labels,
        tree_cut_edges,
    )


def normalize_highlight_options(
    num_trees: int, highlight_options: Optional[Union[List[Optional[Dict]], Dict]]
) -> List[Optional[Dict]]:
    """
    Normalize highlight options for multiple trees.

    Args:
        num_trees: Number of trees to process
        highlight_options: Options for highlighting branches and leaves

    Returns:
        Normalized list of highlight options
    """
    tree_highlights = [None] * num_trees

    if isinstance(highlight_options, list):
        for i, opt in enumerate(highlight_options[:num_trees]):
            if opt:
                # Process edge IDs to ensure proper format
                processed_opt = opt.copy()
                if "edges" in processed_opt:
                    processed_opt["edges"] = prepare_highlight_edges(
                        processed_opt["edges"]
                    )
                tree_highlights[i] = processed_opt
            else:
                tree_highlights[i] = opt
    elif isinstance(highlight_options, dict) and num_trees > 0:
        # Process edge IDs to ensure proper format
        processed_opt = highlight_options.copy()
        if "edges" in processed_opt:
            processed_opt["edges"] = prepare_highlight_edges(processed_opt["edges"])
        tree_highlights[0] = processed_opt

    return tree_highlights


def normalize_enclosure_options(
    num_trees: int, enclose_subtrees: Optional[Union[List[Optional[Dict]], Dict]]
) -> List[Optional[Dict]]:
    """
    Normalize enclosure options for multiple trees.

    Args:
        num_trees: Number of trees to process
        enclose_subtrees: Options for enclosing subtrees

    Returns:
        Normalized list of enclosure options
    """
    tree_enclosures = [None] * num_trees

    if isinstance(enclose_subtrees, list):
        for i, opt in enumerate(enclose_subtrees[:num_trees]):
            tree_enclosures[i] = opt
    elif isinstance(enclose_subtrees, dict) and num_trees > 0:
        tree_enclosures[0] = enclose_subtrees

    return tree_enclosures


def normalize_footer_texts(
    num_trees: int, footer_texts: Optional[Union[str, List[Optional[str]]]]
) -> List[Optional[str]]:
    """
    Normalize footer texts for multiple trees.

    Args:
        num_trees: Number of trees to process
        footer_texts: Footer texts for trees

    Returns:
        Normalized list of footer texts
    """
    tree_footers = [None] * num_trees

    if isinstance(footer_texts, list):
        for i, text in enumerate(footer_texts[:num_trees]):
            tree_footers[i] = text
    elif isinstance(footer_texts, str) and num_trees > 0:
        tree_footers[0] = footer_texts

    return tree_footers


def normalize_node_labels(
    num_trees: int,
    node_labels: Optional[Union[List[Optional[Dict[str, Dict]]], Dict[str, Dict]]],
) -> List[Optional[Dict[str, Dict]]]:
    """
    Normalize node labels for multiple trees.

    Args:
        num_trees: Number of trees to process
        node_labels: Node labels for trees

    Returns:
        Normalized list of node labels
    """
    tree_node_labels = [None] * num_trees

    if isinstance(node_labels, list):
        for i, labels in enumerate(node_labels[:num_trees]):
            tree_node_labels[i] = labels
    elif isinstance(node_labels, dict) and num_trees > 0:
        tree_node_labels[0] = node_labels

    return tree_node_labels


def normalize_cut_edges(
    num_trees: int,
    cut_edges: Optional[
        Union[List[Optional[Set[Tuple[str, str]]]], Set[Tuple[str, str]]]
    ],
) -> List[Optional[Set[Tuple[str, str]]]]:
    """
    Normalize cut edges for multiple trees.

    Args:
        num_trees: Number of trees to process
        cut_edges: Set of edges to cut in each tree

    Returns:
        Normalized list of cut edges
    """
    tree_cut_edges = [None] * num_trees

    if isinstance(cut_edges, list):
        for i, edges in enumerate(cut_edges[:num_trees]):
            if edges:
                tree_cut_edges[i] = prepare_highlight_edges(edges)
            else:
                tree_cut_edges[i] = edges
    elif isinstance(cut_edges, set) and num_trees > 0:
        tree_cut_edges[0] = prepare_highlight_edges(cut_edges)

    return tree_cut_edges


# -----------------------------------------------------------------------------
# D. Node Depth Functions
# -----------------------------------------------------------------------------
def calculate_all_node_depths(
    roots: List[Node],
) -> Tuple[List[Dict[Node, float]], List[float]]:
    """
    Calculate node depths for all trees.

    Args:
        roots: List of root nodes

    Returns:
        Tuple of all node depths and maximum tree depths
    """
    all_node_depths = []
    max_tree_depths = []

    for root in roots:
        # Calculate and store depths for this tree
        depth_dict = calculate_node_depths(root)
        all_node_depths.append(depth_dict)
        max_tree_depths.append(max(depth_dict.values()) if depth_dict else 0)

    return all_node_depths, max_tree_depths


def inject_depth_information(
    roots: List[Node], layouts: List[Dict], all_node_depths: List[Dict[Node, float]]
) -> None:
    """
    Inject depth information into node coordinates.

    Args:
        roots: List of root nodes
        layouts: List of layout information
        all_node_depths: List of node depths for each tree
    """
    for i, (root, depth_dict) in enumerate(zip(roots, all_node_depths)):
        for node, coords in layouts[i]["coords"].items():
            coords["depth"] = depth_dict.get(node, 0)


# -----------------------------------------------------------------------------
# E. Layout Functions
# -----------------------------------------------------------------------------
def calculate_tree_layouts(
    roots: List[Node],
    layout_type: str,
    h_spacing: float,
    target_height: Optional[int],
    v_spacing: float,
    leaf_padding_top: float = 0.0,
) -> Tuple[List[Dict], float, float, List[float]]:
    """
    Calculate layout information for all trees.

    Args:
        roots: List of root nodes
        layout_type: 'phylogram' or 'cladogram'
        h_spacing: Horizontal spacing between leaf nodes
        target_height: Target height for each tree
        v_spacing: Vertical spacing per depth level
        leaf_padding_top: Top padding for leaves (vertical offset for all y-coordinates)

    Returns:
        Tuple containing layouts information, max height, total width, and all label Y positions
    """
    # Calculate target height if not provided
    target_height = calculate_target_height(
        roots, layout_type, target_height, v_spacing
    )

    # Calculate individual tree layouts
    layouts = []
    max_height = 0
    total_width = 0
    all_label_y = []

    for i, root in enumerate(roots):
        layout_result = calculate_single_tree_layout(
            i, root, layout_type, h_spacing, target_height, leaf_padding_top
        )

        if layout_result is None:
            continue

        layout, label_y, tree_width, tree_height = layout_result

        # Store layout information
        layouts.append(layout)

        # Update dimensions
        max_height = max(max_height, tree_height)
        all_label_y.append(label_y)
        total_width += tree_width

        if i > 0:
            total_width += INTER_TREE_SPACING

    return layouts, max_height, total_width, all_label_y


def calculate_target_height(
    roots: List[Node], layout_type: str, target_height: Optional[int], v_spacing: float
) -> Optional[int]:
    """
    Calculate target height for trees if not provided.

    Args:
        roots: List of root nodes
        layout_type: 'phylogram' or 'cladogram'
        target_height: Provided target height
        v_spacing: Vertical spacing per depth level

    Returns:
        Calculated target height
    """
    if target_height is not None or layout_type != "phylogram":
        return target_height

    max_depth = max(tree_depth(root) for root in roots)
    return max(150, int(max_depth * v_spacing + DEFAULT_LABEL_AREA_HEIGHT))


def calculate_single_tree_layout(
    tree_index: int,
    root: Node,
    layout_type: str,
    h_spacing: float,
    target_height: Optional[int],
    leaf_padding_top: float,
) -> Optional[Tuple[Dict, float, float, float]]:
    """
    Calculate layout for a single tree.

    Args:
        tree_index: Index of the tree
        root: Root node of the tree
        layout_type: 'phylogram' or 'cladogram'
        h_spacing: Horizontal spacing between leaf nodes
        target_height: Target height for the tree
        leaf_padding_top: Top padding for leaves

    Returns:
        Tuple containing layout, label_y, tree_width, and tree_height if successful,
        None if there's an error
    """
    try:
        # Collapse zero-length branches
        processed_root = collapse_zero_length_branches(root)

        # Calculate layout - use provided h_spacing and target_height
        node_coords, label_y, tree_width, tree_height = calculate_layout(
            processed_root,
            layout_type,
            h_spacing,
            target_height,
            leaf_padding_top=leaf_padding_top,
        )

        # Create layout information
        layout = {
            "processed_root": processed_root,
            "coords": node_coords,
            "label_y": label_y,
            "width": tree_width,
            "height": tree_height,
        }

        return layout, label_y, tree_width, tree_height

    except Exception as e:
        print(f"Error processing tree {tree_index}: {e}")
        import traceback

        traceback.print_exc()
        raise ValueError(f"Error during layout of tree {tree_index}: {e}")


def adjust_layouts_for_shared_labels(
    layouts: List[Dict],
    all_label_y: List[float],
    use_shared_label_y: bool,
    label_area_height: float,
) -> float:
    """
    Adjust layouts if using shared labels across trees.

    Args:
        layouts: List of layout information dictionaries
        all_label_y: List of label Y positions
        use_shared_label_y: Whether to align labels across trees
        label_area_height: Height reserved for labels

    Returns:
        Updated maximum height
    """
    max_height = max(layout["height"] for layout in layouts)

    if use_shared_label_y and all_label_y:
        shared_label_y = max(all_label_y)
        for layout in layouts:
            layout["label_y"] = shared_label_y

        max_height = max(max_height, shared_label_y + (label_area_height - 25))

    return max_height


def calculate_footer_height(
    tree_footers: List[Optional[str]], colors: Dict, enable_latex: bool
) -> float:
    """
    Calculate the total footer height needed based on footer content.

    Args:
        tree_footers: List of footer texts for each tree
        colors: Color and style information
        enable_latex: Whether LaTeX rendering is enabled

    Returns:
        Calculated footer height
    """
    if not any(tree_footers):
        return 0

    # Get font size and calculate line height
    font_size_num = get_font_size_from_colors(colors)
    line_height = font_size_num * 1.2

    # Count maximum number of lines in footers
    max_footer_lines = count_max_footer_lines(tree_footers)

    if max_footer_lines > 0:
        return calculate_footer_height_from_lines(
            max_footer_lines, line_height, enable_latex
        )

    return 0


def get_font_size_from_colors(colors: Dict) -> float:
    """
    Extract font size from colors dictionary.

    Args:
        colors: Color and style information

    Returns:
        Font size as a float
    """
    try:
        font_size_num = float(colors.get("font_size", 10)) + FOOTER_FONT_SIZE_ADD
    except ValueError:
        font_size_num = 10.0

    return font_size_num


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
    max_footer_lines: int, line_height: float, enable_latex: bool
) -> float:
    """
    Calculate footer height based on number of lines.

    Args:
        max_footer_lines: Maximum number of lines in footers
        line_height: Height of each line
        enable_latex: Whether LaTeX rendering is enabled

    Returns:
        Calculated footer height
    """
    # Add extra buffer for potentially taller LaTeX elements
    estimated_line_height = line_height * 1.5 if enable_latex else line_height
    footer_height = (
        FOOTER_PADDING + (estimated_line_height * max_footer_lines) + FOOTER_PADDING
    )

    # Add extra space for rendering method info text
    if enable_latex:
        footer_height += line_height * 0.8

    return footer_height


def calculate_svg_dimensions(
    total_width: float,
    max_height: float,
    footer_height: float,
    caption: Optional[str],
    margin_left: Optional[float],
    margin_right: Optional[float],
    margin_top: Optional[float],
    margin_bottom: Optional[float],
) -> Tuple[float, float]:
    """
    Calculate final SVG dimensions.

    Args:
        total_width: Total width of all trees
        max_height: Maximum height of all trees
        footer_height: Height needed for footers
        caption: Optional caption text
        margin_left: Custom left margin
        margin_right: Custom right margin
        margin_top: Custom top margin
        margin_bottom: Custom bottom margin

    Returns:
        Tuple containing SVG width and height
    """
    # Calculate caption height
    caption_height = (CAPTION_HEIGHT + 10) if caption else 0

    # Apply custom margins if provided
    left = margin_left if margin_left is not None else MARGIN_X
    right = margin_right if margin_right is not None else MARGIN_X
    top = margin_top if margin_top is not None else MARGIN_Y
    bottom = margin_bottom if margin_bottom is not None else MARGIN_Y

    # Calculate final dimensions
    svg_width = total_width + left + right
    svg_height = max_height + caption_height + footer_height + top + bottom

    return svg_width, svg_height


# -----------------------------------------------------------------------------
# F. SVG Element Creation Functions
# -----------------------------------------------------------------------------
def setup_svg_root(
    svg_width: float,
    svg_height: float,
    colors: Dict,
    enable_latex: bool,
    use_mathjax: bool,
) -> ET.Element:
    """
    Create and setup the SVG root element.

    Args:
        svg_width: Width of the SVG
        svg_height: Height of the SVG
        colors: Color and style information
        enable_latex: Whether LaTeX rendering is enabled
        use_mathjax: Whether to use MathJax for browser-based rendering

    Returns:
        Configured SVG root element
    """
    # Create SVG root
    svg_root = create_svg_root(svg_width, svg_height)

    # Add MathJax script to SVG root if browser-based rendering is enabled
    if enable_latex and use_mathjax:
        add_mathjax_to_svg(svg_root)

    # Add background if color specified
    add_background_to_svg(svg_root, colors)

    return svg_root


def add_background_to_svg(svg_root: ET.Element, colors: Dict) -> None:
    """
    Add background to SVG if specified in colors.

    Args:
        svg_root: SVG root element
        colors: Color and style information
    """
    if colors.get("background"):
        ET.SubElement(
            svg_root,
            "rect",
            {"width": "100%", "height": "100%", "fill": colors["background"]},
        )


def create_container_group(
    svg_root: ET.Element, margin_left: float, margin_top: float
) -> ET.Element:
    """
    Create container group with appropriate margins.

    Args:
        svg_root: SVG root element
        margin_left: Left margin
        margin_top: Top margin

    Returns:
        Container group element
    """
    return ET.SubElement(
        svg_root, "g", {"transform": f"translate({margin_left}, {margin_top})"}
    )


# -----------------------------------------------------------------------------
# G. Tree Rendering Functions
# -----------------------------------------------------------------------------
def render_single_tree(
    container: ET.Element,
    layout: Dict,
    x_offset: float,
    color_mode: str,
    highlight_options: Optional[Dict],
    split_options: Optional[Dict],
    enclose_options: Optional[Dict],
    node_labels_options: Optional[Dict[str, Dict]],
    footer_text: Optional[str],
    colors: Dict,
    leaf_font_size: Optional[float],
    enable_latex: bool,
    use_mathjax: bool,
    latex_scale: float,
    footer_font_size: Optional[float],
    tree_label: Optional[str] = None,
    tree_label_font_size: Optional[float] = None,
    tree_label_style: Optional[Dict] = None,
    leaf_label_offset: float = 0.0,
) -> None:
    """
    Render a single tree with all its options.

    Args:
        container: Parent SVG element
        layout: Layout information for this tree
        x_offset: X offset for positioning
        color_mode: Color scheme name
        highlight_options: Options for highlighting branches and leaves
        split_options: Options for splitting trees
        enclose_options: Options for enclosing subtrees
        node_labels_options: Options for node labels
        footer_text: Footer text for this tree
        colors: Color and style information
        leaf_font_size: Custom font size for leaf labels
        enable_latex: Whether LaTeX rendering is enabled
        use_mathjax: Whether to use MathJax for browser-based rendering
        latex_scale: Scaling factor for LaTeX elements
        footer_font_size: Custom font size for footer text
        tree_label: Label for the tree (e.g., "A", "B", "C")
        tree_label_font_size: Font size for the tree label
        tree_label_style: Custom style for the tree label
        leaf_label_offset: Offset for leaf labels
    """
    # Create group for this tree
    tree_group = create_tree_group(container, x_offset)

    # Calculate visual padding based on tree properties
    visual_padding = calculate_visual_padding(colors, highlight_options)

    # Add tree label if provided
    if tree_label:
        add_tree_label(
            tree_group,
            tree_label,
            tree_label_font_size,
            leaf_font_size,
            tree_label_style,
            enable_latex,
            use_mathjax,
            latex_scale,
        )
        # Account for tree label in visual padding (for top margin)
        visual_padding = max(visual_padding, 30)  # Tree labels need extra space above

    try:
        # Update layout dimensions to account for visual elements
        update_layout_dimensions(layout, visual_padding)

        # Render the tree components
        render_tree_components(
            tree_group,
            layout,
            color_mode,
            highlight_options,
            split_options,
            enclose_options,
            node_labels_options,
            footer_text,
            colors,
            leaf_font_size,
            enable_latex,
            use_mathjax,
            latex_scale,
            footer_font_size,
            leaf_label_offset,
        )

    except Exception as e:
        print(f"Error rendering tree: {e}")
        ET.SubElement(tree_group, "text", {"fill": "red"}).text = f"Render Error: {e}"


def create_tree_group(container: ET.Element, x_offset: float) -> ET.Element:
    """
    Create a group for a single tree.

    Args:
        container: Parent SVG element
        x_offset: X offset for positioning

    Returns:
        Tree group element
    """
    return ET.SubElement(container, "g", {"transform": f"translate({x_offset:.2f}, 0)"})


def calculate_visual_padding(colors: Dict, highlight_options: Optional[Dict]) -> float:
    """
    Calculate padding needed for visual elements.

    Args:
        colors: Color and style information
        highlight_options: Options for highlighting

    Returns:
        Calculated visual padding
    """
    # Get the node marker size which might extend beyond coordinates
    node_marker_size = float(colors.get("node_marker_size", 4.5))

    # Get max stroke width for highlights which might extend beyond coordinates
    highlight_stroke_width = 0
    if highlight_options:
        highlight_stroke_width = float(
            highlight_options.get(
                "stroke_width", colors.get("highlight_stroke_width", 2.5)
            )
        )

    # Calculate padding needed based on visual elements
    visual_padding = max(node_marker_size * 2, highlight_stroke_width * 3)

    # Account for possible blob effects which extend further
    if highlight_options and (
        highlight_options.get("use_blobs", False)
        or highlight_options.get("use_blob", False)
    ):
        visual_padding = max(visual_padding, 30)  # Blob effects can extend up to ~30px

    # Account for waterbrush effects
    if highlight_options and highlight_options.get("use_waterbrush", False):
        visual_padding = max(visual_padding, 20)  # Waterbrush effects can extend ~20px

    # Account for elevation effects (shadows)
    if highlight_options and highlight_options.get("use_elevation", False):
        visual_padding += 10  # Shadows can add ~10px

    return visual_padding


def update_layout_dimensions(layout: Dict, visual_padding: float) -> None:
    """
    Update layout dimensions to account for visual elements.

    Args:
        layout: Layout information
        visual_padding: Padding for visual elements
    """
    # Store original dimensions
    original_width = layout["width"]
    original_height = layout["height"]

    # Expand dimensions to account for visual elements
    layout["effective_width"] = original_width + visual_padding * 2
    layout["effective_height"] = original_height + visual_padding * 2


def render_tree_components(
    tree_group: ET.Element,
    layout: Dict,
    color_mode: str,
    highlight_options: Optional[Dict],
    split_options: Optional[Dict],
    enclose_options: Optional[Dict],
    node_labels_options: Optional[Dict[str, Dict]],
    footer_text: Optional[str],
    colors: Dict,
    leaf_font_size: Optional[float],
    enable_latex: bool,
    use_mathjax: bool,
    latex_scale: float,
    footer_font_size: Optional[float],
    leaf_label_offset: float,
) -> None:
    """
    Render all components of a tree.

    Args:
        tree_group: Tree group element
        layout: Layout information
        color_mode: Color scheme name
        highlight_options: Options for highlighting
        split_options: Options for splitting
        enclose_options: Options for enclosing subtrees
        node_labels_options: Options for node labels
        footer_text: Footer text
        colors: Color and style information
        leaf_font_size: Font size for leaf labels
        enable_latex: Whether LaTeX rendering is enabled
        use_mathjax: Whether to use MathJax
        latex_scale: Scaling factor for LaTeX
        footer_font_size: Font size for footer text
        leaf_label_offset: Offset for leaf labels
    """
    # Render the tree
    render_tree(
        tree_group,
        layout["coords"],
        layout["label_y"],
        color_mode,
        highlight_options,
        split_options,
        leaf_font_size,
        cut_edges=highlight_options.get("cut_edges") if highlight_options else None,
        leaf_label_offset=leaf_label_offset,
    )

    # Render enclosures if specified
    if enclose_options:
        render_enclosures(
            tree_group,
            layout,
            enclose_options,
            colors,
            leaf_label_offset=leaf_label_offset,
        )

        update_layout_for_enclosures(layout, enclose_options)

    # Render node labels if specified
    if node_labels_options:
        render_node_labels(tree_group, layout["coords"], colors, node_labels_options)
        update_layout_for_node_labels(layout, node_labels_options)

    # Render footer text if specified
    if footer_text:
        add_footer_text(
            tree_group,
            footer_text,
            layout["height"],  # Pass the actual tree drawing height
            layout["width"],
            colors,
            enable_latex,
            use_mathjax,
            latex_scale,
            footer_font_size=footer_font_size,
        )


def update_layout_for_enclosures(layout: Dict, enclose_options: Dict) -> None:
    """
    Update layout dimensions to account for enclosures.

    Args:
        layout: Layout information
        enclose_options: Options for enclosing subtrees
    """
    # Enclosures might extend beyond the basic dimensions
    # Check if any enclosure styles specify padding
    enclosure_padding = calculate_enclosure_padding(enclose_options)

    # Add enclosure padding to the effective dimensions
    if enclosure_padding > 0:
        layout["effective_width"] += enclosure_padding * 2
        layout["effective_height"] += enclosure_padding * 2


def calculate_enclosure_padding(enclose_options: Dict) -> float:
    """
    Calculate maximum padding from enclosure options.

    Args:
        enclose_options: Options for enclosing subtrees

    Returns:
        Maximum padding value
    """
    enclosure_padding = 0
    for style in enclose_options.values():
        if isinstance(style, dict):
            padding = style.get("padding", 0)
            try:
                padding = float(padding)
                enclosure_padding = max(enclosure_padding, padding)
            except (ValueError, TypeError):
                pass

    return enclosure_padding


def update_layout_for_node_labels(
    layout: Dict, node_labels_options: Dict[str, Dict]
) -> None:
    """
    Update layout dimensions to account for node labels.

    Args:
        layout: Layout information
        node_labels_options: Options for node labels
    """
    # Calculate max offset of node labels
    max_label_offset = calculate_max_label_offset(node_labels_options)

    # Add label offset to the effective width
    if max_label_offset > 0:
        layout["effective_width"] += max_label_offset


def calculate_max_label_offset(node_labels_options: Dict[str, Dict]) -> float:
    """
    Calculate maximum offset from node label options.

    Args:
        node_labels_options: Options for node labels

    Returns:
        Maximum label offset value
    """
    max_label_offset = 0
    for label_info in node_labels_options.values():
        if isinstance(label_info, dict):
            offset = abs(float(label_info.get("offset", 0)))
            max_label_offset = max(max_label_offset, offset)

    return max_label_offset


def render_all_trees(
    container: ET.Element,
    layouts: List[Dict],
    color_mode: str,
    tree_highlights: List[Optional[Dict]],
    split_options: Optional[Dict],
    tree_enclosures: List[Optional[Dict]],
    tree_node_labels: List[Optional[Dict[str, Dict]]],
    tree_footers: List[Optional[str]],
    colors: Dict,
    leaf_font_size: Optional[float],
    enable_latex: bool,
    use_mathjax: bool,
    latex_scale: float,
    footer_font_size: Optional[float],
    tree_labels: Optional[List[str]] = None,
    tree_label_font_size: Optional[float] = None,
    tree_label_style: Optional[Dict] = None,
    leaf_label_offset: float = 0.0,
) -> None:
    """
    Render all trees with their respective options.

    Args:
        container: Parent SVG element
        layouts: List of layout information dictionaries
        color_mode: Color scheme name
        tree_highlights: Options for highlighting branches and leaves for each tree
        split_options: Options for splitting trees
        tree_enclosures: Options for enclosing subtrees for each tree
        tree_node_labels: Options for node labels for each tree
        tree_footers: Footer texts for each tree
        colors: Color and style information
        leaf_font_size: Custom font size for leaf labels
        enable_latex: Whether LaTeX rendering is enabled
        use_mathjax: Whether to use MathJax for browser-based rendering
        latex_scale: Scaling factor for LaTeX elements
        footer_font_size: Custom font size for footer text
        tree_labels: List of labels for each tree (e.g., ["A", "B", "C"])
        tree_label_font_size: Font size for tree labels
        tree_label_style: Custom style for tree labels
        leaf_label_offset: Offset for leaf labels
    """
    x_offset = 0
    for i, layout in enumerate(layouts):
        render_single_tree(
            container,
            layout,
            x_offset,
            color_mode,
            tree_highlights[i] if i < len(tree_highlights) else None,
            split_options,
            tree_enclosures[i] if i < len(tree_enclosures) else None,
            tree_node_labels[i] if i < len(tree_node_labels) else None,
            tree_footers[i] if i < len(tree_footers) else None,
            colors,
            leaf_font_size,
            enable_latex,
            use_mathjax,
            latex_scale,
            footer_font_size,
            tree_label=tree_labels[i] if tree_labels and i < len(tree_labels) else None,
            tree_label_font_size=tree_label_font_size,
            tree_label_style=tree_label_style,
            leaf_label_offset=leaf_label_offset,
        )

        # Update x offset for next tree
        x_offset += layout["width"] + INTER_TREE_SPACING


# -----------------------------------------------------------------------------
# H. Main Functions
# -----------------------------------------------------------------------------
def generate_tree_svg(
    roots: List[Node],
    layout_type: str = "phylogram",
    color_mode: str = "quanta",
    highlight_options: Optional[Union[List[Optional[Dict]], Dict]] = None,
    split_options: Optional[Dict] = None,
    enclose_subtrees: Optional[Union[List[Optional[Dict]], Dict]] = None,
    caption: Optional[str] = None,
    footer_texts: Optional[Union[str, List[Optional[str]]]] = None,
    enable_latex: bool = False,
    use_mathjax: bool = False,
    latex_scale: float = 1.0,
    target_height: Optional[int] = None,
    use_shared_label_y: bool = True,
    h_spacing: float = DEFAULT_H_SPACING_PER_LEAF,
    v_spacing: float = DEFAULT_V_SPACING_PER_DEPTH,
    label_area_height: float = DEFAULT_LABEL_AREA_HEIGHT,
    output_pdf: Optional[str] = None,
    node_labels: Optional[
        Union[List[Optional[Dict[str, Dict]]], Dict[str, Dict]]
    ] = None,
    colors: Optional[Dict] = None,
    leaf_font_size: Optional[float] = None,
    footer_font_size: Optional[float] = None,
    caption_font_size: Optional[float] = None,
    stroke_style: str = "balanced",
    stroke_width: Optional[float] = None,
    tree_labels: Optional[List[str]] = None,
    tree_label_font_size: Optional[float] = None,
    tree_label_style: Optional[Dict] = None,
    cut_edges: Optional[
        Union[List[Optional[Set[Tuple[str, str]]]], Set[Tuple[str, str]]]
    ] = None,
    margin_left: Optional[float] = None,
    margin_right: Optional[float] = None,
    margin_top: Optional[float] = None,
    margin_bottom: Optional[float] = None,
    leaf_padding_top: float = 0.0,
    leaf_label_offset: float = 0.0,
) -> str:
    """
    Generate SVG for multiple trees with enhanced options.
    """

    # Check and notify about LaTeX rendering mode (uses modified check function)
    check_latex_rendering_mode(enable_latex, use_mathjax, output_pdf)

    try:
        # Call the implementation function (assuming it exists and is correct)
        # Make sure the implementation function also uses the modified add_tree_label etc.
        return generate_tree_svg_implementation(
            roots,
            layout_type,
            color_mode,
            highlight_options,
            split_options,
            enclose_subtrees,
            caption,
            footer_texts,
            enable_latex,
            use_mathjax,
            latex_scale,
            target_height,
            use_shared_label_y,
            h_spacing,
            v_spacing,
            label_area_height,
            output_pdf,
            node_labels,
            colors,
            leaf_font_size,
            footer_font_size,
            caption_font_size,
            stroke_style,
            stroke_width,
            tree_labels,
            tree_label_font_size,
            tree_label_style,
            cut_edges,
            margin_left,
            margin_right,
            margin_top,
            margin_bottom,
            leaf_padding_top,
            leaf_label_offset,
        )
    except Exception as e:
        print(f"Error generating tree SVG: {e}")
        import traceback

        traceback.print_exc()
        # Provide a basic SVG error message
        error_svg = ET.Element("svg", width="200", height="50")
        error_text = ET.SubElement(error_svg, "text", x="10", y="25", fill="red")
        error_text.text = f"Error during tree SVG generation: {e}"
        return svg_to_string(error_svg)  # Use svg_to_string helper


def generate_tree_svg_implementation(
    roots: List[Node],
    layout_type: str,
    color_mode: str,
    highlight_options: Optional[Union[List[Optional[Dict]], Dict]],
    split_options: Optional[Dict],
    enclose_subtrees: Optional[Union[List[Optional[Dict]], Dict]],
    caption: Optional[str],
    footer_texts: Optional[Union[str, List[Optional[str]]]],
    enable_latex: bool,
    use_mathjax: bool,
    latex_scale: float,
    target_height: Optional[int],
    use_shared_label_y: bool,
    h_spacing: float,
    v_spacing: float,
    label_area_height: float,
    output_pdf: Optional[str],
    node_labels: Optional[Union[List[Optional[Dict[str, Dict]]], Dict[str, Dict]]],
    colors: Optional[Dict],
    leaf_font_size: Optional[float],
    footer_font_size: Optional[float],
    caption_font_size: Optional[float],
    stroke_style: str,
    stroke_width: Optional[float],
    tree_labels: Optional[List[str]],
    tree_label_font_size: Optional[float],
    tree_label_style: Optional[Dict],
    cut_edges: Optional[
        Union[List[Optional[Set[Tuple[str, str]]]], Set[Tuple[str, str]]]
    ],
    margin_left: Optional[float],
    margin_right: Optional[float],
    margin_top: Optional[float],
    margin_bottom: Optional[float],
    leaf_padding_top: float,
    leaf_label_offset: float,
) -> str:
    """
    Implementation of the tree SVG generation process.

    Args:
        All parameters are the same as in generate_tree_svg

    Returns:
        SVG string
    """
    # Get number of trees
    num_trees = len(roots)

    # Normalize input options
    tree_highlights, tree_enclosures, tree_footers, tree_node_labels, tree_cut_edges = (
        normalize_input_options(
            num_trees=num_trees,
            highlight_options=highlight_options,
            enclose_subtrees=enclose_subtrees,
            footer_texts=footer_texts,
            node_labels=node_labels,
            cut_edges=cut_edges,
        )
    )

    # Select color scheme
    colors_dict = select_color_scheme(color_mode=color_mode, custom_colors=colors)

    # Apply manual stroke width if provided
    colors_dict, highlight_options = apply_manual_stroke_width(
        stroke_width=stroke_width,
        colors_dict=colors_dict,
        highlight_options=highlight_options,
    )

    # Calculate node depths for all trees
    all_node_depths, max_tree_depths = calculate_all_node_depths(roots)

    # Calculate layouts for all trees
    layouts, max_height, total_width, all_label_y = calculate_tree_layouts(
        roots=roots,
        layout_type=layout_type,
        h_spacing=h_spacing,
        target_height=target_height,
        v_spacing=v_spacing,
        leaf_padding_top=leaf_padding_top,
    )

    # Inject depth information into node coordinates
    inject_depth_information(
        roots=roots, layouts=layouts, all_node_depths=all_node_depths
    )

    # Adjust layouts if using shared labels
    max_height = adjust_layouts_for_shared_labels(
        layouts=layouts,
        all_label_y=all_label_y,
        use_shared_label_y=use_shared_label_y,
        label_area_height=label_area_height,
    )

    # Calculate footer height
    footer_height = calculate_footer_height(
        tree_footers=tree_footers, colors=colors_dict, enable_latex=enable_latex
    )

    # Calculate SVG dimensions
    svg_width, svg_height = calculate_svg_dimensions(
        total_width=total_width,
        max_height=max_height,
        footer_height=footer_height,
        caption=caption,
        margin_left=margin_left,
        margin_right=margin_right,
        margin_top=margin_top,
        margin_bottom=margin_bottom,
    )

    # Get margins
    left = margin_left if margin_left is not None else MARGIN_X
    top = margin_top if margin_top is not None else MARGIN_Y

    # Create and setup SVG root
    svg_root = setup_svg_root(
        svg_width=svg_width,
        svg_height=svg_height,
        colors=colors_dict,
        enable_latex=enable_latex,
        use_mathjax=use_mathjax,
    )

    # Create container group with custom margins
    container = create_container_group(
        svg_root=svg_root, margin_left=left, margin_top=top
    )

    # Render all trees
    render_all_trees(
        container=container,
        layouts=layouts,
        color_mode=color_mode,
        tree_highlights=tree_highlights,
        split_options=split_options,
        tree_enclosures=tree_enclosures,
        tree_node_labels=tree_node_labels,
        tree_footers=tree_footers,
        colors=colors_dict,
        leaf_font_size=leaf_font_size,
        enable_latex=enable_latex,
        use_mathjax=use_mathjax,
        latex_scale=latex_scale,
        footer_font_size=footer_font_size,
        tree_labels=tree_labels,
        tree_label_font_size=tree_label_font_size,
        tree_label_style=tree_label_style,
        leaf_label_offset=leaf_label_offset,
    )

    # Add caption if specified
    if caption:
        add_tree_caption(
            container=container,
            caption=caption,
            max_height=max_height,
            total_width=total_width,
            colors=colors_dict,
            caption_font_size=caption_font_size,
        )

    # Convert to string
    svg_string = svg_to_string(svg_root=svg_root)

    # Attempt PDF conversion if requested
    if output_pdf:
        save_to_pdf(
            svg_string=svg_string,
            output_pdf=output_pdf,
            enable_latex=enable_latex,
            use_mathjax=use_mathjax,
        )

    return svg_string


def add_tree_caption(
    container: ET.Element,
    caption: str,
    max_height: float,
    total_width: float,
    colors: Dict,
    caption_font_size: Optional[float],
) -> None:
    """
    Add caption to tree visualization.

    Args:
        container: Container SVG element
        caption: Caption text
        max_height: Maximum height of trees
        total_width: Total width of trees
        colors: Color and style information
        caption_font_size: Custom font size for caption
    """
    caption_y = max_height + CAPTION_HEIGHT / 2
    add_caption(
        container, caption, caption_y, total_width, colors, font_size=caption_font_size
    )


# -----------------------------------------------------------------------------
# H. Main Functions (refactored)
# -----------------------------------------------------------------------------
def render_trees_to_svg(
    roots: List[Node],
    *,
    layout_opts: Optional[Dict[str, Any]] = None,
    style_opts: Optional[Dict[str, Any]] = None,
    highlight_opts: Optional[Union[List, Dict]] = None,
    enclosure_opts: Optional[Union[List, Dict]] = None,
    node_label_opts: Optional[Union[List, Dict]] = None,
    footer_texts: Optional[Union[str, List[str]]] = None,
    caption: Optional[str] = None,
    colors: Optional[Dict] = None,
    latex_opts: Optional[Dict[str, Any]] = None,
    output_opts: Optional[Dict[str, Any]] = None,
) -> str:
    """
    High‑level API for rendering one or more trees to SVG.

    Args:
        roots: List of tree roots (Node instances)
        layout_opts: {
            "type": str,               # "phylogram" or "cladogram"
            "h_spacing": float,
            "v_spacing": float,
            "use_shared_label_y": bool,
            "target_height": Optional[int],
            "leaf_padding_top": float,
            "leaf_label_offset": float,
        }
        style_opts: {
            "color_mode": str,
            "leaf_font_size": float,
            "footer_font_size": float,
            "caption_font_size": float,
            "tree_labels": List[str],
            "tree_label_font_size": float,
            "tree_label_style": Dict,
            "stroke_style": str,
            "stroke_width": float,
        }
        highlight_opts: per‑tree or global highlight config
        enclosure_opts: per‑tree or global enclosure config
        node_label_opts: per‑tree or global node‑label config
        footer_texts: str or list of footer strings
        caption: overall SVG caption
        colors: custom color palette
        latex_opts: {
            "enable": bool,
            "use_mathjax": bool,
            "scale": float,
        }
        output_opts: {
            "pdf_path": Optional[str],
            "margins": {"left": float, "right": float, "top": float, "bottom": float},
        }

    Returns:
        SVG content as a string.
    """
    # 1) fill defaults
    layout = {
        "type": "phylogram",
        "h_spacing": DEFAULT_H_SPACING_PER_LEAF,
        "v_spacing": DEFAULT_V_SPACING_PER_DEPTH,
        "use_shared_label_y": True,
        "target_height": None,
        "leaf_padding_top": 0.0,
        "leaf_label_offset": 0.0,
    }
    if layout_opts:
        layout.update(layout_opts)

    style = {
        "color_mode": "quanta",
        "leaf_font_size": None,
        "footer_font_size": None,
        "caption_font_size": None,
        "tree_labels": None,
        "tree_label_font_size": None,
        "tree_label_style": None,
        "stroke_style": "balanced",
        "stroke_width": None,
    }
    if style_opts:
        style.update(style_opts)

    latex = {"enable": False, "use_mathjax": False, "scale": 1.0}
    if latex_opts:
        latex.update(latex_opts)

    output = {"pdf_path": None, "margins": {}}
    if output_opts:
        output.update(output_opts)

    # 2) unpack everything and call existing implementation
    return generate_tree_svg_implementation(
        roots,
        layout_type=layout["type"],
        color_mode=style["color_mode"],
        highlight_options=highlight_opts,
        split_options=None,
        enclose_subtrees=enclosure_opts,
        caption=caption,
        footer_texts=footer_texts,
        enable_latex=latex["enable"],
        use_mathjax=latex["use_mathjax"],
        latex_scale=latex["scale"],
        target_height=layout["target_height"],
        use_shared_label_y=layout["use_shared_label_y"],
        h_spacing=layout["h_spacing"],
        v_spacing=layout["v_spacing"],
        label_area_height=DEFAULT_LABEL_AREA_HEIGHT,
        output_pdf=output["pdf_path"],
        node_labels=node_label_opts,
        colors=colors,
        leaf_font_size=style["leaf_font_size"],
        footer_font_size=style["footer_font_size"],
        caption_font_size=style["caption_font_size"],
        stroke_style=style["stroke_style"],
        stroke_width=style["stroke_width"],
        tree_labels=style["tree_labels"],
        tree_label_font_size=style["tree_label_font_size"],
        tree_label_style=style["tree_label_style"],
        cut_edges=None,
        margin_left=output["margins"].get("left"),
        margin_right=output["margins"].get("right"),
        margin_top=output["margins"].get("top"),
        margin_bottom=output["margins"].get("bottom"),
        leaf_padding_top=layout["leaf_padding_top"],
        leaf_label_offset=layout["leaf_label_offset"],
    )
