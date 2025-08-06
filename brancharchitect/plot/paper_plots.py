"""
TreeViz: A library for visualizing phylogenetic trees as SVG.

This module provides the main API for generating SVG visualizations
of phylogenetic trees with various layout options.
The code has been modularized for better maintainability and clarity.
"""

import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional, Set, Union, Any
from brancharchitect.tree import Node
from brancharchitect.plot.paper_plot.tree_renderer import render_tree
from brancharchitect.plot.paper_plot.enclosure_renderer import render_enclosures
from brancharchitect.plot.paper_plot.save_utils import save_to_pdf
from brancharchitect.plot.paper_plot.input_utils import normalize_input_options
from brancharchitect.plot.paper_plot.node_overlays import (
    render_node_labels,
)
from brancharchitect.plot.paper_plot.tree_label import (
    add_tree_label,
)
from brancharchitect.plot.paper_plot.paper_plot_constants import (
    DEFAULT_H_SPACING_PER_LEAF,
    DEFAULT_V_SPACING_PER_DEPTH,
    DEFAULT_LABEL_AREA_HEIGHT,
    INTER_TREE_SPACING,
    MARGIN_X,
    MARGIN_Y,
)
from brancharchitect.plot.paper_plot.color_utils import select_color_scheme
from brancharchitect.plot.paper_plot.enclosure_renderer import (
    update_layout_for_enclosures,
)
from brancharchitect.plot.paper_plot.footer import add_footer_text
from brancharchitect.plot.paper_plot.footer import calculate_footer_height
from brancharchitect.plot.paper_plot.save_utils import _svg_to_string
from brancharchitect.plot.layout import create_container_group
from brancharchitect.plot.layout import calculate_svg_dimensions
from brancharchitect.plot.layout import (
    calculate_all_node_depths,
    calculate_tree_layouts,
    inject_depth_information,
    adjust_layouts_for_shared_labels,
    update_layout_for_node_labels,
)
from brancharchitect.plot.paper_plot.save_utils import create_svg_root
from brancharchitect.plot.paper_plot.leaf_connections import (
    render_leaf_connections,
)


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
    latex_scale: float,
    footer_font_size: Optional[float],
    tree_label: Optional[str] = None,
    tree_label_font_size: Optional[float] = None,
    tree_label_style: Optional[Dict] = None,
    leaf_label_offset: float = 0.0,
    branch_labels: Optional[Dict] = None,
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
        latex_scale: Scaling factor for LaTeX elements
        footer_font_size: Custom font size for footer text
        tree_label: Label for the tree (e.g., "A", "B", "C")
        tree_label_font_size: Font size for the tree label
        tree_label_style: Custom style for the tree label
        leaf_label_offset: Offset for leaf labels
        branch_labels: Dict for branch labels for this tree
    """
    # Create group for this tree
    tree_group = create_tree_group(container, x_offset)

    # Calculate visual padding based on tree properties
    visual_padding = calculate_visual_padding(colors, highlight_options)

    print(latex_scale)

    # Add tree label if provided
    if tree_label:
        add_tree_label(
            tree_group=tree_group,
            tree_label=tree_label,
            tree_label_font_size=tree_label_font_size,
            leaf_font_size=leaf_font_size,
            tree_label_style=tree_label_style,
            enable_latex=enable_latex,
            latex_scale=latex_scale,
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
            latex_scale,
            footer_font_size,
            leaf_label_offset,
            branch_labels=branch_labels,
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
    colors: Dict,  # <-- This is the merged colors dict
    leaf_font_size: Optional[float],
    enable_latex: bool,
    latex_scale: float,
    footer_font_size: Optional[float],
    leaf_label_offset: float,
    branch_labels: Optional[Dict] = None,
) -> None:
    """
    Render all components of a tree, ensuring correct layering with explicit groups.

    Args:
        tree_group: The main SVG group for this individual tree.
        layout: Layout information.
        # ... other args ...
        branch_labels: Dict for branch labels for this tree.
    """
    # Create explicit groups for layering within the main tree_group
    background_group = ET.SubElement(
        tree_group, "{http://www.w3.org/2000/svg}g", {"class": "tree-background-layer"}
    )
    foreground_group = ET.SubElement(
        tree_group, "{http://www.w3.org/2000/svg}g", {"class": "tree-foreground-layer"}
    )

    # Render enclosures into the background group
    if enclose_options:
        # Note: render_enclosures itself might create sub-groups, but they will all be within background_group
        render_enclosures(
            background_group,  # Pass the dedicated background group
            layout,
            enclose_options,
            colors,
            leaf_label_offset=leaf_label_offset,
        )
        # Update layout dimensions based on enclosure padding AFTER calculating boundaries
        # This update still happens on the main layout dict, which is fine.
        update_layout_for_enclosures(layout, enclose_options)

    # Render the tree into the foreground group
    render_tree(
        foreground_group,  # Pass the dedicated foreground group
        layout["coords"],
        layout["label_y"],
        color_mode,
        colors=colors,
        highlight_options=highlight_options,
        split_options=split_options,
        leaf_font_size=leaf_font_size,
        cut_edges=highlight_options.get("cut_edges") if highlight_options else None,
        leaf_label_offset=leaf_label_offset,
        branch_labels=branch_labels,
    )

    # Render node labels into the foreground group (on top of the tree)
    if node_labels_options:
        render_node_labels(
            foreground_group,  # Pass the dedicated foreground group
            layout["coords"],
            colors,
            node_labels_options,
        )
        # Update layout dimensions based on node labels AFTER rendering them
        update_layout_for_node_labels(layout, node_labels_options)

    # Render footer text into the foreground group (or potentially outside, depending on its implementation)
    # If add_footer_text positions relative to the group, adding it here is correct.
    if footer_text:
        add_footer_text(
            foreground_group,  # Pass the dedicated foreground group
            footer_text,
            layout["height"],
            layout["width"],
            colors,
            enable_latex,
            latex_scale,
            footer_font_size=footer_font_size,
        )


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
    latex_scale: float,
    footer_font_size: Optional[float],
    tree_labels: Optional[List[str]] = None,
    tree_label_font_size: Optional[float] = None,
    tree_label_style: Optional[Dict] = None,
    leaf_label_offset: float = 0.0,
    branch_labels: Optional[List[Optional[Dict]]] = None,
    inter_tree_paddings: Optional[List[float]] = None,  # NEW
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
        latex_scale: Scaling factor for LaTeX elements
        footer_font_size: Custom font size for footer text
        tree_labels: List of labels for each tree (e.g., ["A", "B", "C"])
        inter_tree_paddings: Optional list of paddings (in px) between trees
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
            latex_scale,
            footer_font_size,
            tree_label=tree_labels[i] if tree_labels and i < len(tree_labels) else None,
            tree_label_font_size=tree_label_font_size,
            tree_label_style=tree_label_style,
            leaf_label_offset=leaf_label_offset,
            branch_labels=(
                branch_labels[i] if branch_labels and i < len(branch_labels) else None
            ),
        )

        # Update x offset for next tree
        if inter_tree_paddings and i < len(inter_tree_paddings):
            x_offset += layout["width"] + inter_tree_paddings[i]
        else:
            x_offset += layout["width"] + INTER_TREE_SPACING


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
    stroke_width: Optional[float],
    node_marker_size: Optional[float],  # <-- Add node_marker_size parameter
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
    branch_labels: Optional[Union[List[Optional[Dict]], Dict]] = None,
    inter_tree_paddings: Optional[List[float]] = None,  # NEW
    leaf_connections: Optional[List[Dict[str, Any]]] = None,  # New parameter
) -> str:
    """
    Implementation of the tree SVG generation process.

    Args:
        All parameters are the same as in generate_tree_svg
        node_marker_size: Optional override for node marker size from style_opts.

    Returns:
        SVG string
    """
    # Get number of trees
    num_trees = len(roots)

    # Normalize input options
    (
        tree_highlights,
        tree_enclosures,
        tree_footers,
        tree_node_labels,
        tree_cut_edges,
        tree_branch_labels,
    ) = normalize_input_options(
        num_trees=num_trees,
        highlight_options=highlight_options,
        enclose_subtrees=enclose_subtrees,
        footer_texts=footer_texts,
        node_labels=node_labels,
        cut_edges=cut_edges,
        branch_labels=branch_labels,
    )

    # Select color scheme
    colors_dict = select_color_scheme(color_mode=color_mode, custom_colors=colors)

    # Apply manual stroke width if provided
    colors_dict, highlight_options = apply_manual_stroke_width(
        stroke_width=stroke_width,
        colors_dict=colors_dict,
        highlight_options=highlight_options,
    )

    # Apply node_marker_size override from style_opts if provided
    if node_marker_size is not None:
        colors_dict["node_marker_size"] = node_marker_size

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
        tree_footers=tree_footers,
        footer_font_size=footer_font_size,
        enable_latex=enable_latex,
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
    svg_root = create_svg_root(
        width=svg_width,
        height=svg_height,
        # colors=colors_dict,
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
        latex_scale=latex_scale,
        footer_font_size=footer_font_size,
        tree_labels=tree_labels,
        tree_label_font_size=tree_label_font_size,
        tree_label_style=tree_label_style,
        leaf_label_offset=leaf_label_offset,
        branch_labels=tree_branch_labels,
        inter_tree_paddings=inter_tree_paddings,  # NEW
    )

    if leaf_connections:
        render_leaf_connections(
            container=container,
            layouts=layouts,
            leaf_connections=leaf_connections,
            inter_tree_paddings=inter_tree_paddings,
        )

    # Convert to string
    svg_string = _svg_to_string(svg_root=svg_root)
    print(f"SVG string length: {len(svg_string)}")
    print(svg_string)

    print(f"SVG string length: {len(svg_string)}")

    # Attempt PDF conversion if requested
    if output_pdf:
        save_to_pdf(
            svg_string=svg_string,
            output_pdf=output_pdf,
        )

    return svg_string


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
    footer_texts: Optional[Union[str, List[Optional[str]]]] = None,
    caption: Optional[str] = None,
    tree_labels: Optional[List[str]] = None,
    colors: Optional[Dict] = None,
    latex_opts: Optional[Dict[str, Any]] = None,
    output_opts: Optional[Dict[str, Any]] = None,
    branch_labels: Optional[Union[List[Optional[Dict]], Dict]] = None,
    inter_tree_paddings: Optional[List[float]] = None,
    leaf_connections: Optional[List[Dict[str, Any]]] = None,  # New parameter
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
            "tree_label_font_size": float,
            "tree_label_style": Dict,
            "stroke_style": str,
            "stroke_width": float,
            "node_marker_size": float, # Added here for clarity
        }
        highlight_opts: per‑tree or global highlight config
        enclosure_opts: per‑tree or global enclosure config
        node_label_opts: per‑tree or global node‑label config
        footer_texts: str or list of footer strings
        caption: overall SVG caption
        tree_labels: List of per-tree labels (panel labels)
        colors: custom color palette
        latex_opts: {
            "enable": bool,
            "scale": float,
        }
        output_opts: {
            "pdf_path": Optional[str],
            "margins": {"left": float, "right": float, "top": float, "bottom": float},
        }
        branch_labels: per-tree or global branch label config
        inter_tree_paddings: Optional list of paddings (in px) between trees

    Returns:
        SVG content as a string.
    """
    # 1) fill defaults
    layout = {
        "type": "phylogram",
        "h_spacing": float(DEFAULT_H_SPACING_PER_LEAF),
        "v_spacing": float(DEFAULT_V_SPACING_PER_DEPTH),
        "use_shared_label_y": True,
        "target_height": None,
        "leaf_padding_top": 0.0,
        "leaf_label_offset": 0.0,
    }
    if layout_opts:
        layout.update(layout_opts)
        # Ensure correct types for h_spacing, v_spacing, leaf_padding_top, leaf_label_offset
        if "h_spacing" in layout:
            layout["h_spacing"] = float(layout["h_spacing"])
        if "v_spacing" in layout:
            layout["v_spacing"] = float(layout["v_spacing"])
        if "leaf_padding_top" in layout:
            layout["leaf_padding_top"] = float(layout["leaf_padding_top"])
        if "leaf_label_offset" in layout:
            layout["leaf_label_offset"] = float(layout["leaf_label_offset"])
        if "target_height" in layout and layout["target_height"] is not None:
            layout["target_height"] = int(layout["target_height"])
        if "use_shared_label_y" in layout:
            layout["use_shared_label_y"] = bool(layout["use_shared_label_y"])

    style = {
        "color_mode": "quanta",
        "leaf_font_size": None,
        "footer_font_size": None,
        "caption_font_size": None,
        "tree_label_font_size": None,
        "tree_label_style": None,
        "stroke_style": "balanced",
        "stroke_width": None,
        "node_marker_size": None,  # Default to None
    }
    if style_opts:
        style.update(style_opts)
        # Ensure correct types for style options
        if "leaf_font_size" in style and style["leaf_font_size"] is not None:
            style["leaf_font_size"] = float(style["leaf_font_size"])
        if "footer_font_size" in style and style["footer_font_size"] is not None:
            style["footer_font_size"] = float(style["footer_font_size"])
        if "caption_font_size" in style and style["caption_font_size"] is not None:
            style["caption_font_size"] = float(style["caption_font_size"])
        if (
            "tree_label_font_size" in style
            and style["tree_label_font_size"] is not None
        ):
            style["tree_label_font_size"] = float(style["tree_label_font_size"])
        if "stroke_width" in style and style["stroke_width"] is not None:
            style["stroke_width"] = float(style["stroke_width"])
        if "node_marker_size" in style and style["node_marker_size"] is not None:
            style["node_marker_size"] = float(style["node_marker_size"])
        if "stroke_style" in style and style["stroke_style"] is not None:
            style["stroke_style"] = str(style["stroke_style"])
        if "color_mode" in style and style["color_mode"] is not None:
            style["color_mode"] = str(style["color_mode"])
        if (
            "tree_label_style" in style
            and style["tree_label_style"] is not None
            and not isinstance(style["tree_label_style"], dict)
        ):
            style["tree_label_style"] = None  # fallback to None if not dict

    latex = {"enable": False, "scale": 1.0}
    if latex_opts:
        latex.update(latex_opts)
    latex["scale"] = float(latex.get("scale", 1.0))
    latex["enable"] = bool(latex.get("enable", False))

    output: Dict[str, Any] = {"pdf_path": None, "margins": {}}
    if output_opts:
        output.update(output_opts)
    # Ensure margins are floats if present
    if "margins" in output and output["margins"]:
        for k in ["left", "right", "top", "bottom"]:
            if k in output["margins"] and output["margins"][k] is not None:
                output["margins"][k] = float(output["margins"][k])

    # 2) unpack everything and call existing implementation
    return generate_tree_svg_implementation(
        roots,
        layout_type=str(layout["type"]),
        color_mode=str(style["color_mode"]),
        highlight_options=highlight_opts,
        split_options=None,
        enclose_subtrees=enclosure_opts,
        caption=caption,
        footer_texts=footer_texts,  # type: ignore
        enable_latex=bool(latex["enable"]),
        latex_scale=float(latex["scale"]),
        target_height=int(layout["target_height"])
        if layout["target_height"] is not None
        else None,
        use_shared_label_y=bool(layout["use_shared_label_y"]),
        h_spacing=float(layout["h_spacing"]),
        v_spacing=float(layout["v_spacing"]),
        label_area_height=float(DEFAULT_LABEL_AREA_HEIGHT),
        output_pdf=str(output["pdf_path"]) if output["pdf_path"] is not None else None,
        node_labels=node_label_opts,
        colors=colors,
        leaf_font_size=float(style["leaf_font_size"])
        if style["leaf_font_size"] is not None
        else None,
        footer_font_size=float(style["footer_font_size"])
        if style["footer_font_size"] is not None
        else None,
        stroke_width=float(style["stroke_width"])
        if style["stroke_width"] is not None
        else None,
        node_marker_size=float(style["node_marker_size"])
        if style["node_marker_size"] is not None
        else None,
        tree_labels=tree_labels,
        tree_label_font_size=float(style["tree_label_font_size"])
        if style["tree_label_font_size"] is not None
        else None,
        tree_label_style=style["tree_label_style"]
        if isinstance(style["tree_label_style"], dict)
        or style["tree_label_style"] is None
        else None,
        cut_edges=None,
        margin_left=output["margins"].get("left") if output.get("margins") else None,
        margin_right=output["margins"].get("right") if output.get("margins") else None,
        margin_top=output["margins"].get("top") if output.get("margins") else None,
        margin_bottom=output["margins"].get("bottom")
        if output.get("margins")
        else None,
        leaf_padding_top=float(layout["leaf_padding_top"]),
        leaf_label_offset=float(layout["leaf_label_offset"]),
        branch_labels=branch_labels,
        inter_tree_paddings=inter_tree_paddings,  # NEW
        leaf_connections=leaf_connections,  # New parameter
    )
