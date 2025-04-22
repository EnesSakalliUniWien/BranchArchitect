"""
Layout calculation functions for phylogenetic trees.

This module provides functions for calculating layouts of phylogenetic trees
in various formats (phylogram, cladogram).
"""

from typing import Dict, Tuple, List, Optional
from brancharchitect.tree import Node
from brancharchitect.plot.tree_utils import get_node_id, tree_depth, get_leaves, calculate_max_path_length

# Layout type definition
LayoutResult = Tuple[Dict[Node, Dict], float, float, float]

# Default Layout Constants (these can be overridden by parameters)
ZERO_LENGTH_TOLERANCE = 1e-9
LABEL_AREA_HEIGHT = 50  # Default label area height

def calculate_layout(
    root: Node,
    layout_type: str = "phylogram",
    h_spacing: float = 20,
    target_height: Optional[float] = None,
    leaf_padding_top: float = 0.0,
) -> LayoutResult:
    """
    Calculate the layout coordinates for a tree.

    Args:
        root: The root node of the tree
        layout_type: 'phylogram' or 'cladogram'
        h_spacing: Horizontal spacing between leaves
        target_height: Target height for the tree (for phylogram)
        leaf_padding_top: Top padding for leaves

    Returns:
        Tuple of (node_coords, label_y, width, height)
    """
    if layout_type == "cladogram":
        # Default vertical spacing for cladogram if not specified in parameters
        v_spacing = 50  # This can be exposed as a parameter too
        return calculate_cladogram_layout(root, h_spacing, v_spacing, leaf_padding_top=leaf_padding_top)
    elif layout_type == "phylogram":
        if target_height is None:
            depth = tree_depth(root)
            # Default values if not specified
            target_height = max(150, depth * 50 + LABEL_AREA_HEIGHT)
        return calculate_phylogram_layout(root, h_spacing, target_height, leaf_padding_top=leaf_padding_top)
    else:
        raise ValueError(f"Unknown layout type: {layout_type}")


def calculate_cladogram_layout(
    root: Node, h_spacing: float, v_spacing: float, leaf_padding_top: float = 0.0
) -> LayoutResult:
    """
    Calculate layout for a cladogram (fixed vertical spacing).
    
    Args:
        root: The root node of the tree
        h_spacing: Horizontal spacing between leaves
        v_spacing: Vertical spacing between depths
        leaf_padding_top: Top padding for leaves
        
    Returns:
        Tuple of (node_coords, label_y, width, height)
    """
    leaves = get_leaves(root)
    leaf_count = max(1, len(leaves))
    max_depth = tree_depth(root)

    # Calculate dimensions
    width = (leaf_count + 1) * h_spacing
    height_coords = max_depth * v_spacing
    total_height = height_coords + LABEL_AREA_HEIGHT + leaf_padding_top
    label_y = height_coords + 25 + leaf_padding_top

    # Calculate coordinates
    node_coords = {}
    leaf_index_ref = [0]  # Use list for mutable reference

    assign_cladogram_coords(root, node_coords, h_spacing, v_spacing, 0, leaf_index_ref, leaf_padding_top=0.0)

    # After all coordinates are assigned, shift all y by leaf_padding_top
    if leaf_padding_top != 0.0:
        for coords in node_coords.values():
            coords["y"] += leaf_padding_top
    label_y += leaf_padding_top
    total_height += leaf_padding_top

    return node_coords, label_y, width, total_height


def assign_cladogram_coords(
    node: Node,
    node_coords: Dict[Node, Dict],
    h_spacing: float,
    v_spacing: float,
    depth: int = 0,
    leaf_index_ref: List[int] = [0],
    leaf_padding_top: float = 0.0,
) -> Tuple[float, float]:
    """
    Recursively assign x,y coordinates for cladogram layout.

    Args:
        node: Current node
        node_coords: Dictionary to populate with coordinates
        h_spacing: Horizontal spacing between leaves
        v_spacing: Vertical spacing between depths
        depth: Current depth (for recursion)
        leaf_index_ref: Reference to current leaf index
        leaf_padding_top: Top padding for leaves

    Returns:
        (x, y) coordinates of the current node
    """
    y = depth * v_spacing
    
    # Get the raw node ID (without internal- prefix) for compatibility with highlight edges
    node_id = node.name if node.name else ""

    if node.is_leaf():
        # For leaves, assign x based on leaf index
        x = (leaf_index_ref[0] + 1) * h_spacing
        leaf_index_ref[0] += 1
        y_leaf = y + leaf_padding_top

        # Store coordinates
        node_coords[node] = {"x": x, "y": y_leaf, "depth": depth, "id": node_id}
        return (x, y_leaf)
    else:
        # For internal nodes, position at average of children
        child_positions = []
        child_nodes = []

        for child in node.children:
            child_x, child_y = assign_cladogram_coords(
                child, node_coords, h_spacing, v_spacing, depth + 1, leaf_index_ref, leaf_padding_top=leaf_padding_top
            )
            child_positions.append((child_x, child_y))
            child_nodes.append(child)

        # Average x position of children
        x = (
            sum(pos[0] for pos in child_positions) / len(child_positions)
            if child_positions
            else 0
        )

        # Store coordinates and references to children
        node_coords[node] = {
            "x": x,
            "y": y,
            "depth": depth,
            "id": node_id,
            "children_coords": child_positions,
            "child_nodes": child_nodes,
        }
        return (x, y)


def calculate_phylogram_layout(
    root: Node, h_spacing: float, target_height: float, leaf_padding_top: float = 0.0
) -> LayoutResult:
    """
    Calculate layout for a phylogram (branch lengths determine vertical spacing).
    
    Args:
        root: The root node of the tree
        h_spacing: Horizontal spacing between leaves
        target_height: Target height for the tree
        leaf_padding_top: Top padding for leaves
        
    Returns:
        Tuple of (node_coords, label_y, width, height)
    """
    # First, calculate the maximum cumulative branch length
    max_cumulative_length = calculate_max_path_length(root)

    # Get leaf count for width calculation
    leaves = get_leaves(root)
    leaf_count = max(1, len(leaves))
    width = (leaf_count + 1) * h_spacing

    # Initialize coordinates dictionary
    node_coords = {}
    leaf_index_ref = [0]

    # Handle zero total branch length case
    if max_cumulative_length <= ZERO_LENGTH_TOLERANCE:
        print(
            f"Warning: Tree has zero cumulative branch length. Using cladogram layout."
        )
        max_depth = tree_depth(root)
        v_spacing = 50
        total_height = (max_depth * v_spacing) + LABEL_AREA_HEIGHT + leaf_padding_top
        label_y = (max_depth * v_spacing) + 25 + leaf_padding_top

        assign_cladogram_coords(
            root, node_coords, h_spacing, v_spacing, 0, leaf_index_ref, leaf_padding_top=0.0
        )
    else:
        # Calculate y-scale factor for desired height
        drawable_height = max(10, target_height - LABEL_AREA_HEIGHT)
        y_scale = drawable_height / max_cumulative_length
        total_height = target_height + leaf_padding_top

        # Assign coordinates using branch lengths
        assign_phylogram_coords(
            root, node_coords, h_spacing, y_scale, 0, 0, 0, leaf_index_ref, leaf_padding_top=0.0
        )

        # Find maximum y-coordinate for label positioning
        max_y = max(coords.get("y", 0) for coords in node_coords.values())
        label_y = max_y + 25
        total_height = max(total_height, label_y + (LABEL_AREA_HEIGHT - 25))

    # After all coordinates are assigned, shift all y by leaf_padding_top
    if leaf_padding_top != 0.0:
        for coords in node_coords.values():
            coords["y"] += leaf_padding_top
    label_y += leaf_padding_top
    total_height += leaf_padding_top

    return node_coords, label_y, width, total_height


def assign_phylogram_coords(
    node: Node,
    node_coords: Dict[Node, Dict],
    h_spacing: float,
    y_scale: float,
    parent_y: float = 0.0,
    distance_from_root: float = 0.0,
    depth: int = 0,
    leaf_index_ref: List[int] = [0],
    leaf_padding_top: float = 0.0,
) -> Tuple[float, float]:
    """
    Recursively assign x,y coordinates for phylogram layout.

    Args:
        node: Current node
        node_coords: Dictionary to populate with coordinates
        h_spacing: Horizontal spacing between leaves
        y_scale: Scale factor for vertical distances
        parent_y: Y-coordinate of parent node
        distance_from_root: Cumulative distance from root
        depth: Current depth (for tree traversal)
        leaf_index_ref: Reference to current leaf index
        leaf_padding_top: Top padding for leaves

    Returns:
        (x, y) coordinates of the current node
    """
    # Get branch length (default to 0 if None or negative)
    branch_length = node.length if node.length is not None and node.length > 0 else 0.0

    # Calculate position
    distance = distance_from_root + branch_length
    y = parent_y + (branch_length * y_scale)
    
    # Get the raw node ID (without internal- prefix) for compatibility with highlight edges
    node_id = node.name if node.name else ""

    if node.is_leaf():
        # For leaves, assign x based on leaf index
        x = (leaf_index_ref[0] + 1) * h_spacing
        leaf_index_ref[0] += 1
        y_leaf = y + leaf_padding_top

        # Store coordinates
        node_coords[node] = {
            "x": x,
            "y": y_leaf,
            "depth": depth,
            "id": node_id,
            "dist_from_root": distance,
        }
        return (x, y_leaf)
    else:
        # For internal nodes, position at average of children
        child_positions = []
        child_nodes = []

        for child in node.children:
            child_x, child_y = assign_phylogram_coords(
                child,
                node_coords,
                h_spacing,
                y_scale,
                y,
                distance,
                depth + 1,
                leaf_index_ref,
                leaf_padding_top=leaf_padding_top
            )
            child_positions.append((child_x, child_y))
            child_nodes.append(child)

        # Average x position of children
        x = (
            sum(pos[0] for pos in child_positions) / len(child_positions)
            if child_positions
            else 0
        )

        # Store coordinates and references to children
        node_coords[node] = {
            "x": x,
            "y": y,
            "depth": depth,
            "id": node_id,
            "dist_from_root": distance,
            "children_coords": child_positions,
            "child_nodes": child_nodes,
        }
        return (x, y)


def calculate_tree_layouts(roots, layout_type, h_spacing, target_height, v_spacing, leaf_padding_top=0.0):
    """
    Calculate layouts for multiple trees.

    Args:
        roots: List of root nodes for the trees
        layout_type: Layout type ('phylogram' or 'cladogram')
        h_spacing: Horizontal spacing between leaves
        target_height: Target height for the trees
        v_spacing: Vertical spacing for cladogram layout
        leaf_padding_top: Top padding for leaves

    Returns:
        List of layout results for each tree
    """
    layouts = []
    for tree_index, root in enumerate(roots):
        layout = calculate_single_tree_layout(tree_index, root, layout_type, h_spacing, target_height, leaf_padding_top)
        layouts.append(layout)
    return layouts


def calculate_target_height(roots, layout_type, target_height, v_spacing):
    """
    Calculate the target height for multiple trees.

    Args:
        roots: List of root nodes for the trees
        layout_type: Layout type ('phylogram' or 'cladogram')
        target_height: Initial target height
        v_spacing: Vertical spacing for cladogram layout

    Returns:
        Adjusted target height
    """
    if layout_type == "cladogram":
        max_depth = max(tree_depth(root) for root in roots)
        return max(target_height, max_depth * v_spacing + LABEL_AREA_HEIGHT)
    return target_height


def calculate_single_tree_layout(tree_index, root, layout_type, h_spacing, target_height, leaf_padding_top):
    """
    Calculate layout for a single tree.

    Args:
        tree_index: Index of the tree
        root: Root node of the tree
        layout_type: Layout type ('phylogram' or 'cladogram')
        h_spacing: Horizontal spacing between leaves
        target_height: Target height for the tree
        leaf_padding_top: Top padding for leaves

    Returns:
        Layout result for the tree
    """
    return calculate_layout(root, layout_type, h_spacing, target_height, leaf_padding_top)


def adjust_layouts_for_shared_labels(layouts, all_label_y, use_shared_label_y, label_area_height):
    """
    Adjust layouts for shared labels.

    Args:
        layouts: List of layouts
        all_label_y: List of label y-coordinates
        use_shared_label_y: Whether to use shared label y-coordinate
        label_area_height: Height of the label area

    Returns:
        Adjusted layouts
    """
    if use_shared_label_y:
        shared_label_y = max(all_label_y)
        for layout in layouts:
            layout[1] = shared_label_y
            layout[3] = max(layout[3], shared_label_y + label_area_height)
    return layouts


def calculate_footer_height(tree_footers, colors, enable_latex):
    """
    Calculate footer height based on tree footers and colors.

    Args:
        tree_footers: List of tree footers
        colors: List of colors
        enable_latex: Whether LaTeX is enabled

    Returns:
        Footer height
    """
    max_footer_lines = count_max_footer_lines(tree_footers)
    line_height = get_font_size_from_colors(colors)
    return calculate_footer_height_from_lines(max_footer_lines, line_height, enable_latex)


def get_font_size_from_colors(colors):
    """
    Get font size based on colors.

    Args:
        colors: List of colors

    Returns:
        Font size
    """
    return max(10, len(colors) * 2)


def count_max_footer_lines(tree_footers):
    """
    Count the maximum number of footer lines.

    Args:
        tree_footers: List of tree footers

    Returns:
        Maximum number of footer lines
    """
    return max(len(footer.split("\n")) for footer in tree_footers)


def calculate_footer_height_from_lines(max_footer_lines, line_height, enable_latex):
    """
    Calculate footer height based on lines.

    Args:
        max_footer_lines: Maximum number of footer lines
        line_height: Height of a single line
        enable_latex: Whether LaTeX is enabled

    Returns:
        Footer height
    """
    return max_footer_lines * line_height + (10 if enable_latex else 0)


def calculate_svg_dimensions(total_width, max_height, footer_height, caption, margin_left, margin_right, margin_top, margin_bottom):
    """
    Calculate SVG dimensions.

    Args:
        total_width: Total width of the SVG
        max_height: Maximum height of the SVG
        footer_height: Height of the footer
        caption: Caption text
        margin_left: Left margin
        margin_right: Right margin
        margin_top: Top margin
        margin_bottom: Bottom margin

    Returns:
        Tuple of (width, height)
    """
    caption_height = 30 if caption else 0
    width = total_width + margin_left + margin_right
    height = max_height + footer_height + caption_height + margin_top + margin_bottom
    return width, height