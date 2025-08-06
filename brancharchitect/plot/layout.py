"""
Layout calculation functions for phylogenetic trees.

This module provides functions for calculating the coordinates (x, y) for each node
in a phylogenetic tree to prepare it for plotting. It supports 'phylogram'
(where branch lengths determine vertical spacing) and 'cladogram' (where vertical
spacing is fixed per depth level) layouts.
"""

import xml.etree.ElementTree as ET
from typing import Dict, Tuple, List, Optional
from brancharchitect.tree import Node
from brancharchitect.plot.tree_utils import (
    tree_depth,
    get_leaves,
    calculate_max_path_length,
)
from brancharchitect.plot.tree_utils import (
    collapse_zero_length_branches,
    calculate_node_depths,
)
from brancharchitect.plot.paper_plot.paper_plot_constants import (
    DEFAULT_LABEL_AREA_HEIGHT,
    CAPTION_HEIGHT,
    INTER_TREE_SPACING,
    MARGIN_X,
    MARGIN_Y,
    ZERO_LENGTH_TOLERANCE,
    DEFAULT_CLADOGRAM_V_SPACING,
    DEFAULT_TARGET_HEIGHT_BASE,
    DEFAULT_TARGET_HEIGHT_PER_DEPTH,
)

# --- Layout Type Definition ---
# Stores the calculated layout:
# - Dict mapping each Node to its coordinate dictionary ('x', 'y', 'depth', 'id', etc.)
# - Float representing the y-coordinate for placing leaf labels.
# - Float representing the total calculated width of the layout.
# - Float representing the total calculated height of the layout.
LayoutResult = Tuple[Dict[Node, Dict], float, float, float]


def calculate_layout(
    root: Node,
    layout_type: str = "phylogram",
    h_spacing: float = 20,
    target_height: Optional[float] = None,
    leaf_padding_top: float = 0.0,
    label_area_height: float = DEFAULT_LABEL_AREA_HEIGHT,
) -> LayoutResult:
    """
    Calculate the layout coordinates for a tree, dispatching to specific layout functions.

    Args:
        root: The root node of the phylogenetic tree.
        layout_type: The type of layout ('phylogram' or 'cladogram').
        h_spacing: The horizontal spacing between adjacent leaf nodes.
        target_height: The desired total vertical height for the tree drawing area
                       (used in phylograms, excluding label area). If None, a default
                       is calculated based on tree depth.
        leaf_padding_top: Vertical padding to add above the entire tree layout.
        label_area_height: Vertical space reserved at the bottom for leaf labels.

    Returns:
        A LayoutResult tuple containing node coordinates, label y-position, width, and height.

    Raises:
        ValueError: If an unknown layout_type is provided.
    """
    if layout_type == "cladogram":
        return _calculate_cladogram_layout(
            root,
            h_spacing,
            DEFAULT_CLADOGRAM_V_SPACING,
            leaf_padding_top,
            label_area_height,
        )
    elif layout_type == "phylogram":
        # Calculate default target height if not provided
        if target_height is None:
            depth = tree_depth(root)
            target_height = max(
                DEFAULT_TARGET_HEIGHT_BASE,
                depth * DEFAULT_TARGET_HEIGHT_PER_DEPTH + label_area_height,
            )
        return _calculate_phylogram_layout(
            root, h_spacing, target_height, leaf_padding_top, label_area_height
        )
    else:
        raise ValueError(f"Unknown layout type: {layout_type}")


# --- Cladogram Layout ---


def _calculate_cladogram_layout(
    root: Node,
    h_spacing: float,
    v_spacing: float,
    leaf_padding_top: float,
    label_area_height: float,
) -> LayoutResult:
    """
    Calculate layout coordinates for a cladogram (fixed vertical spacing per depth).

    Args:
        root: The root node of the tree.
        h_spacing: Horizontal spacing between adjacent leaf nodes.
        v_spacing: Vertical spacing between depth levels.
        leaf_padding_top: Vertical padding to add above the entire tree layout.
        label_area_height: Vertical space reserved at the bottom for leaf labels.

    Returns:
        A LayoutResult tuple.
    """
    leaves = get_leaves(root)
    leaf_count = max(1, len(leaves))
    max_depth = tree_depth(root)

    # --- Calculate initial dimensions (before padding) ---
    width = (leaf_count + 1) * h_spacing
    # Height based purely on node vertical positions
    tree_drawing_height = max_depth * v_spacing
    # Initial total height includes space for labels
    total_height = tree_drawing_height + label_area_height
    # Initial label y position (relative to tree drawing area)
    initial_label_y = tree_drawing_height + (
        label_area_height / 2
    )  # Center in label area

    # --- Calculate node coordinates (relative to top-left 0,0) ---
    node_coords: Dict[Node, Dict] = {}
    leaf_index_ref = [0]  # Use list for mutable reference across recursion
    _assign_cladogram_coords(
        node=root,
        node_coords=node_coords,
        h_spacing=h_spacing,
        v_spacing=v_spacing,
        depth=0,
        leaf_index_ref=leaf_index_ref,
    )

    # --- Apply top padding ---
    final_label_y = initial_label_y
    if leaf_padding_top != 0.0:
        for coords in node_coords.values():
            coords["y"] += leaf_padding_top
        final_label_y += leaf_padding_top
        total_height += leaf_padding_top  # Add padding to total height

    return node_coords, final_label_y, width, total_height


def _assign_cladogram_coords(
    node: Node,
    node_coords: Dict[Node, Dict],
    h_spacing: float,
    v_spacing: float,
    depth: int,
    leaf_index_ref: List[int],
) -> Tuple[float, float]:
    """
    Recursively assign x, y coordinates for cladogram layout (relative to 0,0).

    Args:
        node: The current node being processed.
        node_coords: The dictionary to populate with node coordinates.
        h_spacing: Horizontal spacing between adjacent leaf nodes.
        v_spacing: Vertical spacing between depth levels.
        depth: The current depth of the node in the tree (root is 0).
        leaf_index_ref: A mutable list containing the current leaf index counter.

    Returns:
        The (x, y) coordinates calculated for the current node.
    """
    # Calculate y based on depth
    y = depth * v_spacing
    node_id = node.name if node.name else f"internal_{id(node)}"  # Ensure some ID

    if node.is_leaf():
        # Assign x based on the order leaves are visited
        x = (leaf_index_ref[0] + 1) * h_spacing
        leaf_index_ref[0] += 1
        node_coords[node] = {"x": x, "y": y, "depth": depth, "id": node_id}
        return (x, y)
    else:
        # Position internal nodes based on the average x of their children
        child_positions = []
        child_nodes = []
        for child in node.children:
            child_x, child_y = _assign_cladogram_coords(
                child, node_coords, h_spacing, v_spacing, depth + 1, leaf_index_ref
            )
            child_positions.append((child_x, child_y))
            child_nodes.append(child)

        # Calculate average x position
        x = (
            sum(pos[0] for pos in child_positions) / len(child_positions)
            if child_positions
            else h_spacing  # Fallback if no children (shouldn't happen in valid tree)
        )

        node_coords[node] = {
            "x": x,
            "y": y,
            "depth": depth,
            "id": node_id,
            "children_coords": child_positions,  # Store for potential drawing use
            "child_nodes": child_nodes,  # Store for potential drawing use
        }
        return (x, y)


# --- Phylogram Layout ---


def _calculate_phylogram_layout(
    root: Node,
    h_spacing: float,
    target_height: float,
    leaf_padding_top: float,
    label_area_height: float,
) -> LayoutResult:
    """
    Calculate layout coordinates for a phylogram (vertical spacing based on branch lengths).

    Args:
        root: The root node of the tree.
        h_spacing: Horizontal spacing between adjacent leaf nodes.
        target_height: The desired total vertical height for the tree drawing area
                       (excluding the label area).
        leaf_padding_top: Vertical padding to add above the entire tree layout.
        label_area_height: Vertical space reserved at the bottom for leaf labels.

    Returns:
        A LayoutResult tuple.
    """
    max_cumulative_length = calculate_max_path_length(root)
    leaves = get_leaves(root)
    leaf_count = max(1, len(leaves))

    # --- Calculate dimensions ---
    width = (leaf_count + 1) * h_spacing
    node_coords: Dict[Node, Dict] = {}
    leaf_index_ref = [0]  # Mutable leaf counter

    # --- Handle zero branch length case: Fallback to cladogram logic ---
    if max_cumulative_length <= ZERO_LENGTH_TOLERANCE:
        print(
            "Warning: Tree has zero or negligible cumulative branch length. "
            f"Using cladogram layout logic with v_spacing={DEFAULT_CLADOGRAM_V_SPACING}."
        )
        # Use cladogram assignment logic but respect the overall height structure
        max_depth = tree_depth(root)
        tree_drawing_height = max_depth * DEFAULT_CLADOGRAM_V_SPACING
        _assign_cladogram_coords(
            root, node_coords, h_spacing, DEFAULT_CLADOGRAM_V_SPACING, 0, leaf_index_ref
        )
        # Ensure total height respects target_height if it's larger
        total_height = max(target_height, tree_drawing_height + label_area_height)
        initial_label_y = max(
            target_height - label_area_height + (label_area_height / 2),
            tree_drawing_height + (label_area_height / 2),
        )

    # --- Standard Phylogram Calculation ---
    else:
        # Calculate y-scale factor to fit within the target drawing height
        # Ensure drawable_height is positive
        drawable_height = max(1.0, target_height - label_area_height)
        y_scale = drawable_height / max_cumulative_length
        initial_total_height = target_height  # Start with target height

        # Assign coordinates using branch lengths (relative to 0,0)
        _assign_phylogram_coords(
            node=root,
            node_coords=node_coords,
            h_spacing=h_spacing,
            y_scale=y_scale,
            parent_y=0.0,
            distance_from_root=0.0,
            depth=0,
            leaf_index_ref=leaf_index_ref,
        )

        # Find the actual maximum y reached by nodes
        max_y_coord = max(coords.get("y", 0) for coords in node_coords.values())

        # Calculate label y position and adjust total height if needed
        # Place labels centered in the label area below the max node y
        initial_label_y = max_y_coord + (label_area_height / 2)
        total_height = max(initial_total_height, max_y_coord + label_area_height)

    # --- Apply top padding ---
    final_label_y = initial_label_y
    if leaf_padding_top != 0.0:
        for coords in node_coords.values():
            coords["y"] += leaf_padding_top
        final_label_y += leaf_padding_top
        total_height += leaf_padding_top  # Add padding to total height

    return node_coords, final_label_y, width, total_height


def _assign_phylogram_coords(
    node: Node,
    node_coords: Dict[Node, Dict],
    h_spacing: float,
    y_scale: float,
    parent_y: float,
    distance_from_root: float,
    depth: int,
    leaf_index_ref: List[int],
) -> Tuple[float, float]:
    """
    Recursively assign x, y coordinates for phylogram layout (relative to 0,0).

    Args:
        node: The current node being processed.
        node_coords: The dictionary to populate with node coordinates.
        h_spacing: Horizontal spacing between adjacent leaf nodes.
        y_scale: The scaling factor to convert branch lengths to y-coordinates.
        parent_y: The y-coordinate of the parent node.
        distance_from_root: The cumulative branch length from the root to this node's parent.
        depth: The current depth (number of nodes) from the root.
        leaf_index_ref: A mutable list containing the current leaf index counter.

    Returns:
        The (x, y) coordinates calculated for the current node.
    """
    # Use 0 for missing or non-positive branch lengths
    branch_length = node.length if node.length is not None and node.length > 0 else 0.0

    # Calculate y based on parent's y and scaled branch length
    current_distance = distance_from_root + branch_length
    y = parent_y + (branch_length * y_scale)
    node_id = node.name if node.name else f"internal_{id(node)}"  # Ensure some ID

    if node.is_leaf():
        # Assign x based on the order leaves are visited
        x = (leaf_index_ref[0] + 1) * h_spacing
        leaf_index_ref[0] += 1
        node_coords[node] = {
            "x": x,
            "y": y,
            "depth": depth,
            "id": node_id,
            "dist_from_root": current_distance,
        }
        return (x, y)
    else:
        # Position internal nodes based on the average x of their children
        child_positions = []
        child_nodes = []
        for child in node.children:
            child_x, child_y = _assign_phylogram_coords(
                child,
                node_coords,
                h_spacing,
                y_scale,
                parent_y=y,  # Current node's y is parent_y for children
                distance_from_root=current_distance,
                depth=depth + 1,
                leaf_index_ref=leaf_index_ref,
            )
            child_positions.append((child_x, child_y))
            child_nodes.append(child)

        # Calculate average x position
        x = (
            sum(pos[0] for pos in child_positions) / len(child_positions)
            if child_positions
            else h_spacing  # Fallback if no children
        )

        node_coords[node] = {
            "x": x,
            "y": y,
            "depth": depth,
            "id": node_id,
            "dist_from_root": current_distance,
            "children_coords": child_positions,  # Store for potential drawing use
            "child_nodes": child_nodes,  # Store for potential drawing use
        }
        return (x, y)


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


# -----------------------------------------------------------------------------
# E. Layout Functions
# -----------------------------------------------------------------------------


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
