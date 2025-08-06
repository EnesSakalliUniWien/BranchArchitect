import math
import xml.etree.ElementTree as ET
import uuid
from typing import List, Dict, Tuple, Optional
from brancharchitect.tree import Node  # Cleaned: use Node from tree.py
from brancharchitect.plot.tree_utils import (
    get_node_label,
    is_leaf,
)
from brancharchitect.plot.svg import (
    add_svg_path,
    get_svg_root,
    add_labels,
    VisualNode,
)

###############################################################################
# Constants
###############################################################################
STROKE_WIDTH = 1
DEFAULT_NODE_LENGTH = 1.0
DEFAULT_NODE_NAME = "Unnamed"
DEFAULT_FONT_FAMILY = "Monospace"
DEFAULT_FONT_SIZE = "12"
DEFAULT_STROKE_COLOR = "#000"
DEFAULT_RECT_STROKE_COLOR = "#c8c8c8"
CONNECTION_STROKE_COLOR = "#808080"  # Grey color for connections
CONNECTION_STROKE_WIDTH = 0.5
CONNECTION_STROKE_DASHARRAY = "4,4"  # Dashed line for connections


def _get_branch_length(node: Node) -> float:
    """Get branch length from node, supporting multiple attribute names."""
    return getattr(
        node,
        "distance",
        getattr(node, "branch_length", getattr(node, "length", DEFAULT_NODE_LENGTH)),
    )


def tree_to_circular_visual_nodes(
    root: Node,
    total_angle: float,
    order: List[str],
    parent_radius: float,
    ignore_branch_lengths: bool = False,
) -> VisualNode:
    """
    Convert a Node tree to a VisualNode tree with radius and angle properties.

    Args:
        root: The Node from your data
        total_angle: Angular range to distribute leaves (usually 2*pi)
        order: List of leaf names for angle calculations
        parent_radius: The parent's radius
        ignore_branch_lengths: If True, treat each branch as length=1

    Returns:
        A VisualNode with radius/angle, plus children
    """
    branch_length = _get_branch_length(root)

    node_radius = parent_radius + (
        DEFAULT_NODE_LENGTH if ignore_branch_lengths else branch_length
    )
    node_name = get_node_label(root)
    node_is_leaf = is_leaf(root)

    if node_is_leaf:
        # Leaf angle based on position in `order`
        idx = 0  # Default index
        if (
            order and node_name in order
        ):  # Check if order is not empty and node_name is in order
            idx = order.index(node_name)

        leaf_angle = 0.0  # Default angle
        if order and len(order) > 0:  # Check if order is not empty before division
            leaf_angle = (total_angle / len(order)) * idx

        return VisualNode(node_radius, leaf_angle, node_name, node_is_leaf, [])
    else:
        # Internal node: build children first
        child_visuals = [
            tree_to_circular_visual_nodes(
                child, total_angle, order, node_radius, ignore_branch_lengths
            )
            for child in root.children
        ]
        # The node's own angle is the average of child angles
        node_angle = 0.0  # Default angle
        if len(child_visuals) > 0:
            angle_sum = sum(cv.angle for cv in child_visuals)
            node_angle = angle_sum / len(child_visuals)

        vnode = VisualNode(
            node_radius, node_angle, node_name, node_is_leaf, child_visuals
        )
        return vnode


###############################################################################
# Circular Tree Link Generation
###############################################################################
def build_circular_link_path(source: VisualNode, target: VisualNode) -> str:
    """
    Build an SVG path string connecting two visual nodes in a circular tree layout.

    Creates a path with an arc around source's radius followed by a straight line
    to the target.

    Args:
        source: Parent visual node
        target: Child visual node

    Returns:
        SVG path data string
    """
    sx, sy = source.cartesian()
    tx, ty = target.cartesian()

    # Create an arc from the parent's angle to target's angle at parent's radius
    cx = source.radius * math.cos(target.angle)
    cy = source.radius * math.sin(target.angle)

    arc_flag = 1 if abs(target.angle - source.angle) > math.pi else 0
    sweep_flag = 1 if abs(source.angle) < abs(target.angle) else 0

    path_d = f"M {sx},{sy} A {source.radius},{source.radius} 0 {arc_flag} {sweep_flag} {cx},{cy} L {tx},{ty}"
    return path_d


def generate_circular_links(
    node: VisualNode, stroke_color: str = DEFAULT_STROKE_COLOR
) -> List[Dict[str, str]]:
    """
    Recursively build SVG path attributes for links from node to each child in a circular tree.

    Args:
        node: The visual node to generate links from
        stroke_color: The color of the links.

    Returns:
        List of SVG path attribute dictionaries
    """
    all_paths = []
    for child in node.children:
        attrs = {
            "class": "links",
            "stroke-width": str(STROKE_WIDTH),
            "fill": "none",
            "style": f"stroke-opacity:1;stroke:{stroke_color}",
            "d": build_circular_link_path(node, child),
        }
        all_paths.append(attrs)
        all_paths.extend(generate_circular_links(child, stroke_color=stroke_color))
    return all_paths


###############################################################################
# Circular Tree Coordinate Calculation
###############################################################################
def calculate_circular_tree_coordinates(
    root: Node,
    total_angle: float,
    order: List[str],
    parent_radius: float,
    ignore_branch_lengths: bool = False,
    scale_factor: float = 1.0,
) -> Dict[str, object]:
    """
    Calculate coordinates for nodes in a circular tree layout.

    Args:
        root: The root node of the tree
        total_angle: Angular range to distribute leaves (usually 2*pi)
        order: List of leaf names for angle calculations
        parent_radius: The parent's radius
        ignore_branch_lengths: If True, treat each branch as length=1
        scale_factor: Scale factor for radius calculations

    Returns:
        Dictionary with visual node, scaled visual node, and path data
    """
    # Convert the tree to visual nodes with radii and angles
    visual_node = tree_to_circular_visual_nodes(
        root, total_angle, order, parent_radius, ignore_branch_lengths
    )

    # Scale the radii of all nodes if scale_factor is provided
    if scale_factor != 1.0:
        for node in visual_node.traverse():
            node.scale_radius(scale_factor)

    # Generate path data for the links
    links = generate_circular_links(visual_node)

    return {"visual_node": visual_node, "links": links, "order": order}


###############################################################################
# Tree Visualization Generators
###############################################################################


def generate_circular_two_trees_svg(
    root1: Node,
    root2: Node,
    size: int = 400,
    margin: int = 30,
    label_offset: int = 2,
    ignore_branch_lengths: bool = False,
    font_family: str = DEFAULT_FONT_FAMILY,
    font_size: str = DEFAULT_FONT_SIZE,
    stroke_color: str = DEFAULT_STROKE_COLOR,
) -> ET.Element:
    """
    Create an SVG with exactly two circular trees side by side.
    Returns the SVG root element (not a string).
    """
    total_width = 2 * size
    total_height = size
    svg_root = get_svg_root(total_width, total_height)

    # Group 1 centered in the left box
    group1 = ET.SubElement(
        svg_root, "g", {"transform": f"translate({size / 2}, {size / 2})"}
    )

    # Group 2 centered in the right box
    group2 = ET.SubElement(
        svg_root, "g", {"transform": f"translate({size * 1.5}, {size / 2})"}
    )

    # Render first tree in the left group
    render_single_circular_tree(
        root=root1,
        group=group1,
        size=size,
        margin=margin,
        label_offset=label_offset,
        ignore_branch_lengths=ignore_branch_lengths,
        font_family=font_family,
        font_size=font_size,
        stroke_color=stroke_color,
        svg_root=svg_root,
    )

    # Render second tree in the right group
    render_single_circular_tree(
        root=root2,
        group=group2,
        size=size,
        margin=margin,
        label_offset=label_offset,
        ignore_branch_lengths=ignore_branch_lengths,
        font_family=font_family,
        font_size=font_size,
        stroke_color=stroke_color,
        svg_root=svg_root,
    )

    return svg_root


def _prepare_highlight_dict(highlight_branches, highlight_colors):
    """Prepare highlight dictionary from various input formats."""
    highlight_dict = {}
    if highlight_branches:
        if isinstance(highlight_branches, dict):
            highlight_dict = highlight_branches
        elif isinstance(highlight_branches, list):
            for item in highlight_branches:
                if isinstance(item, tuple) and len(item) == 3:
                    highlight_dict[(item[0], item[1])] = item[2]
                elif isinstance(item, tuple) and len(item) == 2:
                    highlight_dict[(item[0], item[1])] = None
    if highlight_colors:
        highlight_dict.update(highlight_colors)
    return highlight_dict


def _create_path_attributes(
    path_d: str, color: str, width: float, opacity: str = "1.0"
) -> dict:
    """Create standardized SVG path attributes."""
    return {
        "d": path_d,
        "stroke": color,
        "stroke-width": str(width),
        "stroke-opacity": opacity,
        "fill": "none",
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
    }


def _create_highlighted_path(
    highlight_info, path_d: str, highlight_width: float, svg_root, node, child
) -> dict:
    """Create path attributes for highlighted branches."""
    # Extract highlight properties
    if isinstance(highlight_info, dict):
        color = highlight_info.get("highlight_color", "#FFD700")
        width = highlight_info.get("stroke_width", highlight_width)
        gradient_end = highlight_info.get("gradient_end", None)
        use_elevation = highlight_info.get("use_elevation", False)
    else:
        color = highlight_info or "#FFD700"
        width = highlight_width
        gradient_end = None
        use_elevation = False

    attrs = _create_path_attributes(path_d, color, width, "0.95")

    # Add gradient if specified
    if gradient_end and svg_root is not None:
        defs = svg_root.find("defs") or ET.SubElement(svg_root, "defs")
        grad_id = f"grad-{uuid.uuid4().hex[:8]}"
        grad = ET.SubElement(
            defs,
            "linearGradient",
            {
                "id": grad_id,
                "gradientUnits": "userSpaceOnUse",
                "x1": str(node.radius * math.cos(node.angle)),
                "y1": str(node.radius * math.sin(node.angle)),
                "x2": str(child.radius * math.cos(child.angle)),
                "y2": str(child.radius * math.sin(child.angle)),
            },
        )
        ET.SubElement(grad, "stop", {"offset": "0%", "stop-color": color})
        ET.SubElement(grad, "stop", {"offset": "100%", "stop-color": gradient_end})
        attrs["stroke"] = f"url(#{grad_id})"

    # Add elevation filter if specified
    if use_elevation and svg_root is not None:
        defs = svg_root.find("defs") or ET.SubElement(svg_root, "defs")
        filter_id = f"glow-{uuid.uuid4().hex[:8]}"
        flt = ET.SubElement(
            defs,
            "filter",
            {
                "id": filter_id,
                "x": "-20%",
                "y": "-20%",
                "width": "140%",
                "height": "140%",
            },
        )
        ET.SubElement(
            flt,
            "feGaussianBlur",
            {
                "in": "SourceGraphic",
                "stdDeviation": "2.5",
                "result": "blur",
            },
        )
        merge = ET.SubElement(flt, "feMerge")
        merge.append(ET.Element("feMergeNode", {"in": "blur"}))
        merge.append(ET.Element("feMergeNode", {"in": "SourceGraphic"}))
        attrs["filter"] = f"url(#{filter_id})"

    return attrs


def render_single_circular_tree(
    root: Node,
    group: ET.Element,
    size: int,
    margin: int,
    label_offset: int,
    ignore_branch_lengths: bool = False,
    font_family: str = DEFAULT_FONT_FAMILY,
    font_size: str = DEFAULT_FONT_SIZE,
    stroke_color: str = DEFAULT_STROKE_COLOR,
    highlight_branches: Optional[
        list
    ] = None,  # List of (parent_name, child_name) or dict or list of (tuple, color)
    highlight_width: float = 4.0,
    highlight_colors: Optional[dict] = None,  # Optional dict for per-branch color
    svg_root: Optional[ET.Element] = None,  # Pass the SVG root explicitly
) -> List[VisualNode]:
    """
    Render a single circular tree within the provided SVG group element.
    Returns a list of scaled leaf VisualNode objects.
    highlight_branches: list of (parent, child) or dict {(parent, child): color}
    highlight_colors: dict {(parent, child): color} (optional, overrides tuple color)
    If highlight_branches is a dict, use its values as colors.
    If highlight_branches is a list of (parent, child, color), use color.
    """
    order = root.get_current_order()
    tree_data = calculate_circular_tree_coordinates(
        root, 2 * math.pi, order, 0, ignore_branch_lengths
    )
    visual_node = tree_data["visual_node"]

    # Apply consistent scaling
    factor = _calculate_scaling_factor(visual_node, size, margin, label_offset)
    scaled_leaf_nodes: List[VisualNode] = []
    for node_v in visual_node.traverse():
        node_v.scale_radius(factor)
        if node_v.is_leaf:
            scaled_leaf_nodes.append(node_v)
    # Prepare highlight lookup
    highlight_dict = _prepare_highlight_dict(highlight_branches, highlight_colors)
    # Draw links (branches)
    for node in visual_node.traverse():
        for child in node.children:
            edge = (get_node_label(node), get_node_label(child))
            highlight_info = highlight_dict.get(edge, None)
            path_d = build_circular_link_path(node, child)

            if highlight_info is not None:
                attrs = _create_highlighted_path(
                    highlight_info, path_d, highlight_width, svg_root, node, child
                )
            else:
                attrs = _create_path_attributes(path_d, stroke_color, STROKE_WIDTH)

            ET.SubElement(group, "path", attrs)
    # Add labels
    add_labels(
        group,
        order,
        (size / 2 - margin),
        font_family=font_family,
        font_size=font_size,
        nodes=scaled_leaf_nodes,
    )
    return scaled_leaf_nodes


def _create_svg_root_and_groups(
    n: int, size: int
) -> Tuple[ET.Element, ET.Element, ET.Element]:
    """Create the SVG root and main groups for trees and connections. Ensures connections are drawn behind trees."""
    total_width = n * size
    total_height = size
    svg_root = get_svg_root(total_width, total_height)
    connections_group = ET.SubElement(svg_root, "g", {"id": "connections_group"})
    trees_group = ET.SubElement(svg_root, "g", {"id": "all_trees_group"})
    return svg_root, trees_group, connections_group


def _calculate_scaling_factor(
    visual_node: VisualNode, size: int, margin: int, label_offset: int
) -> float:
    """Calculate radius scaling factor for visual nodes."""
    all_nodes = list(visual_node.traverse())
    max_r = max((node.radius for node in all_nodes), default=0.0)
    usable_radius = size / 2 - margin - label_offset
    return usable_radius / max_r if max_r != 0 else 1.0


def _add_zero_length_indicators_simple(
    root: Node,
    group: ET.Element,
    indicator_color: str,
    indicator_size: float,
    size: int,
    margin: int,
    label_offset: int,
    ignore_branch_lengths: bool,
) -> None:
    """Add visual indicators (dots) for nodes with zero-length branches using a simplified approach."""

    order = root.get_current_order()
    tree_data = calculate_circular_tree_coordinates(
        root, 2 * math.pi, list(order), 0, ignore_branch_lengths=ignore_branch_lengths
    )
    visual_node = tree_data["visual_node"]

    # Apply consistent scaling
    factor = _calculate_scaling_factor(visual_node, size, margin, label_offset)
    for node_v in visual_node.traverse():
        node_v.scale_radius(factor)

    # Create mapping from original nodes to scaled visual nodes
    def create_node_mapping(
        orig_node: Node, vis_node: VisualNode, mapping: Dict[int, VisualNode]
    ) -> None:
        mapping[id(orig_node)] = vis_node

        # Map children recursively
        if hasattr(orig_node, "children") and hasattr(vis_node, "children"):
            for orig_child, vis_child in zip(orig_node.children, vis_node.children):
                create_node_mapping(orig_child, vis_child, mapping)

    node_mapping: Dict[int, VisualNode] = {}
    create_node_mapping(root, visual_node, node_mapping)

    def traverse_and_mark_zero_branches(node: Node) -> None:
        """Recursively traverse tree and mark nodes at the end of zero-length branches."""
        if not hasattr(node, "children"):
            return

        for child in node.children:
            # Check for zero-length branch
            branch_length = _get_branch_length(child)
            if abs(branch_length) < 1e-6:
                # The child is at the end of a zero-length branch, so we mark it.
                visual_node_match = node_mapping.get(id(child))
                if visual_node_match and hasattr(visual_node_match, "cartesian"):
                    x, y = visual_node_match.cartesian()
                    ET.SubElement(
                        group,
                        "circle",
                        {
                            "cx": str(x),
                            "cy": str(y),
                            "r": str(indicator_size / 2),
                            "fill": indicator_color,
                            "stroke": "#000",
                            "stroke-width": "0.5",
                            "opacity": "0.8",
                        },
                    )

            # Continue traversal
            traverse_and_mark_zero_branches(child)

    traverse_and_mark_zero_branches(root)


def _render_trees_and_collect_coords(
    roots: List[Node],
    trees_group: ET.Element,
    size: int,
    margin: int,
    label_offset: int,
    ignore_branch_lengths: bool,
    font_family: str,
    font_size: str,
    stroke_color: str,
    svg_root: ET.Element,
    highlight_branches: Optional[list] = None,
    highlight_width: Optional[list] = None,
    highlight_colors: Optional[list] = None,
    show_zero_length_indicators: bool = False,
    zero_length_indicator_color: str = "#ff4444",
    zero_length_indicator_size: float = 6.0,
) -> List[Dict[str, Tuple[float, float]]]:
    """Render each tree and collect leaf coordinates."""
    all_trees_leaf_coords: List[Dict[str, Tuple[float, float]]] = []
    for i, root_node in enumerate(roots):
        cx = i * size + size / 2
        cy = size / 2
        group = ET.SubElement(trees_group, "g", {"transform": f"translate({cx}, {cy})"})
        # Pick per-tree highlight config if available
        per_tree_highlight_branches = (
            highlight_branches[i]
            if highlight_branches and i < len(highlight_branches)
            else None
        )
        per_tree_highlight_width = (
            highlight_width[i] if highlight_width and i < len(highlight_width) else 4.0
        )
        per_tree_highlight_colors = (
            highlight_colors[i]
            if highlight_colors and i < len(highlight_colors)
            else None
        )
        scaled_leaf_nodes = render_single_circular_tree(
            root=root_node,
            group=group,
            size=size,
            margin=margin,
            label_offset=label_offset,
            ignore_branch_lengths=ignore_branch_lengths,
            font_family=font_family,
            font_size=font_size,
            stroke_color=stroke_color,
            svg_root=svg_root,  # Pass svg_root here
            highlight_branches=per_tree_highlight_branches,
            highlight_width=per_tree_highlight_width,
            highlight_colors=per_tree_highlight_colors,
        )

        # Add zero-length branch indicators if requested
        if show_zero_length_indicators:
            _add_zero_length_indicators_simple(
                root_node,
                group,
                zero_length_indicator_color,
                zero_length_indicator_size,
                size,
                margin,
                label_offset,
                ignore_branch_lengths,
            )

        current_tree_leaf_coords: Dict[str, Tuple[float, float]] = {}
        for leaf_node in scaled_leaf_nodes:
            relative_x, relative_y = leaf_node.cartesian()
            absolute_x = relative_x + cx
            absolute_y = relative_y + cy
            current_tree_leaf_coords[leaf_node.name] = (absolute_x, absolute_y)
        all_trees_leaf_coords.append(current_tree_leaf_coords)
    return all_trees_leaf_coords


###############################################################################
# Circular Tree Bézier Connection Utilities
###############################################################################


def _create_bezier_path(x1, y1, x2, y2, control_point_offset):
    """Return SVG path string for a cubic Bézier curve between (x1, y1) and (x2, y2)."""
    cp1x = x1 + control_point_offset
    cp1y = y1
    cp2x = x2 - control_point_offset
    cp2y = y2
    return f"M {x1},{y1} C {cp1x},{cp1y} {cp2x},{cp2y} {x2},{y2}"


def _add_svg_gradient(defs, grad_id, x1, y1, x2, y2, color):
    """Add a linear gradient to SVG defs and return its id."""
    grad = ET.SubElement(
        defs,
        "linearGradient",
        {
            "id": grad_id,
            "gradientUnits": "userSpaceOnUse",
            "x1": str(x1),
            "y1": str(y1),
            "x2": str(x2),
            "y2": str(y2),
        },
    )
    ET.SubElement(grad, "stop", {"offset": "0%", "stop-color": "#e0e0e0"})
    ET.SubElement(grad, "stop", {"offset": "100%", "stop-color": color})
    return grad_id


def _add_svg_glow_filter(defs, filter_id, stddev="2.5"):
    """Add a glow filter to SVG defs."""
    flt = ET.SubElement(
        defs,
        "filter",
        {"id": filter_id, "x": "-20%", "y": "-20%", "width": "140%", "height": "140%"},
    )
    ET.SubElement(
        flt,
        "feGaussianBlur",
        {"in": "SourceGraphic", "stdDeviation": stddev, "result": "blur"},
    )
    merge = ET.SubElement(flt, "feMerge")
    merge.extend(
        [
            ET.Element("feMergeNode", {"in": "blur"}),
            ET.Element("feMergeNode", {"in": "SourceGraphic"}),
        ]
    )


def _get_or_create_defs(parent: ET.Element) -> ET.Element:
    """Get or create the <defs> element inside the parent element."""
    return parent.find("defs") or ET.SubElement(parent, "defs")


def _maybe_add_glow_filter(defs: ET.Element, glow: bool) -> Optional[str]:
    """Add a glow filter to defs if requested, return its id or None."""
    if not glow:
        return None
    glow_filter_id = f"glow-{uuid.uuid4().hex}"
    _add_svg_glow_filter(defs, glow_filter_id)
    return glow_filter_id


def _draw_bezier_connections_between_pair(
    coords_tree1: Dict[str, Tuple[float, float]],
    coords_tree2: Dict[str, Tuple[float, float]],
    connections_group: ET.Element,
    defs: ET.Element,
    size: int,
    color_list: Optional[List[str]],
    width_list: Optional[List[float]],
    glow_filter_id: Optional[str],
) -> None:
    """Draw all Bézier connections between two trees' leaf coordinates."""
    control_point_offset = size / 3.0
    for leaf_idx, (leaf_name, (x1, y1)) in enumerate(coords_tree1.items()):
        if leaf_name not in coords_tree2:
            continue
        x2, y2 = coords_tree2[leaf_name]
        color = (
            color_list[leaf_idx]
            if color_list and leaf_idx < len(color_list)
            else CONNECTION_STROKE_COLOR
        )
        stroke_width = (
            width_list[leaf_idx]
            if width_list and leaf_idx < len(width_list)
            else CONNECTION_STROKE_WIDTH
        )
        path_d = _create_bezier_path(x1, y1, x2, y2, control_point_offset)
        _draw_bezier_connection(
            connections_group,
            defs,
            path_d,
            x1,
            y1,
            x2,
            y2,
            color,
            stroke_width,
            glow_filter_id,
        )
        # Endpoint marker for all moved taxa (distance > 0)
        if (
            width_list
            and leaf_idx < len(width_list)
            and width_list[leaf_idx] > CONNECTION_STROKE_WIDTH
        ):
            _draw_bezier_endpoint_marker(
                connections_group, x2, y2, stroke_width, color, glow_filter_id
            )


def _draw_bezier_connection(
    connections_group: ET.Element,
    defs: ET.Element,
    path_d: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str,
    stroke_width: float,
    glow_filter_id: Optional[str],
) -> None:
    """Draw a single Bézier connection with halo and gradient."""
    # Halo effect
    halo_attrs = _create_path_attributes(path_d, "#fff", stroke_width + 4, "0.8")
    halo_attrs["stroke-dasharray"] = CONNECTION_STROKE_DASHARRAY
    if glow_filter_id:
        halo_attrs["filter"] = f"url(#{glow_filter_id})"
    add_svg_path(connections_group, halo_attrs)

    # Gradient effect
    grad_id = f"bezier-gradient-{uuid.uuid4().hex}"
    _add_svg_gradient(defs, grad_id, x1, y1, x2, y2, color)
    attrs = _create_path_attributes(path_d, f"url(#{grad_id})", stroke_width)
    attrs["stroke-dasharray"] = CONNECTION_STROKE_DASHARRAY
    if glow_filter_id:
        attrs["filter"] = f"url(#{glow_filter_id})"
    add_svg_path(connections_group, attrs)


def _draw_bezier_endpoint_marker(
    connections_group: ET.Element,
    x: float,
    y: float,
    stroke_width: float,
    color: str,
    glow_filter_id: Optional[str],
) -> None:
    """Draw a circle marker at the endpoint of a Bézier connection."""
    marker_attrs = {
        "cx": str(x),
        "cy": str(y),
        "r": str(stroke_width * 1.5),
        "fill": color,
        "stroke": "#222",
        "stroke-width": "1.5",
        "opacity": "0.95",
    }
    if glow_filter_id:
        marker_attrs["filter"] = f"url(#{glow_filter_id})"
    ET.SubElement(connections_group, "circle", marker_attrs)


def _draw_bezier_connections(
    all_trees_leaf_coords: List[Dict[str, Tuple[float, float]]],
    connections_group: ET.Element,
    size: int,
    bezier_colors: Optional[List[List[str]]] = None,
    bezier_stroke_widths: Optional[List[List[float]]] = None,
    glow: bool = False,
) -> None:
    """
    Draw Bézier curve connections between adjacent trees, with optional color and stroke width per connection.
    Adds a halo (shadow) effect, SVG gradients, and endpoint markers for all moved taxa.
    Optionally adds a glow effect to connections and markers.
    """
    if len(all_trees_leaf_coords) <= 1:
        return
    defs = _get_or_create_defs(connections_group)
    glow_filter_id = _maybe_add_glow_filter(defs, glow) if glow else None
    for i in range(len(all_trees_leaf_coords) - 1):
        _draw_bezier_connections_between_pair(
            all_trees_leaf_coords[i],
            all_trees_leaf_coords[i + 1],
            connections_group,
            defs,
            size,
            bezier_colors[i] if bezier_colors and i < len(bezier_colors) else None,
            bezier_stroke_widths[i]
            if bezier_stroke_widths and i < len(bezier_stroke_widths)
            else None,
            glow_filter_id,
        )


###############################################################################
# Main SVG Generation for Multiple Circular Trees
###############################################################################
def generate_multiple_circular_trees_svg(
    roots: List[Node],
    size: int = 200,
    margin: int = 30,
    label_offset: int = 2,
    ignore_branch_lengths: bool = False,
    font_family: str = DEFAULT_FONT_FAMILY,
    font_size: str = DEFAULT_FONT_SIZE,
    stroke_color: str = DEFAULT_STROKE_COLOR,
    bezier_colors: Optional[List[List[str]]] = None,
    bezier_stroke_widths: Optional[List[List[float]]] = None,
    glow: bool = False,
    highlight_branches: Optional[list] = None,  # List of lists, one per tree
    highlight_width: Optional[list] = None,  # List of floats, one per tree
    highlight_colors: Optional[list] = None,  # List of dicts, one per tree
    show_zero_length_indicators: bool = False,
    zero_length_indicator_color: str = "#ff4444",
    zero_length_indicator_size: float = 6.0,
) -> Tuple[ET.Element, List[Dict[str, Tuple[float, float]]]]:
    """
    Create an SVG with multiple circular trees laid out horizontally.
    Returns the SVG root element and a list of dictionaries mapping leaf names to absolute coordinates.
    Optionally, color and stroke width for Bézier connections can be set per connection, per tree-pair.
    Set `glow=True` for a subtle glow effect on connections and markers.
    highlight_branches: list of lists (per tree) or None
    highlight_width: list of floats (per tree) or None
    highlight_colors: list of dicts (per tree) or None
    """
    if len(roots) == 0:
        empty_svg = get_svg_root(width=1, height=1)
        return empty_svg, []
    svg_root, trees_group, connections_group = _create_svg_root_and_groups(
        len(roots), size
    )
    all_trees_leaf_coords = _render_trees_and_collect_coords(
        roots,
        trees_group,
        size,
        margin,
        label_offset,
        ignore_branch_lengths,
        font_family,
        font_size,
        stroke_color,
        svg_root=svg_root,  # Pass svg_root here
        highlight_branches=highlight_branches,
        highlight_width=highlight_width,
        highlight_colors=highlight_colors,
        show_zero_length_indicators=show_zero_length_indicators,
        zero_length_indicator_color=zero_length_indicator_color,
        zero_length_indicator_size=zero_length_indicator_size,
    )
    _draw_bezier_connections(
        all_trees_leaf_coords,
        connections_group,
        size,
        bezier_colors=bezier_colors,
        bezier_stroke_widths=bezier_stroke_widths,
        glow=glow,
    )
    return svg_root, all_trees_leaf_coords


def plot_circular_tree(
    root: Node,
    size: int = 400,
    margin: int = 30,
    label_offset: int = 2,
    ignore_branch_lengths: bool = False,
    font_family: str = DEFAULT_FONT_FAMILY,
    font_size: str = DEFAULT_FONT_SIZE,
    stroke_color: str = DEFAULT_STROKE_COLOR,
    output_path: Optional[str] = None,
) -> ET.Element:
    """
    Plot a single circular tree.

    Args:
        root: The root node of the tree
        size: Size of the plot
        margin: Margin around the tree
        label_offset: Offset for labels
        ignore_branch_lengths: If True, treat each branch as length=1
        font_family: Font family for labels
        font_size: Font size for labels
        stroke_color: Color for tree branches
        output_path: Optional path to save the SVG

    Returns:
        SVG root element
    """
    # Use the existing function to generate SVG for a single tree
    svg_root, _ = generate_multiple_circular_trees_svg(
        roots=[root],
        size=size,
        margin=margin,
        label_offset=label_offset,
        ignore_branch_lengths=ignore_branch_lengths,
        font_family=font_family,
        font_size=font_size,
        stroke_color=stroke_color,
    )

    # Save to file if path provided
    if output_path:
        tree = ET.ElementTree(svg_root)
        tree.write(output_path, encoding="unicode", xml_declaration=True)

    return svg_root
