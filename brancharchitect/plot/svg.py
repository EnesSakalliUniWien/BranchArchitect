import math
import xml.etree.ElementTree as ET
from typing import List, Generator, Dict, Tuple, Optional, TypedDict
from brancharchitect.tree import Node  # Cleaned: use Node from tree.py
from brancharchitect.plot.tree_utils import (tree_depth, get_node_label, traverse, get_node_length, is_leaf, get_order, get_node_id)

class LeafCoordinates(TypedDict):
    node: Tuple[float, float]
    branch_end: Tuple[float, float]
    label: Tuple[float, float]
    y: float
    connector_end: Tuple[float, float]
    text_anchor: str


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


###############################################################################
# Node Utilities
###############################################################################


###############################################################################
# SVG Element Creation Utilities
###############################################################################
def get_svg_root(width: float, height: float) -> ET.Element:
    """
    Creates an SVG root element with a 0,0 origin and given width/height.
    """
    data = {
        "viewBox": f"0 0 {width} {height}",
        "version": "1.1",
        "xmlns": "http://www.w3.org/2000/svg",
        "xmlns:xlink": "http://www.w3.org/1999/xlink",
        "xml:space": "preserve",
        "width": str(width),
        "height": str(height),
    }
    return ET.Element("svg", data)


def add_svg_path(parent: ET.Element, attrs: Dict[str, str]) -> ET.Element:
    """Add a path element to the parent and return the created element."""
    return ET.SubElement(parent, "path", attrs)


def add_labels(parent: ET.Element, order: List[str], radius: float) -> None:
    """Add circular labels around the outer radius for radial tree layouts."""
    count = len(order)
    if count == 0:
        return

    for i, name in enumerate(order):
        angle_deg = (360.0 * i) / count
        anchor = "end" if 90 < angle_deg < 270 else "start"
        rotate_label = 180 if 90 < angle_deg < 270 else 0

        text_el = ET.SubElement(
            parent,
            "text",
            {
                "text-anchor": anchor,
                "transform": f"rotate({angle_deg}) translate({radius},0) rotate({rotate_label})",
                "font-size": DEFAULT_FONT_SIZE,
                "font-weight": "normal",
                "font-family": DEFAULT_FONT_FAMILY,
                "style": f"fill:{DEFAULT_STROKE_COLOR}",
                "dominant-baseline": "middle",
            },
        )
        text_el.text = name


###############################################################################
# Visual Node Class for Radial Layouts
###############################################################################


class VisualNode:
    """Represents a node's visual properties for radial tree layouts."""

    def __init__(
        self, radius: float, angle: float, children: Optional[List["VisualNode"]] = None
    ):
        if children is None:
            children = []
        self.radius = radius
        self.angle = angle
        self.children = children

    def cartesian(self) -> Tuple[float, float]:
        """Convert polar coordinates (radius, angle) to cartesian (x, y)."""
        return (self.radius * math.cos(self.angle), self.radius * math.sin(self.angle))

    def scale_radius(self, factor: float) -> None:
        """Scale the radius by the given factor."""
        self.radius *= factor

    def traverse(self) -> Generator["VisualNode", None, None]:
        """Traverse this visual node and all descendants."""
        yield self
        for child in self.children:
            yield from child.traverse()


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
    node_radius = parent_radius + (
        1.0 if ignore_branch_lengths else get_node_length(root)
    )

    if is_leaf(root):
        # Leaf angle based on position in `order`
        idx = order.index(get_node_label(root)) if order else 0
        leaf_angle = (total_angle / len(order)) * idx if order else 0.0
        return VisualNode(node_radius, leaf_angle, [])
    else:
        # Internal node: build children first
        child_visuals = [
            tree_to_circular_visual_nodes(
                child, total_angle, order, node_radius, ignore_branch_lengths
            )
            for child in root.children
        ]
        # The node's own angle is the average of child angles
        if len(child_visuals) > 0:
            angle_sum = sum(cv.angle for cv in child_visuals)
            node_angle = angle_sum / len(child_visuals)
        else:
            node_angle = 0.0
        vnode = VisualNode(node_radius, node_angle, child_visuals)
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


def generate_circular_links(node: VisualNode) -> List[Dict[str, str]]:
    """
    Recursively build SVG path attributes for links from node to each child in a circular tree.

    Args:
        node: The visual node to generate links from

    Returns:
        List of SVG path attribute dictionaries
    """
    all_paths = []
    for child in node.children:
        attrs = {
            "class": "links",
            "stroke-width": str(STROKE_WIDTH),
            "fill": "none",
            "style": f"stroke-opacity:1;stroke:{DEFAULT_STROKE_COLOR}",
            "d": build_circular_link_path(node, child),
        }
        all_paths.append(attrs)
        all_paths.extend(generate_circular_links(child))
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
) -> Dict:
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
) -> str:
    """
    Create an SVG with exactly two circular trees side by side.

    Args:
        root1: First tree root node
        root2: Second tree root node
        size: Size of each tree's container (width & height)
        margin: Margin around each tree
        label_offset: Extra space for labels
        ignore_branch_lengths: If True, use unit length for all branches

    Returns:
        SVG as a string
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
    )

    # Render second tree in the right group
    render_single_circular_tree(
        root=root2,
        group=group2,
        size=size,
        margin=margin,
        label_offset=label_offset,
        ignore_branch_lengths=ignore_branch_lengths,
    )

    return ET.tostring(svg_root, encoding="utf8").decode("utf8")


def render_single_circular_tree(
    root: Node,
    group: ET.Element,
    size: int,
    margin: int,
    label_offset: int,
    ignore_branch_lengths: bool = False,
) -> None:
    """
    Render a single circular tree within the provided SVG group element.

    Args:
        root: Root node of the tree to render
        group: SVG group element to add the tree to
        size: Size of the tree's container
        margin: Margin around the tree
        label_offset: Extra space for labels
        ignore_branch_lengths: If True, use unit length for all branches
    """
    # Get leaf order
    order = get_order(root)

    # Calculate coordinates for this tree
    tree_data = calculate_circular_tree_coordinates(
        root, 2 * math.pi, order, 0, ignore_branch_lengths
    )

    # Get the visual node for scaling
    visual_node = tree_data["visual_node"]

    # Scale to fit
    max_r = max(node.radius for node in visual_node.traverse())
    usable_radius = size / 2 - margin - label_offset
    factor = usable_radius / max_r if max_r != 0 else 1.0

    # Apply scaling
    for node_v in visual_node.traverse():
        node_v.scale_radius(factor)

    # Add links
    links = generate_circular_links(visual_node)
    for attrs in links:
        add_svg_path(group, attrs)

    # Add labels
    add_labels(group, order, (size / 2 - margin))


def generate_multiple_circular_trees_svg(
    roots: List[Node],
    size: int = 200,
    margin: int = 30,
    label_offset: int = 2,
    ignore_branch_lengths: bool = False,
) -> str:
    """
    Create an SVG with multiple circular trees laid out horizontally.

    Args:
        roots: List of tree root nodes
        size: Size of each tree's container (width & height)
        margin: Margin around each tree
        label_offset: Extra space for labels
        ignore_branch_lengths: If True, use unit length for all branches

    Returns:
        SVG as a string
    """
    n = len(roots)
    if n == 0:
        raise ValueError("No trees provided.")

    total_width = n * size
    total_height = size
    svg_root = get_svg_root(total_width, total_height)

    for i, root in enumerate(roots):
        # Center each tree in its own "cell"
        cx = i * size + size / 2
        cy = size / 2
        group = ET.SubElement(svg_root, "g", {"transform": f"translate({cx}, {cy})"})

        # Render this tree in its group
        render_single_circular_tree(
            root=root,
            group=group,
            size=size,
            margin=margin,
            label_offset=label_offset,
            ignore_branch_lengths=ignore_branch_lengths,
        )

    return ET.tostring(svg_root, encoding="utf8").decode("utf8")


###############################################################################
# Rectangular Layout Tree Coordinates Helper
###############################################################################


def generate_multiple_rectangular_trees_svg(
    roots: List[Node],
    size: int = 200,
    margin: int = 30,
    label_offset: int = 2,
    ignore_branch_lengths: bool = False,
) -> str:
    """
    Create an SVG with multiple circular trees laid out horizontally.

    Args:
        roots: List of tree root nodes
        size: Size of each tree's container (width & height)
        margin: Margin around each tree
        label_offset: Extra space for labels
        ignore_branch_lengths: If True, use unit length for all branches

    Returns:
        SVG as a string
    """
    n = len(roots)
    if n == 0:
        raise ValueError("No trees provided.")

    total_width = n * size
    total_height = size
    svg_root = get_svg_root(total_width, total_height)

    for i, root in enumerate(roots):
        # Center each tree in its own "cell"
        cx = i * size + size / 2
        cy = size / 2
        group = ET.SubElement(svg_root, "g", {"transform": f"translate({cx}, {cy})"})

        # Render this tree in its group
        render_single_rectangular_tree(
            root=root,
            group=group,
            margin=margin,
            label_offset=label_offset,
            ignore_branch_lengths=ignore_branch_lengths,
        )

    return ET.tostring(svg_root, encoding="utf8").decode("utf8")


def assign_rectangular_positions(
    node: Node,
    node_coords: Dict[Node, Dict[str, float]],
    horizontal_spacing: float,
    vertical_spacing: float,
    depth=0,
    leaf_index=[0],  # Use list as a mutable counter reference
) -> Tuple[float, float]:
    """
    Assign x,y coordinates to each node recursively in a rectangular tree layout.
    Stores coordinates in node_coords.

    Args:
        node: The current tree node
        node_coords: Dictionary to store node coordinates
        horizontal_spacing: Horizontal spacing between leaves
        vertical_spacing: Vertical spacing between levels
        depth: Current depth in the tree
        leaf_index: Counter for leaf nodes (as a mutable list reference)

    Returns:
        Tuple of (x, y) coordinates for the current node
    """
    # Calculate y based on depth
    y = depth * vertical_spacing

    if not node.children:
        # Leaf node: calculate x based on leaf order
        x = (leaf_index[0] + 1) * horizontal_spacing
        leaf_index[0] += 1

        # Store leaf coordinates

        node_coords[node] = {"x": x, "y": y}
        return (x, y)
    else:
        # Internal node: process children first to get their positions
        child_positions = []
        for child in node.children:
            pos = assign_rectangular_positions(
                child,
                node_coords,
                horizontal_spacing,
                vertical_spacing,
                depth + 1,
                leaf_index,
            )
            child_positions.append(pos)  # Store (child_x, child_y)

        # Position internal node x at the average of its children's x
        x = sum(pos[0] for pos in child_positions) / len(child_positions)

        # Store internal node coordinates and its direct children's coordinates
        node_coords[node] = {
            "x": x,
            "y": y,
            "children_coords": child_positions,  # Store list of (x,y) tuples
        }
        return (x, y)


###############################################################################
# Rectangular Layout Tree Visualization
###############################################################################
def calculate_rectangular_tree_coordinates(
    root: Node, width: int, height: int, leaf_count: int
) -> Tuple[Dict[Node, Dict[str, float]], float]:  # Return coords and label_y
    """
    Calculate coordinates for nodes in a rectangular tree layout.

    Args:
        root: The root node of the tree
        width: Available width for drawing area
        height: Available height for drawing area
        leaf_count: Total number of leaf nodes

    Returns:
        Tuple containing:
            - Dictionary mapping each node to its coordinates ({'x': float, 'y': float, ...})
            - The calculated y-coordinate for labels (label_y)
    """
    # Use the correct depth calculation function
    max_depth = tree_depth(root)

    # Define margins and calculate available space
    margin_top = 20  # Keep consistent with the group transform later
    margin_bottom = 50  # Space for labels
    drawable_height = height - margin_top - margin_bottom
    drawable_width = width  # Use full width for horizontal spacing calculation

    # Calculate spacings
    # Add 1 to leaf_count for padding on sides
    horizontal_spacing = (
        drawable_width / (leaf_count + 1) if leaf_count > 0 else drawable_width / 2
    )
    # Add 1 to max_depth because depth is 0-based (levels = depth + 1)
    # Avoid division by zero if tree is just a single node (max_depth=0)
    vertical_spacing = (
        drawable_height / max_depth if max_depth > 0 else drawable_height / 2
    )

    # Calculate Y position for labels at the bottom
    label_y = margin_top + drawable_height + 20  # Place labels below the main tree area

    # Store coordinates for all nodes
    node_coords = {}
    leaf_index = [0]  # Mutable counter for leaf positioning

    # Assign positions recursively
    assign_rectangular_positions(
        node=root,
        node_coords=node_coords,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
        depth=0,
        leaf_index=leaf_index,
    )

    return node_coords, label_y

def draw_cut_line(group, x1, y1, x2, y2, slash_length=8, stroke="red", stroke_width="2"):
        """
        Draw a short perpendicular line (slash) across the midpoint 
        of the edge from (x1,y1) to (x2,y2).
        """
        # Midpoint of the edge
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0

        # Vector from parent to child
        dx = x2 - x1
        dy = y2 - y1

        # Perpendicular direction: ( -dy, dx ) or ( dy, -dx )
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return  # No slash if the edge is degenerate

        # Perpendicular direction (unit vector)
        px = -dy / length
        py =  dx / length

        # slash_length is the full size; half each side
        half = slash_length / 2.0
        # Scale perp vector by half
        px *= half
        py *= half

        # Endpoints of slash
        xA = mx - px
        yA = my - py
        xB = mx + px
        yB = my + py

        # Add the slash line to the SVG
        ET.SubElement(
            group,
            "line",
            {
                "x1": str(xA),
                "y1": str(yA),
                "x2": str(xB),
                "y2": str(yB),
                "stroke": stroke,
                "stroke-width": stroke_width,
            },
        )

def render_single_rectangular_tree(
    root: Node,
    group: ET.Element,
    node_coords: Dict[Node, Dict[str, float]],
    label_y: float,
    cut_edges: Optional[set] = None,  # New parameter to specify edges to cut
    leaf_label_offset: float = 0.0,   # New parameter for label offset
) -> None:
    """
    Render a single rectangular tree within the provided SVG group element
    using pre-calculated coordinates.

    Args:
        root: Root node of the tree to render (used implicitly via node_coords)
        group: SVG group element to add the tree to
        node_coords: Dictionary of node coordinates calculated earlier
        label_y: Y-coordinate for placing leaf labels
        cut_edges: Set of edges (parent_label, child_label) to cut
        leaf_label_offset: Offset for leaf label and dashed connector
    """
    cut_edges = cut_edges or set()

   

    # Draw the tree using calculated coordinates
    for node, coords in node_coords.items():
        x, y = coords["x"], coords["y"]

        if "children_coords" not in coords:  # It's a leaf node
            # Draw the main branch line (solid)
            branch_end_y = (
                y + (label_y - y) * 0.4
            )  # End branch at 40% of the way to label (unchanged)

            ET.SubElement(
                group,
                "path",
                {
                    "d": f"M {x},{y} V {branch_end_y}",
                    "fill": "none",
                    "stroke": DEFAULT_RECT_STROKE_COLOR,
                    "stroke-width": "1.5",
                },
            )

            # Draw dashed connector from branch end to near label (no offset)
            ET.SubElement(
                group,
                "path",
                {
                    "d": f"M {x},{branch_end_y} V {label_y - 10}",  # No leaf_label_offset here
                    "fill": "none",
                    "stroke": DEFAULT_RECT_STROKE_COLOR,
                    "stroke-width": "1.5",
                    "stroke-dasharray": "4,4",
                    "stroke-opacity": "0.7",
                },
            )

            # Add label text below the connector (with offset)
            ET.SubElement(
                group,
                "text",
                {
                    "x": str(x),
                    "y": str(label_y + leaf_label_offset),  # Only the label moves
                    "font-family": DEFAULT_FONT_FAMILY,
                    "font-size": DEFAULT_FONT_SIZE,
                    "fill": DEFAULT_RECT_STROKE_COLOR,
                    "text-anchor": "middle",
                    "dominant-baseline": "middle",
                },
            ).text = get_node_label(node)

        else:  # It's an internal node
            # Draw connections to children
            child_positions = coords["children_coords"]
            for child_x, child_y in child_positions:
                # Draw vertical line from parent (x,y) down to child's y level
                ET.SubElement(
                    group,
                    "path",
                    {
                        "d": f"M {x},{y} V {child_y}",
                        "fill": "none",
                        "stroke": DEFAULT_RECT_STROKE_COLOR,
                        "stroke-width": "1.5",
                    },
                )
                # Draw horizontal line from parent's x to child's x at child's y level
                ET.SubElement(
                    group,
                    "path",
                    {
                        "d": f"M {x},{child_y} H {child_x}",
                        "fill": "none",
                        "stroke": DEFAULT_RECT_STROKE_COLOR,
                        "stroke-width": "1.5",
                    },
                )

                # Check if this edge needs a cut line
                parent_label = get_node_label(node)
                child_label = get_node_label(next(
                    (n for n in node_coords if node_coords[n]["x"] == child_x and node_coords[n]["y"] == child_y),
                    None
                ))
                if (parent_label, child_label) in cut_edges:
                    draw_cut_line(group, x, y, child_x, child_y)


def generate_rectangular_tree_svg(
    root: Node, width: Optional[int] = None, height: Optional[int] = None
) -> str:
    """
    Generate a classical rectangular layout tree with vertical orientation.

    Args:
        root: The root node of the tree
        width: Width of the SVG. Defaults to 600 if None for better spacing.
        height: Height of the SVG. Defaults to 400 if None.

    Returns:
        SVG as a string
    """
    # Set default values if None is provided. Increased defaults for better visibility.
    width = width if width is not None else 600
    height = height if height is not None else 400

    svg_root = get_svg_root(width, height)

    # Get all leaves for horizontal spacing calculation
    leaves = [n for n in traverse(root) if not n.children]
    leaf_count = len(leaves)

    # Handle the edge case of a tree with only a root node
    if leaf_count == 0 and root is not None:
        leaf_count = 1  # Treat the root as a single leaf for spacing

    if not root:
        return ET.tostring(svg_root, encoding="utf8").decode("utf8")  # Return empty SVG if no root

    # Calculate node coordinates and label y position
    node_coords, label_y = calculate_rectangular_tree_coordinates(
        root, width, height, leaf_count
    )

    # Create main group with top margin and default drawing styles
    group = ET.SubElement(
        svg_root,
        "g",
        {
            "transform": f"translate(0, {20})",  # Consistent with margin_top in calculation
            "stroke": DEFAULT_RECT_STROKE_COLOR,
            "stroke-width": "1.5",
            "fill": "none",  # Ensure shapes aren't filled by default
        },
    )

    # Render the tree within the group, passing the calculated label_y
    render_single_rectangular_tree(root, group, node_coords, label_y)

    return ET.tostring(svg_root, encoding="utf8").decode("utf8")


###############################################################################
# Tanglegram Coordinate Calculation
###############################################################################
def calculate_tanglegram_coordinates(
    left_tree: Node, right_tree: Node, tree_width: float, usable_height: float
) -> Dict:
    """
    Calculate coordinates for a tanglegram visualization.

    Args:
        left_tree: Left tree root node
        right_tree: Right tree root node
        tree_width: Width available for each tree
        usable_height: Available height for the visualization

    Returns:
        Dictionary with node coordinates and connection data
    """
    # Get all leaves to calculate vertical spacing
    left_leaves = [n for n in traverse(left_tree) if not n.children]
    right_leaves = [n for n in traverse(right_tree) if not n.children]
    total_leaves = max(len(left_leaves), len(right_leaves))

    # Calculate spacings
    vertical_spacing = usable_height / (total_leaves + 1)

    # Calculate node positions for both trees without drawing
    left_coords = calculate_tree_side_coordinates(
        left_tree, tree_width, vertical_spacing, usable_height, is_left=True
    )

    right_coords = calculate_tree_side_coordinates(
        right_tree, tree_width, vertical_spacing, usable_height, is_left=False
    )

    # Find matching leaves for connections
    connections = []
    # Only match nodes that have the 'y' key in both trees (ensures they are leaves)
    matching_nodes = {
        node_id: coords
        for node_id, coords in left_coords.items()
        if node_id in right_coords and "y" in coords and "y" in right_coords[node_id]
    }

    # Calculate connection paths between matching leaves
    for name in matching_nodes:
        left_y = left_coords[name]["y"]
        right_y = right_coords[name]["y"]

        # Get label positions
        left_label_x = left_coords[name]["label"][0]
        right_label_x = 2 * tree_width + right_coords[name]["label"][0]

        # Calculate bezier curve control points
        total_width = right_label_x - left_label_x
        mid_x = left_label_x + total_width / 2
        curve_height = vertical_spacing * 0.5

        # Store connection data
        connections.append(
            {
                "node_id": name,
                "left_x": left_label_x,
                "left_y": left_y,
                "right_x": right_label_x,
                "right_y": right_y,
                "mid_x": mid_x,
                "curve_height": curve_height,
            }
        )

    return {
        "left_coords": left_coords,
        "right_coords": right_coords,
        "connections": connections,
        "vertical_spacing": vertical_spacing,
    }


###############################################################################
# Tanglegram Tree Side Coordinate Helper
###############################################################################
def assign_tanglegram_tree_positions(
    node: Node,
    leaf_coords: Dict[str, Dict],
    width: float,
    main_width: float,
    horizontal_spacing: float,
    vertical_spacing: float,
    label_offset: float,
    is_left: bool,
    text_anchor: str,
    leaf_index: List[int],
    depth: int = 0,
) -> Tuple[float, float]:
    """
    Assign positions to nodes in one side of a tanglegram.

    Args:
        node: Current node being processed
        leaf_coords: Dictionary to store node coordinates
        width: Total width available for this tree
        main_width: Width for the main tree structure (excluding labels)
        horizontal_spacing: Horizontal spacing between levels
        vertical_spacing: Vertical spacing between nodes
        label_offset: Offset for labels from the tree
        is_left: Whether this is the left tree (True) or right tree (False)
        text_anchor: Text anchor attribute for labels
        leaf_index: Counter for leaf nodes (as a mutable list)
        depth: Current depth in the tree

    Returns:
        Tuple of (x, y) coordinates for the current node
    """
    # Calculate x based on depth and side
    if is_left:
        x = depth * horizontal_spacing
        label_x_pos = main_width + label_offset
        branch_end_x = x + horizontal_spacing
        connector_end_x = label_x_pos - 10  # Space before label
    else:
        x = width - (depth * horizontal_spacing)
        label_x_pos = -label_offset
        branch_end_x = x - horizontal_spacing
        connector_end_x = label_x_pos + 10  # Space before label

    if not node.children:  # Leaf node
        # Calculate y position
        y = (leaf_index[0] + 1) * vertical_spacing
        leaf_index[0] += 1

        # Use consistent identification function
        node_id = get_node_id(node)

        # Store coordinates with consistent identifier
        leaf_coords[node_id] = {
            "node": (x, y),
            "branch_end": (branch_end_x, y),
            "label": (label_x_pos, y),
            "y": y,  # Store y position for easier access
            "connector_end": (connector_end_x, y),
            "text_anchor": text_anchor,
        }

        return (x, y)

    # Process children first
    child_positions = []
    for child in node.children:
        pos = assign_tanglegram_tree_positions(
            child,
            leaf_coords,
            width,
            main_width,
            horizontal_spacing,
            vertical_spacing,
            label_offset,
            is_left,
            text_anchor,
            leaf_index,
            depth + 1,
        )
        if pos:
            child_positions.append(pos)

    if not child_positions:
        return None

    # Calculate y as average of children
    y = sum(p[1] for p in child_positions) / len(child_positions)

    # Store internal node data for drawing
    child_lines = []
    for child_x, child_y in child_positions:
        child_lines.append(
            {
                "horiz": {"x1": x, "y1": y, "x2": child_x, "y2": y},
                "vert": {"x1": child_x, "y1": y, "x2": child_x, "y2": child_y},
            }
        )

    # Store in leaf_coords even though it's an internal node
    node_id = f"internal-{len(leaf_coords)}"
    leaf_coords[node_id] = {
        "node": (x, y),
        "children": child_lines,
        "is_internal": True,
    }

    return (x, y)


###############################################################################
# Tree Depth Calculation
###############################################################################



def calculate_tree_side_coordinates(
    root: Node,
    width: float,
    vertical_spacing: float,
    height: float,
    is_left: bool = True,
) -> Dict[str, LeafCoordinates]:
    """
    Calculate coordinates for one side of the tanglegram without drawing.

    Args:
        root: Tree root node
        width: Available width for this tree
        vertical_spacing: Vertical distance between nodes
        height: Total available height
        is_left: True if this is the left tree, False for right tree

    Returns:
        Dictionary mapping node IDs to coordinates
    """
    # Calculate tree depth
    max_depth = tree_depth(root)

    # Calculate spacings
    main_width = width * 0.8  # Leave space for labels
    horizontal_spacing = main_width / (max_depth + 1)
    label_offset = width * 0.15

    # Label position based on side
    if is_left:
        label_x = main_width + label_offset
        text_anchor = "start"
    else:
        label_x = -label_offset
        text_anchor = "end"

    # Store coordinates for all leaves
    leaf_coords = {}
    leaf_index = [0]  # Counter for leaf ordering

    # Calculate positions without drawing
    assign_tanglegram_tree_positions(
        root,
        leaf_coords,
        width,
        main_width,
        horizontal_spacing,
        vertical_spacing,
        label_offset,
        is_left,
        text_anchor,
        leaf_index,
    )

    return leaf_coords


###############################################################################
# Tanglegram Visualization
###############################################################################


def plot_tanglegram(
    left_tree: Node,
    right_tree: Node,
    width: int = 800,
    height: int = 400,
    margin: int = 30,
) -> str:
    """
    Create a tanglegram visualization with proper leaf connections.

    Args:
        left_tree: Left tree root node
        right_tree: Right tree root node
        width: SVG width
        height: SVG height
        margin: Margin around visualization

    Returns:
        SVG as a string
    """
    # Create base SVG
    svg_root = get_svg_root(width, height)

    # Calculate dimensions
    usable_width = width - 2 * margin
    tree_width = usable_width // 3  # Each tree gets 1/3, middle 1/3 for connections
    usable_height = height - 2 * margin

    # Calculate all coordinates for both trees and connections
    coords = calculate_tanglegram_coordinates(
        left_tree, right_tree, tree_width, usable_height
    )

    # Create container group
    container = ET.SubElement(
        svg_root, "g", {"transform": f"translate({margin}, {margin})"}
    )

    # Create left tree group with darker stroke
    left_group = ET.SubElement(
        container,
        "g",
        {
            "stroke": "#aaaaaa",
            "stroke-width": "2",
            "fill": "none",
        },  # Increased visibility
    )

    # Create right tree group with darker stroke
    right_group = ET.SubElement(
        container,
        "g",
        {
            "transform": f"translate({2 * tree_width}, 0)",  # Position at 2/3 of width
            "stroke": "#aaaaaa",
            "stroke-width": "2",
            "fill": "none",  # Increased visibility
        },
    )

    # Draw left tree
    draw_tree_from_coordinates(left_group, coords["left_coords"])

    # Draw right tree
    draw_tree_from_coordinates(right_group, coords["right_coords"])

    # Create label connections group with stronger visibility
    label_connections = ET.SubElement(
        container,
        "g",
        {
            "class": "label-connections",
            "stroke": "#999999",  # Darker color
            "stroke-width": "1.5",  # Thicker lines
            "stroke-opacity": "0.6",  # Higher opacity
            "fill": "none",
        },
    )

    # Draw connections between matching leaves
    for conn in coords["connections"]:
        # Draw curved connector between labels
        ET.SubElement(
            label_connections,
            "path",
            {
                "d": (
                    f"M {conn['left_x']},{conn['left_y']} "
                    f"C {conn['mid_x']},{conn['left_y'] - conn['curve_height']} "
                    f"{conn['mid_x']},{conn['left_y'] + conn['curve_height']} "
                    f"{conn['right_x']},{conn['right_y']}"
                ),
                "stroke": "#999999",  # Darker color
                "stroke-width": "1.5",  # Thicker lines
                "stroke-opacity": "0.7",  # Higher opacity
                "stroke-dasharray": "4,4",
                "fill": "none",
            },
        )

    return ET.tostring(svg_root, encoding="utf8").decode("utf8")


def draw_tree_from_coordinates(
    parent: ET.Element, coords: Dict[str, LeafCoordinates]
) -> None:
    """
    Draw a tree using pre-calculated coordinates.

    Args:
        parent: Parent SVG element to add tree to
        coords: Dictionary of node coordinates
    """
    for node_id, node_data in coords.items():
        # Skip internal nodes that don't have the is_internal flag
        if "is_internal" in node_data and node_data["is_internal"]:
            # Draw lines to children for internal nodes
            for line in node_data["children"]:
                # Horizontal line
                horiz = line["horiz"]
                ET.SubElement(
                    parent,
                    "path",
                    {"d": f"M {horiz['x1']},{horiz['y1']} H {horiz['x2']}"},
                )

                # Vertical line
                vert = line["vert"]
                ET.SubElement(
                    parent, "path", {"d": f"M {vert['x1']},{vert['y1']} V {vert['y2']}"}
                )
        elif "node" in node_data and "branch_end" in node_data:  # Leaf node
            x, y = node_data["node"]
            branch_end_x, _ = node_data["branch_end"]
            label_x, _ = node_data["label"]
            connector_end_x = node_data.get(
                "connector_end",
                (
                    label_x - 10
                    if "text_anchor" in node_data
                    and node_data["text_anchor"] == "start"
                    else label_x + 10
                ),
            )[0]

            # Draw branch extension (solid line)
            ET.SubElement(
                parent, "path", {"d": f"M {x},{y} H {branch_end_x}", "fill": "none"}
            )

            # Draw dashed connector to label
            ET.SubElement(
                parent,
                "path",
                {
                    "d": f"M {branch_end_x},{y} H {connector_end_x}",
                    "fill": "none",
                    "stroke-dasharray": "4,4",
                },
            )

            # Add label (only if we have a text_anchor value)
            if "text_anchor" in node_data:
                ET.SubElement(
                    parent,
                    "text",
                    {
                        "x": str(label_x),
                        "y": str(y),
                        "text-anchor": node_data["text_anchor"],
                        "font-family": DEFAULT_FONT_FAMILY,
                        "font-size": DEFAULT_FONT_SIZE,
                        "fill": DEFAULT_RECT_STROKE_COLOR,
                        "dominant-baseline": "middle",
                    },
                ).text = node_id