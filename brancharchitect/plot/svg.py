import math
import xml.etree.ElementTree as ET
from typing import List, Generator, Dict, Tuple, Union
from brancharchitect.tree import Node  # Cleaned: use Node from tree.py

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
def is_leaf(node: Node) -> bool:
    """Check if a node is a leaf (has no children)."""
    return len(node.children) == 0


def get_node_length(node: Node) -> float:
    """Return node.length if not None, else default to 1.0 for visualization."""
    return node.length if (node.length is not None) else DEFAULT_NODE_LENGTH


def get_node_name(node: Node) -> str:
    """
    Return node.leaf_name if available for leaf nodes, otherwise node.name,
    or 'Unnamed' as fallback. This ensures consistent identification across trees.
    """
    return node.name if (node.name is not None) else DEFAULT_NODE_NAME


def get_consistent_node_id(node: Node) -> str:
    """
    Get a consistent identifier for a node that can be matched across trees.
    For leaf nodes, use just the name to ensure proper matching in tanglgrams.
    """
    if is_leaf(node):
        return node.name

    # For non-leaf nodes, use a placeholder since they won't be connected
    return f"internal-{node.name}" if node.name else "internal"


def traverse(root: Node) -> Generator[Node, None, None]:
    """Traverse a tree in pre-order, yielding each node."""
    yield root
    for child in root.children:
        yield from traverse(child)


def get_order(root: Node) -> List[str]:
    """
    Return the list of leaf names in the order encountered,
    using consistent node identification.
    """
    leaves = []
    for node in traverse(root):
        if is_leaf(node):
            leaves.append(get_consistent_node_id(node))
    return leaves


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
        self, radius: float, angle: float, children: List["VisualNode"] = None
    ):
        """
        Initialize visual node with position in polar coordinates.

        Args:
            radius: Distance from center
            angle: Angle in radians
            children: List of child visual nodes
        """
        self.radius = radius
        self.angle = angle
        self.children: List["VisualNode"] = children if children else []

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


def tree_to_visual_nodes(
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
        idx = order.index(get_node_name(root))
        leaf_angle = (total_angle / len(order)) * idx if order else 0.0
        return VisualNode(node_radius, leaf_angle, [])
    else:
        # Internal node: build children first
        child_visuals = [
            tree_to_visual_nodes(
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
# Radial Layout Link Generation
###############################################################################
def build_link_path(source: VisualNode, target: VisualNode) -> str:
    """
    Build an SVG path string connecting two visual nodes in a radial layout.

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


def generate_links(node: VisualNode) -> List[Dict[str, str]]:
    """
    Recursively build SVG path attributes for links from node to each child.

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
            "d": build_link_path(node, child),
        }
        all_paths.append(attrs)
        all_paths.extend(generate_links(child))
    return all_paths


###############################################################################
# Tree Visualization Generators
###############################################################################
def generate_svg_two_trees(
    root1: Node,
    root2: Node,
    size: int = 400,
    margin: int = 30,
    label_offset: int = 2,
    ignore_branch_lengths: bool = False,
) -> str:
    """
    Create an SVG with exactly two trees side by side.

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
        svg_root, "g", {"transform": f"translate({size/2}, {size/2})"}
    )

    # Group 2 centered in the right box
    group2 = ET.SubElement(
        svg_root, "g", {"transform": f"translate({size * 1.5}, {size/2})"}
    )

    # Process Tree 1
    order1 = get_order(root1)
    vn1 = tree_to_visual_nodes(root1, 2 * math.pi, order1, 0, ignore_branch_lengths)
    max_r1 = max(vn.radius for vn in vn1.traverse())
    usable1 = size / 2 - margin - label_offset
    scale1 = usable1 / max_r1 if max_r1 != 0 else 1.0

    # Scale tree 1
    for n1 in vn1.traverse():
        n1.scale_radius(scale1)

    # Add links for tree 1
    links1 = generate_links(vn1)
    for attrs in links1:
        add_svg_path(group1, attrs)

    # Add labels for tree 1
    add_labels(group1, order1, (size / 2 - margin))

    # Process Tree 2 (same pattern)
    order2 = get_order(root2)
    vn2 = tree_to_visual_nodes(root2, 2 * math.pi, order2, 0, ignore_branch_lengths)
    max_r2 = max(vn.radius for vn in vn2.traverse())
    usable2 = size / 2 - margin - label_offset
    scale2 = usable2 / max_r2 if max_r2 != 0 else 1.0

    # Scale tree 2
    for n2 in vn2.traverse():
        n2.scale_radius(scale2)

    # Add links for tree 2
    links2 = generate_links(vn2)
    for attrs in links2:
        add_svg_path(group2, attrs)

    # Add labels for tree 2
    add_labels(group2, order2, (size / 2 - margin))

    return ET.tostring(svg_root, encoding="utf8").decode("utf8")


def generate_svg_multiple_trees(
    roots: List[Node],
    size: int = 200,
    margin: int = 30,
    label_offset: int = 2,
    ignore_branch_lengths: bool = False,
) -> str:
    """
    Create an SVG with N trees laid out horizontally.

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

        # Process this tree
        order = get_order(root)
        vn = tree_to_visual_nodes(root, 2 * math.pi, order, 0, ignore_branch_lengths)

        # Scale to fit
        max_r = max(vn.radius for vn in vn.traverse())
        usable_radius = size / 2 - margin - label_offset
        factor = usable_radius / max_r if max_r != 0 else 1.0

        for node_v in vn.traverse():
            node_v.scale_radius(factor)

        # Add links
        links = generate_links(vn)
        for attrs in links:
            add_svg_path(group, attrs)

        # Add labels
        add_labels(group, order, (size / 2 - margin))

    return ET.tostring(svg_root, encoding="utf8").decode("utf8")


###############################################################################
# Rectangular Layout Tree Visualization
###############################################################################
def generate_rectangular_tree(root: Node, width: int = 400, height: int = 180) -> str:
    """
    Generate a classical rectangular layout tree with vertical orientation.

    Args:
        root: Tree root node
        width: SVG width
        height: SVG height

    Returns:
        SVG as a string
    """
    svg_root = get_svg_root(width, height)

    # Get all leaves for horizontal spacing
    leaves = [n for n in traverse(root) if not n.children]
    leaf_count = len(leaves)

    # Calculate spacings
    horizontal_spacing = width / (leaf_count + 1)
    max_depth = max(len(list(n.traverse())) for n in root.traverse())
    vertical_spacing = (height - 60) / (
        max_depth + 1
    )  # Adjusted to leave space for labels
    label_y = height - 40  # Position for labels at the bottom

    # Create main group with margin
    group = ET.SubElement(
        svg_root,
        "g",
        {
            "transform": "translate(0, 20)",
            "stroke": DEFAULT_RECT_STROKE_COLOR,
            "stroke-width": "1.5",
        },
    )

    def assign_positions(node, depth=0, leaf_index=[0]):
        """Assign x,y coordinates to each node."""
        y = depth * vertical_spacing

        if not node.children:
            # Leaf node
            x = (leaf_index[0] + 1) * horizontal_spacing
            leaf_index[0] += 1
            # Draw branch extension and dashed connector
            branch_end_y = (depth + 1) * vertical_spacing

            # Draw solid line extension
            ET.SubElement(
                group,
                "path",
                {
                    "d": f"M {x},{y} V {branch_end_y}",
                    "fill": "none",
                },
            )

            # Draw dashed connector to label
            ET.SubElement(
                group,
                "path",
                {
                    "d": f"M {x},{branch_end_y} V {label_y - 10}",
                    "fill": "none",
                    "stroke-dasharray": "4,4",
                },
            )

            # Add label
            ET.SubElement(
                group,
                "text",
                {
                    "x": str(x),
                    "y": str(label_y),
                    "font-family": "'Share Tech Mono', monospace",
                    "font-size": "12",
                    "fill": DEFAULT_RECT_STROKE_COLOR,
                    "text-anchor": "middle",  # Center text under the branch
                    "dominant-baseline": "middle",
                },
            ).text = get_node_name(node)

            return (x, y)

        # Internal node - process children first
        child_positions = []
        for child in node.children:
            pos = assign_positions(child, depth + 1, leaf_index)
            child_positions.append(pos)

        # Position internal node at average of children's x-coordinates
        x = sum(pos[0] for pos in child_positions) / len(child_positions)

        # Draw lines to children
        for child_x, child_y in child_positions:
            # Vertical line
            ET.SubElement(
                group, "path", {"d": f"M {x},{y} V {child_y}", "fill": "none"}
            )
            # Horizontal line
            ET.SubElement(
                group, "path", {"d": f"M {x},{child_y} H {child_x}", "fill": "none"}
            )

        return (x, y)

    # Process the tree and get root position
    assign_positions(root)

    return ET.tostring(svg_root, encoding="utf8").decode("utf8")


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

    # Get all leaves to calculate vertical spacing
    left_leaves = [n for n in traverse(left_tree) if not n.children]
    right_leaves = [n for n in traverse(right_tree) if not n.children]
    total_leaves = max(len(left_leaves), len(right_leaves))

    # Calculate spacings
    vertical_spacing = usable_height / (total_leaves + 1)

    # Create container group
    container = ET.SubElement(
        svg_root, "g", {"transform": f"translate({margin}, {margin})"}
    )

    # Draw left tree
    left_group = ET.SubElement(
        container,
        "g",
        {"stroke": DEFAULT_RECT_STROKE_COLOR, "stroke-width": "1.5", "fill": "none"},
    )

    # Draw right tree
    right_group = ET.SubElement(
        container,
        "g",
        {
            "transform": f"translate({2 * tree_width}, 0)",  # Position at 2/3 of width
            "stroke": DEFAULT_RECT_STROKE_COLOR,
            "stroke-width": "1.5",
            "fill": "none",
        },
    )

    # Get leaf positions from drawing trees
    left_coords = draw_tree_side(
        left_tree, left_group, tree_width, vertical_spacing, usable_height, is_left=True
    )

    right_coords = draw_tree_side(
        right_tree,
        right_group,
        tree_width,
        vertical_spacing,
        usable_height,
        is_left=False,
    )

    # Create label connections group
    label_connections = ET.SubElement(
        container,
        "g",
        {
            "class": "label-connections",
            "stroke": DEFAULT_RECT_STROKE_COLOR,
            "stroke-width": "1",
            "stroke-opacity": "0.3",
            "fill": "none",
        },
    )

    # Connect matching leaves
    left_names = {
        node_id: coords
        for node_id, coords in left_coords.items()
        if node_id in right_coords
    }

    # Draw connections between matching leaves
    for name in left_names:
        left_y = left_coords[name]["y"]
        right_y = right_coords[name]["y"]

        # Draw label connection (bezier curve)
        left_label_x = left_coords[name]["label"][0]
        right_label_x = 2 * tree_width + right_coords[name]["label"][0]

        # Calculate bezier curve control points
        total_width = right_label_x - left_label_x
        mid_x = left_label_x + total_width / 2
        curve_height = vertical_spacing * 0.5

        # Draw curved connector between labels
        ET.SubElement(
            label_connections,
            "path",
            {
                "d": (
                    f"M {left_label_x},{left_y} "
                    f"C {mid_x},{left_y-curve_height} "
                    f"{mid_x},{left_y+curve_height} "
                    f"{right_label_x},{right_y}"
                ),
                "stroke": DEFAULT_RECT_STROKE_COLOR,
                "stroke-width": "1",
                "stroke-opacity": "0.5",
                "stroke-dasharray": "4,4",
                "fill": "none",
            },
        )

    return ET.tostring(svg_root, encoding="utf8").decode("utf8")


def draw_tree_side(
    root: Node,
    parent: ET.Element,
    width: float,
    vertical_spacing: float,
    height: float,
    is_left: bool = True,
) -> Dict[str, Dict[str, Union[Tuple[float, float], float]]]:
    """
    Draw one side of the tanglegram in rectangular style.

    Args:
        root: Tree root node
        parent: Parent SVG element to add tree to
        width: Available width for this tree
        vertical_spacing: Vertical distance between nodes
        height: Total available height
        is_left: True if this is the left tree, False for right tree

    Returns:
        Dictionary mapping node IDs to coordinates
    """
    # Calculate tree depth and structure
    max_depth = 0

    def get_depth(node, depth=0):
        nonlocal max_depth
        max_depth = max(max_depth, depth)
        for child in node.children:
            get_depth(child, depth + 1)

    get_depth(root)

    # Calculate spacings
    main_width = width * 0.8  # Leave space for labels
    horizontal_spacing = main_width / (max_depth + 1)
    label_offset = width * 0.15

    # Label position
    if is_left:
        label_x = main_width + label_offset
        text_anchor = "start"
    else:
        label_x = -label_offset
        text_anchor = "end"

    # Track node positions and leaf coordinates
    leaf_coords = {}
    leaf_index = [0]  # Counter for leaf ordering

    def assign_positions(node, depth=0):
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

            # Use the same consistent identification function here
            node_id = get_consistent_node_id(node)

            # Store coordinates for connections with consistent identifier
            leaf_coords[node_id] = {
                "node": (x, y),
                "branch_end": (branch_end_x, y),
                "label": (label_x_pos, y),
                "y": y,  # Store y position for easier access
            }

            # Draw branch extension (solid line)
            ET.SubElement(
                parent, "path", {"d": f"M {x},{y} H {branch_end_x}", "fill": "none"}
            )

            # Draw dashed connector to label (same style for both trees)
            ET.SubElement(
                parent,
                "path",
                {
                    "d": f"M {branch_end_x},{y} H {connector_end_x}",
                    "fill": "none",
                    "stroke-dasharray": "4,4",
                },
            )

            # Add label
            ET.SubElement(
                parent,
                "text",
                {
                    "x": str(label_x_pos),
                    "y": str(y),
                    "text-anchor": text_anchor,
                    "font-family": "'Share Tech Mono', monospace",
                    "font-size": "12",
                    "fill": DEFAULT_RECT_STROKE_COLOR,
                    "dominant-baseline": "middle",
                },
            ).text = node_id

            return (x, y)

        # Process children first
        child_positions = []
        for child in node.children:
            pos = assign_positions(child, depth + 1)
            if pos:
                child_positions.append(pos)

        if not child_positions:
            return None

        # Calculate y as average of children
        y = sum(p[1] for p in child_positions) / len(child_positions)

        # Draw lines to children
        for child_x, child_y in child_positions:
            # Horizontal line
            ET.SubElement(parent, "path", {"d": f"M {x},{y} H {child_x}"})
            # Vertical line
            ET.SubElement(parent, "path", {"d": f"M {child_x},{y} V {child_y}"})

        return (x, y)

    # Draw the tree
    assign_positions(root)
    return leaf_coords
