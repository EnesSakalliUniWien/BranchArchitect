import math
import xml.etree.ElementTree as ET
from typing import List, Generator, Dict


#####################################################################
# 1) We assume you have the exact Node class from your snippet:
#    (Shown here in condensed form for completeness.)
#####################################################################
class ReorderStrategy:
    AVERAGE = "average"
    MAXIMUM = "maximum"
    MINIMUM = "minimum"
    MEDIAN = "median"


# -- your Node class, truncated to essential parts for reference --
# (Use the full definition from your snippet.)
class Node:
    def __init__(
        self,
        children=None,
        name=None,
        length=None,
        values=None,
        split_indices=(),
        leaf_name=None,
        _visual_order_indices=None,
        _order=None,
    ):
        self.children = children if children is not None else []
        self.name = name
        self.length = length
        self.values = values if values is not None else {}
        self.split_indices = split_indices
        self.leaf_name = leaf_name
        self._visual_order_indices = _visual_order_indices
        self._order = _order if _order is not None else []
        self._cached_splits = None
        self._split_index = None

    def append_child(self, node: "Node") -> None:
        self.children.append(node)

    def is_internal(self) -> bool:
        return bool(self.children)

    def traverse(self) -> Generator["Node", None, None]:
        yield self
        for child in self.children:
            yield from child.traverse()

    def get_leaves(self) -> List["Node"]:
        if not self.children:
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaves())
        return leaves

    def __repr__(self) -> str:
        return f"Node('{self.name}')"


#####################################################################
# 2) Utility: "is_leaf" check and "safe" accessors
#####################################################################
def is_leaf(node: Node) -> bool:
    return len(node.children) == 0


def get_node_length(node: Node) -> float:
    """
    Return node.length if not None, else default to 1.0 for visualization.
    """
    return node.length if (node.length is not None) else 1.0


def get_node_name(node: Node) -> str:
    """
    Return node.name if not None, else 'Unnamed'.
    """
    return node.name if (node.name is not None) else "Unnamed"


#####################################################################
# 3) Basic traversal to list all nodes
#####################################################################
def traverse(root: Node) -> Generator[Node, None, None]:
    yield root
    for child in root.children:
        yield from traverse(child)


#####################################################################
# 4) Create an SVG <svg> element
#####################################################################
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


#####################################################################
# 5) Add a <path> element to parent
#####################################################################
def add_svg_path(parent: ET.Element, attrs: Dict[str, str]) -> None:
    ET.SubElement(parent, "path", attrs)


#####################################################################
# 6) Compute the leaf order of a tree (for labeling angles)
#####################################################################
def get_order(root: Node) -> List[str]:
    """
    Return the list of leaf names in the order encountered.
    """
    leaves = []
    for node in traverse(root):
        if is_leaf(node):
            leaves.append(get_node_name(node))
    return leaves


#####################################################################
# 7) We'll store the "visual" radius/angle in a small class
#####################################################################
class VisualNode:
    def __init__(self, radius: float, angle: float, children=None):
        self.radius = radius
        self.angle = angle
        self.children: List["VisualNode"] = children if children else []


#####################################################################
# 8) Convert your Node tree -> VisualNode tree (with radius & angle)
#####################################################################
def tree_to_visual_nodes(
    root: Node,
    total_angle: float,
    order: List[str],
    parent_radius: float,
    ignore_branch_lengths: bool = False,
) -> VisualNode:
    """
    :param root: The Node from your data
    :param total_angle: usually 2*pi
    :param order: list of leaf names for angle calculations
    :param parent_radius: the parent's radius
    :param ignore_branch_lengths: if True, treat each branch as length=1
    :return: a VisualNode with radius/angle, plus children
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


#####################################################################
# 9) Build link <path> data from parent to child
#####################################################################
STROKE_WIDTH = 1


def build_link_path(source: VisualNode, target: VisualNode) -> str:
    sx = source.radius * math.cos(source.angle)
    sy = source.radius * math.sin(source.angle)
    tx = target.radius * math.cos(target.angle)
    ty = target.radius * math.sin(target.angle)

    # We'll do a short arc from the parent's angle to target's angle at parent's radius
    cx = source.radius * math.cos(target.angle)
    cy = source.radius * math.sin(target.angle)

    arc_flag = 1 if abs(target.angle - source.angle) > math.pi else 0
    sweep_flag = 1 if abs(source.angle) < abs(target.angle) else 0

    path_d = f"M {sx},{sy} A {source.radius},{source.radius} 0 {arc_flag} {sweep_flag} {cx},{cy} L {tx},{ty}"
    return path_d


def generate_links(node: VisualNode) -> List[Dict[str, str]]:
    """
    Recursively build <path> 'd' attributes for links from `node` to each child.
    """
    all_paths = []
    for child in node.children:
        attrs = {
            "class": "links",
            "stroke-width": str(STROKE_WIDTH),
            "fill": "none",
            "style": "stroke-opacity:1;stroke:#000",
            "d": build_link_path(node, child),
        }
        all_paths.append(attrs)
        all_paths.extend(generate_links(child))
    return all_paths


#####################################################################
# 10) Add labels around the outer radius
#####################################################################
def add_labels(parent: ET.Element, order: List[str], radius: float) -> None:
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
                "font-size": "12",
                "font-weight": "normal",
                "font-family": "Courier New",
                "style": "fill:#000",
                "dominant-baseline": "middle",
            },
        )
        text_el.text = name


#####################################################################
# 11) Generate SVG for two trees side by side
#####################################################################
def generate_svg_two_trees(
    root1: Node,
    root2: Node,
    size: int = 400,
    margin: int = 30,
    label_offset: int = 2,
    ignore_branch_lengths: bool = False,
) -> str:
    """
    Create an SVG with exactly two trees, each in a box of width=size, total width=2*size.
    """
    total_width = 2 * size
    total_height = size
    svg_root = get_svg_root(total_width, total_height)

    # group1 centered in the left box
    group1 = ET.SubElement(
        svg_root, "g", {"transform": f"translate({size/2}, {size/2})"}
    )

    # group2 centered in the right box
    group2 = ET.SubElement(
        svg_root, "g", {"transform": f"translate({size * 1.5}, {size/2})"}
    )

    # Process Tree 1
    order1 = get_order(root1)
    vn1 = tree_to_visual_nodes(root1, 2 * math.pi, order1, 0, ignore_branch_lengths)
    max_r1 = max(vn.radius for vn in traverse(vn1))
    usable1 = size / 2 - margin - label_offset
    scale1 = usable1 / max_r1 if max_r1 != 0 else 1.0
    for n1 in traverse(vn1):
        n1.radius *= scale1
    links1 = generate_links(vn1)
    for attrs in links1:
        add_svg_path(group1, attrs)
    # labels
    add_labels(group1, order1, (size / 2 - margin))

    # Process Tree 2
    order2 = get_order(root2)
    vn2 = tree_to_visual_nodes(root2, 2 * math.pi, order2, 0, ignore_branch_lengths)
    max_r2 = max(vn.radius for vn in traverse(vn2))
    usable2 = size / 2 - margin - label_offset
    scale2 = usable2 / max_r2 if max_r2 != 0 else 1.0
    for n2 in traverse(vn2):
        n2.radius *= scale2
    links2 = generate_links(vn2)
    for attrs in links2:
        add_svg_path(group2, attrs)
    # labels
    add_labels(group2, order2, (size / 2 - margin))

    return ET.tostring(svg_root, encoding="utf8").decode("utf8")


#####################################################################
# 12) Generate SVG for N trees in a row
#####################################################################
def generate_svg_multiple_trees(
    roots: List[Node],
    size: int = 200,
    margin: int = 30,
    label_offset: int = 2,
    ignore_branch_lengths: bool = False,
) -> str:
    """
    Create an SVG with N trees laid out horizontally, each in a box of width=size.
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

        # 1) Determine leaf order
        order = get_order(root)
        # 2) Convert to VisualNodes
        vn = tree_to_visual_nodes(root, 2 * math.pi, order, 0, ignore_branch_lengths)
        # 3) Find max radius and scale
        max_r = max(vn.radius for vn in traverse(vn))
        usable_radius = size / 2 - margin - label_offset
        factor = usable_radius / max_r if max_r != 0 else 1.0
        for node_v in traverse(vn):
            node_v.radius *= factor

        # 4) Draw links
        links = generate_links(vn)
        for attrs in links:
            add_svg_path(group, attrs)

        # 5) Add labels
        add_labels(group, order, (size / 2 - margin))

    return ET.tostring(svg_root, encoding="utf8").decode("utf8")
