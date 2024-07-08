import math
from dataclasses import dataclass, field
import xml.etree.ElementTree as ET

STROKE_WIDTH = 1

@dataclass(slots=True)
class VisualNode:
    radius: float
    angle: float
    children: list["VisualNode"] = field(default_factory=list)


def traverse(root):
    yield root
    for child in root.children:
        yield from traverse(child)


def get_svg_root(size):
    data = {
            'viewBox': f'{-size/2} {-size/2} {size} {size}',
            'version': '1.1',
            'xmlns': "http://www.w3.org/2000/svg",
            'xmlns:xlink': "http://www.w3.org/1999/xlink",
            'xml:space': 'preserve',
           }

    root = ET.Element('svg', data)
    return root


def add_svg_path(parent, data):
    return ET.SubElement(parent, 'path', data)

def get_order(root):
    order = []
    for node in traverse(root):
        if len(node.children) == 0:
            order.append(node.name)
    return order

def generate_svg(root, size=100, margin=10, label_offset=2, ignore_branch_lengths=False):
    order = get_order(root)
    vn = tree_to_visual_nodes(root, 2*math.pi, order, 0, ignore_branch_lengths=ignore_branch_lengths)

    max_radius = max(node.radius for node in traverse(vn))
    factor = (size/2-margin-label_offset) / max_radius
    for node in traverse(vn):
        node.radius = node.radius * factor

    links = generate_links(vn)

    svg_root = get_svg_root(size)
    for link in links:
        add_svg_path(svg_root, link)

    add_labels(svg_root, order, (size/2-margin))
    s = ET.tostring(svg_root, encoding='utf8')
    return s


def tree_to_visual_nodes(root, angle, order, radius, ignore_branch_lengths=False):
    if ignore_branch_lengths:
        node_radius = radius + 1
    else:
        node_radius = radius + root.length

    if len(root.children) == 0:
        children = [] 
        node_angle = (angle / len(order)) * order.index(root.name)
    else:
        children = [tree_to_visual_nodes(child, angle, order, node_radius, ignore_branch_lengths=ignore_branch_lengths)  for child in root.children]
        node_angle = sum(child.angle for child in children) / len(children)

    vn = VisualNode(node_radius, node_angle, children)
    return vn


def generate_links(node):
    datas = []
    for target in node.children:
        data = {
            'class': 'links',
            'stroke-width': str(STROKE_WIDTH),
            'fill': 'none',
            'd': build_link_path(node, target),
            'style': 'stroke-opacity:1;stroke:#000',
        }
        datas.append(data)
        datas.extend(generate_links(target))
    return datas

def build_link_path(source, target):
    source_x = source.radius * math.cos(source.angle)
    source_y = source.radius * math.sin(source.angle)
    target_x = target.radius * math.cos(target.angle)
    target_y = target.radius * math.sin(target.angle)

    curveX = source.radius * math.cos(target.angle)
    curveY = source.radius * math.sin(target.angle)

    arcFlag = 1 if abs(target.angle - source.angle) > math.pi else 0
    sweepFlag = 1 if abs(source.angle) < abs(target.angle) else 0
    return f"M {source_x}, {source_y} A {source.radius}, {source.radius} 0 {arcFlag} {sweepFlag} {curveX}, {curveY} L {target_x} {target_y}"



def add_labels(parent, order, radius):
    for i, name in enumerate(order):
        angle = 180*i*2 / len(order)
        data = {
                'text-anchor': 'end' if 90 < angle < 270 else 'start',
                'transform':  f'rotate({angle}) translate({radius}, 0) rotate({180 if 90 < angle < 270 else 0})',
                'font-size': '12',
                'font-weight': 'normal',
                'font-family': 'Courier New',
                'style': 'fill:#000',
                'dominant-baseline': "middle",
               }
        label = ET.SubElement(parent, 'text', data)
        label.text = name



