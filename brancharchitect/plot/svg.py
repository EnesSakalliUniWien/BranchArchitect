import math
import xml.etree.ElementTree as ET
from typing import List, Generator, Dict, Tuple, Optional, TypedDict

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


def add_labels(
    parent: ET.Element,
    order: List[str],
    radius: float,
    font_family: str = DEFAULT_FONT_FAMILY,
    font_size: str = DEFAULT_FONT_SIZE,
    nodes: Optional[List["VisualNode"]] = None
) -> None:
    """Add circular labels around the outer radius for radial tree layouts.
    If nodes is provided, use their angle and name for label placement.
    Otherwise, use order and uniform angle spacing.
    """
    if nodes is not None and len(nodes) > 0:
        for node in nodes:
            angle_deg = math.degrees(node.angle)
            anchor = "end" if 90 < angle_deg < 270 else "start"
            rotate_label = 180 if 90 < angle_deg < 270 else 0
            text_el = ET.SubElement(
                parent,
                "text",
                {
                    "text-anchor": anchor,
                    "transform": f"rotate({angle_deg}) translate({radius},0) rotate({rotate_label})",
                    "font-size": font_size,
                    "font-weight": "normal",
                    "font-family": font_family,
                    "style": f"fill:{DEFAULT_STROKE_COLOR}",
                    "dominant-baseline": "middle",
                },
            )
            text_el.text = node.name
    else:
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
                    "font-size": font_size,
                    "font-weight": "normal",
                    "font-family": font_family,
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
        self,
        radius: float,
        angle: float,
        name: str,
        is_leaf: bool,
        children: Optional[List["VisualNode"]] = None,
    ):
        if children is None:
            children = []
        self.radius = radius
        self.angle = angle
        self.name = name
        self.is_leaf = is_leaf
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
