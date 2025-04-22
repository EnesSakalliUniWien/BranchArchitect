from brancharchitect.tree import Node
from brancharchitect.partition_set import PartitionSet, Partition
from typing import (
    TypeVar,
    NewType,
    Any,
)


Component = NewType("Component", Partition)
ComponentSet = tuple[Component, ...]  # type: ignore
EdgeType = str
X = TypeVar("X")
Y = TypeVar("Y")
NodeName = NewType("NodeName", Component)


class FunctionalTree:
    """
    Represents a tree structure used for analyzing topological changes in phylogenetic trees.
    """

    def __init__(
        self,
        all_sedges: set[Node],
        edge_types: dict[NodeName, EdgeType],
        ancestor_edges: dict[ComponentSet, Node],
        arms: dict[NodeName, list[ComponentSet]],
    ):

        self._all_sedges: set[Node] = all_sedges
        self._edge_types: dict[NodeName, EdgeType] = edge_types
        self._ancestor_edges : dict[ComponentSet, Node] = ancestor_edges
        self._arms: dict[NodeName, list[ComponentSet]] = arms

    def __add__(self, other: "FunctionalTree") -> "FunctionalTree":
        """
        Combines this FunctionalTree with another FunctionalTree, merging their attributes to create a new FunctionalTree.
        This method is used to merge analyses from different parts of a tree or from different trees,
        creating a comprehensive view of the combined topological features.
        Args:
            other (FunctionalTree): Another FunctionalTree instance to be combined with this one.
        Returns:
            FunctionalTree: A new FunctionalTree instance representing the combined attributes of both trees.

        TODO I don't understand exactly what this function does from the docstring. maybe an example would help.
        """

        all_sedges = set(list(self._all_sedges) + list(other._all_sedges))

        _edge_types = {}
        _edge_types.update(self._edge_types)
        _edge_types.update(other._edge_types)

        _ancestor_edges = {}
        _ancestor_edges.update(self._ancestor_edges)
        _ancestor_edges.update(other._ancestor_edges)

        arms = {}
        arms.update(self._arms)
        arms.update(other._arms)

        return FunctionalTree(all_sedges, _edge_types, _ancestor_edges, arms)


def calculate_component_set_tree(node: Node) -> ComponentSet:
    """
    Calculates the component set for a given node, used in the construction of a FunctionalTree.
    """
    components: list[Component] = []

    if node.length != 0:
        components.append(NodeName(Component(node.split_indices)))
        return tuple(components)

    if len(node.children) == 0:
        return (Component(node.split_indices),)

    for child in node.children:
        components += sorted(calculate_component_set_tree(child))

    return tuple(sorted(components))


def get_type(node: Node) -> EdgeType:
    """
    Determines the type of edge for a given node based on the lengths of its branches and children.

    Args:
        node (Node): The node for which the edge type is to be determined.

    Returns:
        EdgeType: The type of edge ('full', 'partial', 'anti', 'leaf', or 'none').

    The edge type helps in understanding the topological relationship of the node within the tree,
    particularly in the context of comparative analysis between trees.
    """
    # Check if the node is a leaf (no children)
    if len(node.children) == 0:
        return "leaf"

    # Check for 'full' edge type: node has a positive length and all children have zero length
    elif all((child.length or 0.0) == 0.0 for child in node.children) and (node.length or 0.0) > 0.0:
        return "full"

    # Check for 'partial' edge type: at least one child has zero length and node has positive length
    elif any((child.length or 0.0) == 0.0 for child in node.children) and (node.length or 0.0) > 0.0:
        return "partial"

    # Check for 'anti' edge type: node has zero length but all children have positive length
    elif all((child.length or 0.0) > 0.0 for child in node.children) and (node.length or 0.0) == 0.0:
        return "anti"

    else:
        return "none"


def build_functional_tree(node: Node) -> FunctionalTree:
    """
    Recursively constructs a FunctionalTree from a given tree node.
    The FunctionalTree is used for analyzing tree topology, focusing on edge types
    and identifying significant topological features (sedges).

    Args:
        node (Node): The root node of the tree to traverse.

    Returns:
        FunctionalTree: The FunctionalTree representing the analyzed structure of the tree.
    """

    all_sedges: list[Node] = []
    edge_types: dict[list[str], EdgeType] = {}
    ancestor_edges: dict[Any, Node] = {}
    arms: dict[list[str], list[ComponentSet]] = {}

    # Determine the edge type of the current node
    type_: EdgeType = get_type(node)

    edge_types[node.split_indices] = type_  # Store the edge type of the current node

    # If the current node is a 'full' or 'partial' s-edge, add it to the list of sedges
    if type_ in ["full", "partial"]:
        all_sedges.append(node)

    # Calculate the component sets (arms) for each child of the current node
    edge_arms = [calculate_component_set_tree(child) for child in node.children]
    arms[node.split_indices] = edge_arms  # Store the arms of the current node

    # Assign the current node as the ancestor for each of its children
    for child in node.children:
        ancestor_edges[child.split_indices] = node

    # Assign the current node as the ancestor for each component set in the arms
    for arm in edge_arms:
        ancestor_edges[arm] = node

    # Create a FunctionalTree instance for the current node
    t1 = FunctionalTree(all_sedges, edge_types, ancestor_edges, arms)

    # Recursively traverse each child and combine their FunctionalTrees with the current one
    for child in node.children:
        child_tree = build_functional_tree(child)
        t1 = t1 + child_tree  # Combine the FunctionalTrees
    # Return the combined FunctionalTree
    return t1