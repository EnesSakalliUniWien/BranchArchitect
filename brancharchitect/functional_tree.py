from typing import (
    TypeVar,
    NewType,
)
from node import Node

Component = NewType("Component", tuple[int])
ComponentSet = tuple[Component, ...]  # type: ignore
EdgeType = str
X = TypeVar("X")
Y = TypeVar("Y")
NodeName = NewType("NodeName", tuple[int])


class FunctionalTree:
    """
    Represents a tree structure used for analyzing topological changes in phylogenetic trees.
    """

    def __init__(
        self,
        all_sedges: list[Node],
        edge_types: dict[NodeName, EdgeType],
        ancestor_edges: dict[ComponentSet, Node],
        arms: dict[NodeName, list[ComponentSet]],
    ):
        # Nodes representing significant topological changes (sedges)
        self._all_sedges: list[Node] = all_sedges

        # Mapping of node names to their edge types for topological analysis
        self._edge_types: dict[NodeName, EdgeType] = edge_types

        # Mapping of component sets to their ancestor nodes
        self._ancestor_edges: dict[ComponentSet] = ancestor_edges

        # Mapping of node names to component sets (branching structures)
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
        """
        # Combine the lists of 'sedges' from both trees.
        # 'Sedges' are significant nodes indicating topological changes.
        all_sedges = self._all_sedges + other._all_sedges
        # Create a new dictionary for edge types and populate it with the edge types from both trees.
        # Edge types categorize nodes based on their structural roles in the tree.
        _edge_types = {}
        _edge_types.update(self._edge_types)
        # Include edge types from the current tree.
        _edge_types.update(other._edge_types)
        # Include edge types from the other tree, possibly overwriting duplicates.
        # Create a new dictionary for ancestor edges and populate it with the ancestor edges from both trees.
        # Ancestor edges map component sets to their ancestor nodes, showing hierarchical relationships.

        _ancestor_edges = {}
        # Include ancestor edges from the current tree.
        _ancestor_edges.update(self._ancestor_edges)
        # Include ancestor edges from the other tree.
        _ancestor_edges.update(other._ancestor_edges)
        # Create a new dictionary for arms and populate it with the arms from both trees.
        # Arms represent branching structures of nodes in the tree.
        arms = {}
        arms.update(self._arms)  # Include arms from the current tree.
        arms.update(other._arms)  # Include arms from the other tree.
        # Return a new FunctionalTree combining all the merged attributes.
        return FunctionalTree(all_sedges, _edge_types, _ancestor_edges, arms)


def calculate_component_set_tree(node: Node) -> ComponentSet:
    """
    Calculates the component set for a given node, used in the construction of a FunctionalTree.
    """
    components: list[Component] = []
    if node.length != 0:
        # Node has a non-zero length, indicating a significant topological feature
        components.append(NodeName(node.name))
        return tuple(components)

    if len(node.children) == 0:
        # Leaf node case
        return (Component(node.name),)

    # Recursively calculate component sets for child nodes
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
    if all(child.length == 0 for child in node.children) and node.length > 0:
        return "full"

    # Check for 'partial' edge type: at least one child has zero length and node has positive length
    elif any(child.length == 0 for child in node.children) and node.length > 0:
        return "partial"

    # Check for 'anti' edge type: node has zero length but all children have positive length
    elif all(child.length > 0 for child in node.children) and node.length == 0:
        return "anti"

    # If none of the above conditions are met, return 'none'
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

    # Initialize lists and dictionaries to store various properties of the tree
    all_sedges = []  # List to store significant nodes (sedges)
    edge_types = {}  # Dictionary to map node names to their edge types
    ancestor_edges = {}  # Dictionary to map component sets to their ancestor nodes
    arms = (
        {}
    )  # Dictionary to map node names to component sets (representing branching structures)

    # Determine the edge type of the current node
    type_ = get_type(node)
    edge_types[node.name] = type_  # Store the edge type of the current node

    # If the current node is a 'full' or 'partial' s-edge, add it to the list of sedges
    if type_ in ["full", "partial"]:
        all_sedges.append(node)

    # Calculate the component sets (arms) for each child of the current node
    edge_arms = [calculate_component_set_tree(child) for child in node.children]
    arms[node.name] = edge_arms  # Store the arms of the current node

    # Assign the current node as the ancestor for each of its children
    for child in node.children:
        ancestor_edges[child.name] = node

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
