from brancharchitect.tree import Node
from typing import (
    TypeVar,
    NewType,
)
from typing import Any, Set, List, FrozenSet, Dict, Collection

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

        self._all_sedges: list[Node] = all_sedges
        self._edge_types: dict[NodeName, EdgeType] = edge_types
        self._ancestor_edges: dict[ComponentSet] = ancestor_edges
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
        components.append(NodeName(node.split_indices))
        return tuple(components)

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
    elif all(child.length == 0 for child in node.children) and node.length > 0:
        return "full"

    # Check for 'partial' edge type: at least one child has zero length and node has positive length
    elif any(child.length == 0 for child in node.children) and node.length > 0:
        return "partial"

    # Check for 'anti' edge type: node has zero length but all children have positive length
    elif all(child.length > 0 for child in node.children) and node.length == 0:
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
    ancestor_edges: dict[list[str], Node] = {}
    arms: dict[list[str], tuple[ComponentSet]] = {}

    # Determine the edge type of the current node
    type_: EdgeType = get_type(node)

    edge_types[node.split_indices] = type_  # Store the edge type of the current node

    # If the current node is a 'full' or 'partial' s-edge, add it to the list of sedges
    if type_ in ["full", "partial"]:
        all_sedges.append(node)

    # Calculate the component sets (arms) for each child of the current node
    edge_arms = tuple([calculate_component_set_tree(child) for child in node.children])
    arms[node.split_indices] = edge_arms  # Store the arms of the current node

    # Assign the current node as the ancestor for each of its children
    for child in node.children:
        ancestor_edges[child.split_indices] = node

    # Assign the current node as the ancestor for each component set in the arms
    for arm in edge_arms:
        ancestor_edges[arm] = node

    # Create a FunctionalTree instance for the current node
    t1 = FunctionalTree(set(all_sedges), edge_types, ancestor_edges, arms)

    # Recursively traverse each child and combine their FunctionalTrees with the current one
    for child in node.children:
        child_tree = build_functional_tree(child)
        t1 = t1 + child_tree  # Combine the FunctionalTrees

    # Return the combined FunctionalTree
    return t1


def get_children_splits(node: "Node") -> Set[Any]:
    """
    Compute the set of child splits for a node n.
    
    Definition:
      If n has children { c₁, c₂, …, cₖ }, then
          D(n) = { s(c) : s(c) = c.split_indices for each c ∈ children(n) }.
    """
    return {child.split_indices for child in node.children}


def get_components_for_tree(tree: "Node", child_splits: Set[Any], common: Set[Any]) -> List[FrozenSet[Any]]:
    """
    For each s ∈ D (child splits), locate the corresponding child node nₛ and compute its component.
    
    Definition:
      For a child node nₛ found via s, define its component as:
          C(nₛ) = nₛ.to_splits() ∩ common.
    
    Returns the collection { C(nₛ) } as a list of frozensets.
    """
    components = []
    for s in child_splits:
        node = tree.find_node_by_split(s)
        if node:
            comp = node.to_splits() & common
            components.append(frozenset(comp))
    return components


def get_maximal_tuples(splits: Collection[Any]) -> Set[Any]:
    """
    Compute the maximal elements of a collection S of splits with respect to set inclusion.
    
    Definition:
      For s ∈ S, s is maximal if there exists no o ∈ S (with o ≠ s) such that s ⊂ o.
      That is, maximal(S) = { s ∈ S : ¬∃ o ∈ S, o ≠ s, with s ⊂ o }.
    """
    return {s for s in splits if not any(set(s) < set(o) for o in splits if s != o)}


def compute_maximal_components(components: List[FrozenSet[Any]]) -> Set[FrozenSet[Any]]:
    """
    For each component set comp (a frozenset of splits), compute its maximal subset.
    
    Definition:
      For comp ⊆ P(S), define
          maximal(comp) = { s ∈ comp : there is no o ∈ comp with s ⊂ o }.
    
    Returns the set of all maximal component sets.
    """
    maximal_components = set()
    for comp in components:
        maximal = get_maximal_tuples(comp)
        if maximal:
            maximal_components.add(frozenset(maximal))
    return maximal_components


def get_edge_types(D: Set[Any], common: Set[Any], node: "Node") -> str:
    """
    Classify the s-edge at node n based on its descendant splits D and the common set C.
    
    Let A = D ∩ common.
    Then, the classification is defined as:
      - "full"    if A = ∅ (i.e. no child split is common),
      - "none"    if A = D (i.e. every child split is common; equivalently, |A| = |children(n)|),
      - "partial" if ∅ ⊂ A ⊂ D (i.e. only some child splits are common).
    """
    A = D & common
    if len(A) == 0:
        return "full"
    if len(A) == len(node.children):
        return "none"
    else: 
        return "partial"

def get_splits_info(tree1: "Node", tree2: "Node") -> Dict[Any, Dict[str, Any]]:
    """
    Compute detailed split information for two trees T₁ and T₂ on a set‐theoretical basis.
    
    Definitions:
      - S(T₁) = { n.split_indices : n ∈ T₁.to_splits(with_leaves=True) } and
        S(T₂) = { n.split_indices : n ∈ T₂.to_splits(with_leaves=True) }.
      - C = S(T₁) ∩ S(T₂) (the set of common splits).
    
    For each common split s ∈ C:
      1. Identify nodes n₁ ∈ T₁ and n₂ ∈ T₂ such that:
             s = n₁.split_indices = n₂.split_indices.
      2. For each node nᵢ, define its child split set:
             D(nᵢ) = { c.split_indices : c ∈ children(nᵢ) }.
      3. Compute the commonality ratio:
             r = |D(nᵢ) ∩ C| / |D(nᵢ)|
         for each node, quantifying the degree of shared child splits.
      4. If not all child splits are common (i.e., |D(nᵢ) ∩ C| < |children(nᵢ)| for either n₁ or n₂),
         then for each tree:
           a. For each s ∈ D(nᵢ), compute the component:
                  C(nₛ) = nₛ.to_splits() ∩ C.
           b. Compute the maximal components:
                  arms_tᵢ = compute_maximal_components({ C(nₛ) }).
      5. Classify the s-edge for each node using get_edge_types(D(nᵢ), C, nᵢ).
     
    Returns a dictionary mapping each common split s ∈ C to:
      {
         'edge_type_one': classification for n₁,
         'edge_type_two': classification for n₂,
    }
    """
    S1 = tree1.to_indexed_split_set(with_leaves=True)
    S2 = tree2.to_indexed_split_set(with_leaves=True)

    C = S1 & S2  # C = S(T₁) ∩ S(T₂)

    s_edges = {}

    for s in C:
        n1 = tree1.find_node_by_split(s)
        n2 = tree2.find_node_by_split(s)    
        if n1 and n2:
            D1 = get_children_splits(n1)
            D2 = get_children_splits(n2)
            # Process further if at least one node has a nontrivial mixture of child splits.
            if len(D1 & C) < len(n1.children) or len(D2 & C) < len(n2.children):
                print(D1 & C)   
        
                arms_t1, arms_t2 = None, None
                comps1 = get_components_for_tree(tree1, D1, C)
                comps2 = get_components_for_tree(tree2, D2, C)
                arms_t1 = compute_maximal_components(comps1)
                arms_t2 = compute_maximal_components(comps2)
                edge_type_t1 = get_edge_types(D1, C, n1)
                edge_type_t2 = get_edge_types(D2, C, n2)

                s_edges[s] = {
                    'edge_type_one': edge_type_t1,
                    'edge_type_two': edge_type_t2,
                    'arms_t_one': arms_t1,
                    'arms_t_two': arms_t2,
                }
    return s_edges
