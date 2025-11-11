from typing import Dict, List, FrozenSet

# Assuming these imports point to valid modules in your project structure
from brancharchitect.tree import Node
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.elements.partition import Partition


##################################################
#           Classification Logic
##################################################


def _ensure_split_indices(tree: Node) -> None:
    """Ensure split_indices are initialized for all nodes in the tree."""
    tree.initialize_split_indices(tree.taxa_encoding)


def _collect_subtree_splits(
    node: Node, subtree_splits_map: Dict[Node, PartitionSet[Partition]]
) -> PartitionSet[Partition]:
    """Recursively collect all splits in the subtree rooted at node."""
    subs: PartitionSet[Partition] = PartitionSet()
    for ch in node.children:
        subs |= _collect_subtree_splits(ch, subtree_splits_map)
    if node.children:
        subs.add(node.split_indices)
    subtree_splits_map[node] = subs
    return subs


def _classify_node(
    node: Node,
    splits: PartitionSet[Partition],
    s_ref: PartitionSet[Partition],
    s_common: PartitionSet[Partition],
) -> str:
    """Classify a node based on its split membership and subtree splits."""
    node_in_ref: bool = node.split_indices in s_ref
    fully_in_ref: bool = splits.issubset(s_common)
    fully_out_ref: bool = splits.isdisjoint(s_common)

    if node_in_ref and fully_in_ref:
        return "full-common"
    elif node_in_ref and not fully_in_ref:
        return "common-changed"
    elif (not node_in_ref) and fully_out_ref:
        return "full-unique"
    else:
        return "unique-changed"


def classify_subtrees_using_set_ops(
    tree_ref: Node, tree_target: Node
) -> Dict[Node, str]:
    """
    For each internal node in tree_target, classify its topological relationship to tree_ref.

    Classifications:
      - 'full-common'   : The node's split and its entire descendant topology exist in tree_ref.
      - 'common-changed': The node's split exists in tree_ref, but its descendant topology differs.
      - 'full-unique'   : The node's split and its descendant topology are entirely absent from tree_ref.
      - 'unique-changed': A mix of unique and common splits in the descendants.
    """
    _ensure_split_indices(tree_ref)
    _ensure_split_indices(tree_target)

    s_ref: PartitionSet[Partition] = tree_ref.to_splits()
    s_tgt: PartitionSet[Partition] = tree_target.to_splits()
    s_common: PartitionSet[Partition] = s_ref & s_tgt

    subtree_splits_map: Dict[Node, PartitionSet[Partition]] = {}
    _collect_subtree_splits(tree_target, subtree_splits_map)

    classification_map: Dict[Node, str] = {}
    for node, splits in subtree_splits_map.items():
        if node.children:  # Only classify internal nodes
            classification_map[node] = _classify_node(node, splits, s_ref, s_common)

    return classification_map


def reorder_tree_if_full_common(
    reference_tree: Node,
    target_tree: Node,
    orientation_map: Dict[Partition, List[FrozenSet[str]]],
) -> Dict[Partition, List[FrozenSet[str]]]:
    """
    Reorders 'full-common' nodes in the target_tree to match the reference_tree's orientation.

    For each split in `orientation_map`, if the corresponding node in `target_tree` is
    classified as 'full-common', its children are reordered. Otherwise, the split is
    removed from the map, preventing further propagation.

    Args:
        reference_tree: The tree providing the correct topology.
        target_tree: The tree to be modified.
        orientation_map: A map from a split to its desired child leaf sets orientation.
                         This map is mutated in place.

    Returns:
        The mutated orientation_map, containing only splits that are still pending propagation.
    """
    if not orientation_map:
        return orientation_map

    classification_map = classify_subtrees_using_set_ops(reference_tree, target_tree)
    splits_to_remove: List[Partition] = []

    # Use _ for the value since it's not used in the loop
    for sp, _ in orientation_map.items():
        node_in_tgt = target_tree.find_node_by_split(sp)
        node_in_rf = reference_tree.find_node_by_split(sp)

        if node_in_tgt is None or node_in_rf is None:
            splits_to_remove.append(sp)
            continue

        # Reorder only if the subtree topology is identical ('full-common')
        if classification_map.get(node_in_tgt) == "full-common":
            node_in_tgt.reorder_taxa(list(node_in_rf.get_current_order()))
        else:
            # If not 'full-common', this orientation cannot be propagated here.
            splits_to_remove.append(sp)

    # Clean up the map for subsequent propagation steps
    for sp in splits_to_remove:
        orientation_map.pop(sp, None)

    return orientation_map


##################################################
#      Orientation Map Builder
##################################################


def build_orientation_map(
    tree: Node, rotated_splits: PartitionSet[Partition]
) -> Dict[Partition, List[FrozenSet[str]]]:
    """
    For each rotated split, capture its resulting child orientation.

    An orientation is represented as an ordered list of its children's leaf sets.

    Args:
        tree: The tree from which to capture the orientations.
        rotated_splits: The set of splits whose orientations are needed.

    Returns:
        A dictionary mapping each split to its child leaf set orientation.
    """
    orientation_map: Dict[Partition, List[FrozenSet[str]]] = {}

    for sp in rotated_splits:
        node = tree.find_node_by_split(sp)
        if node and node.children:
            orientation_map[sp] = [
                frozenset(ch.get_current_order()) for ch in node.children
            ]

    return orientation_map
