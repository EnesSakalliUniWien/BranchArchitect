from typing import Dict, List, Tuple, Optional
from brancharchitect.tree import Node
from brancharchitect.partition_set import PartitionSet
from brancharchitect.partition_set import Partition

##################################################
#           Classification Logic
##################################################


def _ensure_split_indices(tree: Node):
    """Ensure split_indices are initialized for all nodes in the tree."""
    if not hasattr(tree, "split_indices") or tree.split_indices is None:
        if hasattr(tree, "_encoding"):
            tree._initialize_split_indices(tree._encoding)
        else:
            raise ValueError(
                f"{tree} is missing _encoding for split_indices initialization."
            )

def _collect_subtree_splits(
    node: Node, subtree_splits_map: Dict[Node, PartitionSet]
) -> PartitionSet:
    """Recursively collect all splits in the subtree rooted at node."""
    subs: PartitionSet = PartitionSet()
    for ch in node.children:
        subs |= _collect_subtree_splits(ch, subtree_splits_map)
    if node.children:
        subs.add(node.split_indices)
    subtree_splits_map[node] = subs
    return subs


def _classify_node(
    node: Node, splits: PartitionSet, Sref: PartitionSet, S_common: PartitionSet
) -> str:
    """Classify a node based on its split membership and subtree splits."""
    node_in_ref = node.split_indices in Sref
    fully_in_ref = splits.issubset(S_common)
    fully_out_ref = splits.isdisjoint(S_common)
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
    For each internal node in tree_target, classify as:
      - 'full-common'   : node's split is in tree_ref & entire subtree is in tree_ref
      - 'common-changed': node's split is in tree_ref but subtree partially differs
      - 'full-unique'   : node's split not in tree_ref, subtree fully outside
      - 'unique-changed': partial overlap
    """
    _ensure_split_indices(tree_ref)
    _ensure_split_indices(tree_target)

    Sref: PartitionSet = tree_ref.to_splits()
    Stgt: PartitionSet = tree_target.to_splits()
    S_common: PartitionSet = Sref & Stgt

    subtree_splits_map: Dict[Node, PartitionSet] = {}
    _collect_subtree_splits(tree_target, subtree_splits_map)

    classification_map: Dict[Node, str] = {}
    for node, splits in subtree_splits_map.items():
        classification_map[node] = _classify_node(node, splits, Sref, S_common)

    return classification_map


def reorder_tree_if_full_common(
    reference_tree: Node,
    target_tree: Node,
    orientation_map: Dict[Partition, List[frozenset]],
) -> Dict[Partition, List[frozenset]]:
    """
    For each split in orientation_map, if target_tree sees it as 'full-common',
    reorder children to match reference_tree. If not, remove that split from orientation_map.
    We keep optimizing other splits.
    """
    if not orientation_map:
        return orientation_map

    classification_map = classify_subtrees_using_set_ops(reference_tree, target_tree)
    splits_to_remove = []

    for sp, child_leafsets in orientation_map.items():
        node_in_tgt = target_tree.find_node_by_split(sp)
        node_in_rf = reference_tree.find_node_by_split(sp)

        # Add null check before using node_in_tgt as a dictionary key
        if node_in_tgt is None or node_in_rf is None:
            splits_to_remove.append(sp)
            continue

        node_class = classification_map.get(node_in_tgt, None)
        if node_class == "full-common":
            # Convert tuple to list before passing to reorder_taxa
            node_in_tgt.reorder_taxa(list(node_in_rf.get_current_order()))
        else:
            # not full-common => remove
            splits_to_remove.append(sp)
    for sp in splits_to_remove:
        orientation_map.pop(sp, None)
    return orientation_map


##################################################
#   Forward & Backward Orientation Prop
##################################################
def build_orientation_map(
    tree: Node, rotated_splits: PartitionSet
) -> Dict[Partition, List[frozenset]]:
    """
    For each rotated split in `tree`, gather final child orientation
    as a list of leaf-subsets in order.
    """
    orientation_map = {}
    
    for sp in rotated_splits:
        node = tree.find_node_by_split(sp)
        # Add null check before accessing node.children
        if node is not None:
            orientation_map[sp] = [
                frozenset(ch.get_current_order()) for ch in node.children
            ]

        # Skip this split if node is None
    return orientation_map

