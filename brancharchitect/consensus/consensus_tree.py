from typing import List, Dict
from brancharchitect.tree import Node
from brancharchitect.elements.partition_set import Partition, PartitionSet
from collections import Counter
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def get_taxa_circular_order(node: Node) -> List[str]:
    """
    Compute the circular order of taxa (leaf names) in a tree.

    This function traverses the tree recursively and returns a list of leaf names
    in the order they are encountered.

    Args:
        node (Node): The root node of the tree.

    Returns:
        List[str]: A list of taxon names in circular order.
    """
    taxa_order: List[str] = []
    _get_taxa_circular_order(node, taxa_order)
    return taxa_order


def _get_taxa_circular_order(node: Node, taxa_order: List[str]) -> None:
    """
    Recursive helper function to build the taxa circular order.

    Visits each leaf node and appends its name to `taxa_order`.

    Args:
        node (Node): A node in the tree.
        taxa_order (List[str]): The accumulator for taxon names.
    """
    if not node.children:
        taxa_order.append(node.name)
    for child in node.children:
        _get_taxa_circular_order(child, taxa_order)


def incorporate_split_counts(
    split_list: PartitionSet[Partition],
    number_of_splits: Dict[Partition, Dict[str, float]],
) -> None:
    """
    Increment the count for each split in the given split list.

    Only considers splits that contain more than one element.
    The function updates the dictionary in place. For each split with length > 1,
    if the split key does not exist, it is added with an initial "count" of 0,
    then incremented by one.

    Args:
        split_list (PartitionSet): A collection of splits from a tree.
        number_of_splits (Dict[Partition, Dict[str, int]]): A dictionary mapping each split
            (Partition) to a dict holding its count (and possibly other statistics).

    Returns:
        None
    """
    for sp in filter(lambda s: len(s) > 1, split_list):
        number_of_splits.setdefault(sp, {"count": 0})["count"] += 1


def collect_count_of_splits(trees: List[Node]) -> dict[Partition, dict[str, float]]:
    count_of_splits: dict[Partition, dict[str, float]] = {}
    for tree in trees:
        splits: PartitionSet[Partition] = tree.to_splits()
        incorporate_split_counts(splits, count_of_splits)
    for split in count_of_splits.keys():
        count_of_splits[split]["occurrence"] = count_of_splits[split]["count"] / len(
            trees
        )
    return count_of_splits


def create_star_tree(taxon_order: List[str]) -> Node:
    encoding = {name: i for i, name in enumerate(taxon_order)}

    # Create children first
    child_nodes: List[Node] = []
    for taxon in taxon_order:
        child_nodes.append(
            Node(
                name=taxon,
                split_indices=Partition((encoding[taxon],), encoding),
                length=1,
                taxa_encoding=encoding,
            )
        )

    # Create parent with children
    star_tree = Node(name="R", children=child_nodes, taxa_encoding=encoding)
    star_tree.split_indices = Partition(tuple(range(len(taxon_order))), encoding)

    return star_tree


def filter_by_occurrence(
    splits: dict[Partition, dict[str, float]], eq: float
) -> List[Partition]:
    """
    Filter splits whose occurrence exceeds a given threshold.

    Args:
        splits (dict[Partition, dict[str, float]]): A dictionary mapping each split (Partition)
            to its count and occurrence statistics.
        eq (float): The minimum occurrence threshold required for a split to be included.

    Returns:
        List[Partition]: A list of splits (Partition objects) that have an occurrence above the threshold,
        sorted in ascending order by the size of the split.
    """
    over_fifty_split: List[
        Partition
    ] = []  # List to store splits with occurrence over the threshold

    # Filter splits by occurrence and size
    for split in splits:
        if splits[split]["occurrence"] > eq and len(split) > 1:
            over_fifty_split.append(split)

    # Sort the splits by the size of their tuple/list representation
    over_fifty_split.sort(key=lambda x: len(x))

    return over_fifty_split


def check_split_memory_compatibility(
    split_1: Partition, tree_order: List[int], memory: List[Partition]
) -> bool:
    """
    Check if a given split (split_1) is compatible with each split in memory,
    based on the provided tree order.

    Args:
        split_1 (Partition): The split to check.
        tree_order (List[int]): The overall tree order.
        memory (List[Partition]): A list of already accepted splits.

    Returns:
        bool: True if no memory split is compatible with split_1, False otherwise.
    """
    for memory_split in memory:
        if check_split_compatibility(memory_split, split_1, tree_order):
            return False
    return True


def check_split_compatibility(
    split_1: Partition, split_2: Partition, tree_order: List[int]
) -> bool:
    """
    Check if two splits are compatible with respect to the full set of taxa.

    Let U be the set of all taxa (derived from tree_order), and define:
        A = set(split_1)
        B = set(split_2)
        A_complement = U - A
        B_complement = U - B

    Then, splits are compatible if at least one of the four intersections is empty:
        - A ∩ B
        - A ∩ B_complement
        - A_complement ∩ B
        - A_complement ∩ B_complement

    Args:
        split_1 (Partition): The first split.
        split_2 (Partition): The second split.
        tree_order (List[int]): The complete set of taxa indices (defining U).

    Returns:
        bool: True if the splits are compatible (i.e. at least one of the intersections is empty),
              False otherwise.
    """
    U = set(tree_order)
    A = set(split_1)
    B = set(split_2)
    A_complement = U - A
    B_complement = U - B

    if not (A & B):
        return True
    if not (A & B_complement):
        return True
    if not (A_complement & B):
        return True
    if not (A_complement & B_complement):
        return True
    return False


def filter_incompatible_splits(
    sorted_splits_by_occurrence: list[Partition], tree_order: list[int]
) -> list[Partition]:
    """
    Filter out splits from a sorted list, retaining only those that are mutually incompatible.

    This function iterates over a list of splits (sorted, typically by their occurrence
    or size) and checks each split against a "memory" of already accepted splits. A
    split is added to the output only if it is incompatible with all previously accepted
    splits (based on the tree order), as determined by ``check_split_memory_compatibility``.

    Args:
        sorted_splits_by_occurrence (list[Partition]): A list of split partitions sorted
            by occurrence or another criterion.
        tree_order (list[int]): The overall tree order represented as a list of taxon indices.

    Returns:
        list[Partition]: A list of mutually incompatible (accepted) splits.
    """
    compatible_split_memory: list[Partition] = []
    for split in sorted_splits_by_occurrence:
        if check_split_memory_compatibility(split, tree_order, compatible_split_memory):
            compatible_split_memory.append(split)
    return compatible_split_memory


def apply_split_in_tree(split: Partition, node: Node, validate: bool = True):
    """
    Apply a split to a tree by creating a new internal node grouping the appropriate children.

    Args:
        split: The partition to apply to the tree
        node: The root node of the tree
        validate: If True, verifies the split was successfully applied (default: True)

    Raises:
        ValueError: If encoding cannot be converted
        RuntimeError: If validate=True and split was not successfully applied
    """
    # Store original split for validation
    original_split = split

    # Re-encode split to the target tree's encoding if needed
    if split.encoding != node.taxa_encoding:
        try:
            names = [split.reverse_encoding[i] for i in split.indices]
            new_indices = tuple(sorted(node.taxa_encoding[name] for name in names))
            split = Partition(new_indices, node.taxa_encoding)
        except Exception as e:
            raise ValueError(f"Cannot re-encode split to target tree encoding: {e}")

    split_set = set(split)  # Convert split to set

    # Check if split_set is a proper subset of node.split_indices
    if split_set < set(node.split_indices):
        remaining_children: list[Node] = []  # Children not in the split
        reassigned_children: list[Node] = []  # Children in the split

        # Reassign children based on whether they belong to the split
        for child in node.children:
            child_split_set = set(child.split_indices)

            # If child's split indices are entirely within the split, reassign it
            if child_split_set.issubset(split_set):
                reassigned_children.append(child)
            else:
                remaining_children.append(child)

        # Only create a new node if we have children to reassign
        if reassigned_children and len(reassigned_children) > 1:
            new_node = Node(
                name="",
                split_indices=split,
                children=reassigned_children,
                length=0,
                taxa_encoding=node.taxa_encoding,
            )
            # Update original node's children and append new node
            node.children = remaining_children
            node.children.append(new_node)

    # Recursively apply to children
    for child in node.children:
        if child.children:
            apply_split_in_tree(
                split, child, validate=False
            )  # Don't validate recursively

    # Validate split was applied (only at top level)
    if validate:
        # Get root and refresh splits
        root = node.get_root()
        root.initialize_split_indices(root.taxa_encoding)

        # Force cache invalidation to get fresh splits
        root.invalidate_caches()

        # Check if split is now in tree
        tree_splits = root.to_splits()
        if split not in tree_splits:
            # Find incompatible splits that prevented application
            all_indices = set(split.encoding.values())
            incompatible_splits_to_collapse: List[Partition] = []

            for tree_split in tree_splits:
                if not split.is_compatible_with(tree_split, all_indices):
                    incompatible_splits_to_collapse.append(tree_split)

            # Collapse incompatible splits if found
            if incompatible_splits_to_collapse:
                logger.debug(
                    f"[CONSENSUS] Collapsing {len(incompatible_splits_to_collapse)} "
                    f"incompatible split(s) to apply split {list(original_split.indices)}"
                )

                # Collapse each incompatible split by finding its node and setting length to 0
                for incompatible_split in incompatible_splits_to_collapse:
                    _collapse_split_in_tree(root, incompatible_split)

                # Refresh tree structure
                root.initialize_split_indices(root.taxa_encoding)
                root.invalidate_caches()

                # Retry applying the split
                apply_split_in_tree(split, root, validate=True)
            else:
                # No incompatible splits found - different issue
                split_indices_list = [list(s.indices) for s in tree_splits]
                raise RuntimeError(
                    f"Failed to apply split {list(original_split.indices)} to tree. "
                    f"No incompatible splits detected. Tree has {len(tree_splits)} splits: "
                    f"{split_indices_list[:10]}... Target split not present."
                )


def _collapse_split_in_tree(node: Node, split_to_collapse: Partition) -> None:
    """
    Find and collapse a specific split in the tree by removing the internal node.

    Args:
        node: The root or current node to search from
        split_to_collapse: The partition representing the split to collapse
    """
    if not node.children:
        return

    # Check if any child has the split we want to collapse
    for child in node.children:
        if child.split_indices == split_to_collapse and child.children:
            # Found the node to collapse - splice its children up to parent
            logger.debug(
                f"[CONSENSUS] Collapsing split {list(split_to_collapse.indices)}"
            )

            # Remove this child from parent's children
            node.children.remove(child)

            # Add all grandchildren to parent
            for grandchild in child.children:
                grandchild.parent = node
                node.children.append(grandchild)

            return

        # Recursively search in child's subtree
        if child.children:
            _collapse_split_in_tree(child, split_to_collapse)


def sort_splits(splits: Dict[Partition, float]) -> list[Tuple[Partition, float]]:
    return sorted(splits.items(), key=lambda x: (x[1], len(x[0]), x[0]), reverse=True)


def create_consensus_tree(trees: list[Node]) -> Node:
    tree: Node = create_star_tree(list(trees[0].get_current_order()))
    observed_number_of_splits = collect_splits(trees)
    for split, frequency in sorted(
        observed_number_of_splits.items(), key=lambda x: (x[1], x[0]), reverse=True
    ):
        if frequency < 1.0:
            break
        apply_split_in_tree(split, tree)
    return tree


def create_majority_consensus_tree(trees: list[Node], threshold: float = 0.50) -> Node:
    tree: Node = create_star_tree(list(trees[0].get_current_order()))
    splits: Dict[Partition, float] = collect_splits(trees)
    for split, frequency in sort_splits(splits):
        if frequency <= threshold:
            break
        apply_split_in_tree(split, tree)
    return tree


def create_majority_consensus_tree_extended(trees: list[Node]) -> Node:
    """
    Create an extended majority consensus tree from a list of trees by applying mutually
    compatible splits.

    The function begins by constructing a star tree based on the current leaf order of the first tree.
    It then collects split frequencies from all trees using ``collect_splits()``. The splits are
    sorted in descending order based on a tuple of criteria: frequency, the size of the split,
    and the lexicographical order of the split. For each split, the function checks if it is
    mutually compatible with all splits already applied to the tree (using ``compatible()``). If
    it is, then the split is applied to the tree (via ``apply_split_in_tree()``) and added to
    the list of applied splits.

    Args:
        trees (list[Node]): A list of tree nodes. Each tree should have a method ``to_splits()`` that returns a collection
            of Partition splits.

    Returns:
        Node: The extended consensus tree that incorporates all mutually compatible splits.
    """
    tree: Node = create_star_tree(list(trees[0].get_current_order()))
    splits: dict[Partition, float] = collect_splits(trees)
    applied_splits: list[Partition] = []

    for split, _ in sorted(
        splits.items(), key=lambda x: (x[1], len(x[0]), x[0]), reverse=True
    ):
        if all(compatible(split, existing_split) for existing_split in applied_splits):
            apply_split_in_tree(split, tree)
            applied_splits.append(split)
    return tree


def collect_splits(trees: List[Node]) -> dict[Partition, float]:
    """
    Collect split frequencies from a list of trees.

    For each tree, this function extracts the splits using its ``to_splits()`` method
    and counts their occurrences. Only splits that contain more than one element and
    fewer elements than the total taxa (derived from the tree order) are considered.

    The frequency for each split is then computed as the number of trees in which
    the split appears divided by the total number of trees.

    Args:
        trees (List[Node]): A list of tree nodes. Each tree is expected to provide
            ``to_splits()`` returning a collection of Partition splits.

    Returns:
        dict[Partition, float]: A dictionary mapping each split (Partition) to its frequency. Frequency is given as a float value in the range [0, 1].
    """
    taxa_count: int = len(list(trees[0].get_current_order()))
    total_trees: int = len(trees)
    counter: Counter[Partition] = Counter()  # Counter mapping each Partition to an int
    for tree in trees:
        splits_in_tree = set(tree.to_splits(with_leaves=True))
        for split in splits_in_tree:
            if len(split) > 1 and len(split) < taxa_count:
                counter[split] += 1
    # Calculate frequencies and assign to a new variable:
    frequencies: dict[Partition, float] = {
        split: count / total_trees for split, count in counter.items()
    }
    return frequencies


def compatible(split1: Partition, split2: Partition) -> bool:
    s1, s2 = set(split1), set(split2)
    return (s1 <= s2) or (s1 >= s2) or len((s1 & s2)) == 0
