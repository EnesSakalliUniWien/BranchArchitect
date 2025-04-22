from typing import List, Dict
from brancharchitect.tree import Node
from brancharchitect.partition_set import Partition, PartitionSet
from collections import Counter

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
    split_list: PartitionSet, number_of_splits: Dict[Partition, Dict[str, float]]
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
    count_of_splits : dict[Partition, dict[str, float]]  = {}
    for tree in trees:
        splits : PartitionSet = tree.to_splits()
        incorporate_split_counts(splits, count_of_splits)
    for split in count_of_splits.keys():
        count_of_splits[split]["occurrence"] = count_of_splits[split]["count"] / len(
            trees
        )
    return count_of_splits

def create_star_tree(taxon_order: List[int]) -> Node:
    star_tree: Node = Node()
    star_tree.split_indices = Partition(tuple(i for i in range(len(taxon_order))))
    star_tree.name = "R"
    for taxon in taxon_order:
        new_node = Node(name=taxon, split_indices=taxon_order.index(taxon), length=1)
        star_tree.children.append(new_node)
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
    over_fifty_split = []  # List to store splits with occurrence over the threshold

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





def apply_split_in_tree(split: Partition, node: Node):
    split_set = set(map(int, split))  # Convert split to set of integers

    # Check if split_set is a subset of node.split_indices
    if (
        set(split_set).issubset(set(node.split_indices))
        and split_set != set(node.split_indices)
        or set(node.split_indices).issubset(set(split_set))
        and split_set != set(node.split_indices)
    ):

        remaining_children: list[Node] = []  # Added type annotation
        reassigned_children: list[Node] = []  # Added type annotation

        # Reassign children
        for child in node.children:

            if isinstance(child.split_indices, int):
                # Handle integer case
                if child.split_indices in split_set:
                    reassigned_children.append(child)
                else:
                    remaining_children.append(child)

            elif isinstance(child.split_indices, (list, set)):

                # Handle list or set case
                if set(child.split_indices).issubset(split_set) or set(
                    split_set
                ).issubset(child.split_indices):
                    reassigned_children.append(child)
                else:
                    remaining_children.append(child)

        if reassigned_children:
            new_node = Node(
                name="", split_indices=split_set, children=reassigned_children
            )
            # Update original node's children and append new node
            node.children = remaining_children
            node.children.append(new_node)

    # Recursively apply to children
    for child in node.children:
        if child.children:
            apply_split_in_tree(split, child)


def sort_splits(splits):
    return sorted(splits.items(), key=lambda x: (x[1], len(x[0]), x[0]), reverse=True)


def create_consensus_tree(trees: list[Node]) -> Node:
    tree = create_star_tree(trees[0]._order)
    observed_number_of_splits = collect_splits(trees)
    for split, frequency in sorted(
        observed_number_of_splits.items(), key=lambda x: (x[1], x[0]), reverse=True
    ):
        if frequency < 1.0:
            break
        apply_split_in_tree(split, tree)
    return tree


def create_majority_consensus_tree(trees: list[Node], threshold: float = 0.50) -> Node:
    tree = create_star_tree(trees[0]._order)
    splits = collect_splits(trees)
    for split, frequency in sort_splits(splits):
        if frequency <= threshold:
            break
        apply_split_in_tree(split, tree)
    return tree


def create_majority_consensus_tree_extended(trees: list[Node]) -> Node:
    """
    Create an extended majority consensus tree from a list of trees by applying mutually
    compatible splits.

    The function begins by constructing a star tree based on the taxon order of the first tree.
    It then collects split frequencies from all trees using ``collect_splits()``. The splits are
    sorted in descending order based on a tuple of criteria: frequency, the size of the split,
    and the lexicographical order of the split. For each split, the function checks if it is
    mutually compatible with all splits already applied to the tree (using ``compatible()``). If
    it is, then the split is applied to the tree (via ``apply_split_in_tree()``) and added to
    the list of applied splits.

    Args:
        trees (list[Node]): A list of tree nodes. Each tree should have an attribute ``_order``
            representing the taxon order and a method ``to_splits()`` that returns a collection
            of Partition splits.

    Returns:
        Node: The extended consensus tree that incorporates all mutually compatible splits.
    """
    tree: Node = create_star_tree(trees[0]._order)
    splits: dict[Partition, float] = collect_splits(trees)
    applied_splits: list[Partition] = []

    for split, frequency in sorted(
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
        trees (List[Node]): A list of tree nodes. Each tree is expected to have an 
            attribute ``_order`` representing the taxon order and a method ``to_splits()``
            returning a collection of Partition splits.

    Returns:
        dict[Partition, float]: A dictionary mapping each split (Partition) to its frequency. Frequency is given as a float value in the range [0, 1].
    """
    taxa: int = len(trees[0]._order)
    total_trees: int = len(trees)
    counter: Counter[Partition] = Counter()  # Counter mapping each Partition to an int
    for tree in trees:
        splits_in_tree = set(tree.to_splits())
        for split in splits_in_tree:
            if len(split) > 1 and len(split) < taxa:
                counter[split] += 1
    # Calculate frequencies and assign to a new variable:
    frequencies: dict[Partition, float] = {split: count / total_trees for split, count in counter.items()}
    return frequencies


def compatible(split1: Partition, split2: Partition):
    s1, s2 = set(split1), set(split2)
    return (s1 <= s2) or (s1 >= s2) or len((s1 & s2)) == 0