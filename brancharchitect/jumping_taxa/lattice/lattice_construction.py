from brancharchitect.tree import Node
from brancharchitect.partition_set import PartitionSet, count_full_overlaps
from brancharchitect.jumping_taxa.lattice.lattice_edge import LatticeEdge
from brancharchitect.jumping_taxa.lattice.matrix_ops import PMatrix
from brancharchitect.jumping_taxa.debug import (
    jt_logger,
    format_set,
)
from typing import (
    Optional,
    List,
    Tuple,
)  # Added TypingSet, TypeVar, cast


def get_child_splits(node: Node) -> PartitionSet:
    """
    Compute the set of child splits for a node n.

    Definition:
      If n has children { c₁, c₂, …, cₖ }, then D(n) = { s(c) : s(c) = c.split_indices for each c ∈ children(n) }.
    """
    # Ensure node._encoding is not None by using "or {}"
    return PartitionSet(
        {child.split_indices for child in node.children}, encoding=node._encoding or {}
    )


def construct_sub_lattices(tree1: Node, tree2: Node) -> Optional[List[LatticeEdge]]:
    """Compute detailed split information for two trees."""
    # Ensure both trees have their indices built

    # Get splits with validation
    S1: PartitionSet = tree1.to_splits()
    S2: PartitionSet = tree2.to_splits()

    # Get common splits and verify they exist in both trees
    U: PartitionSet = S1.union(S2)
    sub_lattices: List[LatticeEdge] = []

    jt_logger.compare_tree_splits(tree1=tree1, tree2=tree2)

    for s in U:
        if len(s) == 1:
            continue

        if s in S1 and s in S2:
            n_left: Optional[Node] = tree1.find_node_by_split(s)
            n_right: Optional[Node] = tree2.find_node_by_split(s)

            if n_left is None or n_right is None:
                raise ValueError(f"Split {format_set(s)} not found in both trees")

            node_meet: PartitionSet = n_left.to_splits(
                with_leaves=True
            ) & n_right.to_splits(with_leaves=True)

            D1: PartitionSet = get_child_splits(n_left)

            D2: PartitionSet = get_child_splits(n_right)

            direct_child_meet: PartitionSet = D1 & D2

            # Process further if at least one node has a nontrivial mixture of child splits.
            if D1 != direct_child_meet or direct_child_meet != D2:
                node_meet.discard(s)

                jt_logger.info(f"Processing common split {format_set(s)} in both trees")

                # Fix: Use node_meet directly instead of wrapping it in another PartitionSet
                t1_common_covers: List[PartitionSet] = compute_cover_elements(
                    n_left, D1, node_meet
                )

                t2_common_covers: List[PartitionSet] = compute_cover_elements(
                    n_right, D2, node_meet
                )

                t1_unique_atoms: List[PartitionSet] = compute_unique_atoms(
                    n_left, n_right, D1
                )

                t2_unique_atoms: List[PartitionSet] = compute_unique_atoms(
                    n_right, n_left, D2
                )

                t1_unique_covers: List[PartitionSet] = compute_unique_covers(
                    n_left, n_right, D1
                )

                t2_unique_covers: List[PartitionSet] = compute_unique_covers(
                    n_right, n_left, D2
                )

                t1_unique_partition_sets: List[PartitionSet] = (
                    compute_unique_partition_sets(n_left, n_right, D1)
                )
                t2_unique_partition_sets: List[PartitionSet] = (
                    compute_unique_partition_sets(n_right, n_left, D2)
                )

                sub_lattices.append(
                    LatticeEdge(
                        split=s,
                        t1_common_covers=t1_common_covers,
                        t2_common_covers=t2_common_covers,
                        child_meet=direct_child_meet,
                        left_node=n_left,
                        right_node=n_right,
                        encoding=tree1._encoding,
                        t1_unique_atoms=t1_unique_atoms,
                        t2_unique_atoms=t2_unique_atoms,
                        t1_unique_covers=t1_unique_covers,
                        t2_unique_covers=t2_unique_covers,
                        t1_unique_partition_sets=t1_unique_partition_sets,
                        t2_unique_partition_sets=t2_unique_partition_sets,
                    )
                )

    return sub_lattices


def compute_cover_elements(
    node: Node, child_splits: PartitionSet, common_excluding: PartitionSet
) -> List[PartitionSet]:
    """
    For each child split of a node, compute its covering element.

    Cover(child) = (child.to_splits(with_leaves=True) ∩ common_excluding).cover()

    Returns a PartitionSet of cover elements.
    """
    cover_list: List[PartitionSet] = []
    for split in child_splits:
        child_node = node.find_node_by_split(split)
        if child_node is not None:
            node_splits: PartitionSet = child_node.to_splits(with_leaves=True)
            cover_candidate: PartitionSet = node_splits & common_excluding
            covering_element: PartitionSet = cover_candidate.cover()
            cover_list.append(covering_element)
        else:
            raise ValueError(f"Split {split} not found in tree {node}")
    # Add safety check for arms
    if not cover_list:
        raise ValueError(f"Arms not found for split {node.split_indices} ")
    return cover_list


def compute_unique_atoms(
    node_target: Node,
    node_reference: Node,
    child_splits: PartitionSet,
) -> List[PartitionSet]:
    unique_atom_sets: List[PartitionSet] = []

    for split in child_splits:
        child_node = node_target.find_node_by_split(split)
        if child_node:
            node_splits: PartitionSet = child_node.to_splits()
            unique_splits_target: PartitionSet = (
                node_splits - node_reference.to_splits()
            )
            unique_atom_sets.append(unique_splits_target.atom())
    return unique_atom_sets


def compute_unique_covers(
    node_target: Node,
    node_reference: Node,
    child_splits: PartitionSet,
) -> List[PartitionSet]:
    unique_cover_sets: List[PartitionSet] = []
    for split in child_splits:
        child_node = node_target.find_node_by_split(split)
        if child_node:
            node_splits: PartitionSet = child_node.to_splits()
            unique_splits_target: PartitionSet = (
                node_splits - node_reference.to_splits()
            )
            unique_cover_sets.append(unique_splits_target.cover())
    return unique_cover_sets


def compute_unique_partition_sets(
    node_target: Node, node_reference: Node, child_splits: PartitionSet
) -> List[PartitionSet]:
    unique_atom_sets: List[PartitionSet] = []
    for split in child_splits:
        child_node = node_target.find_node_by_split(split)
        if child_node:
            node_splits: PartitionSet = child_node.to_splits()
            unique_splits_target: PartitionSet = (
                node_splits - node_reference.to_splits()
            )
            unique_atom_sets.append(unique_splits_target)
    return unique_atom_sets


def build_partition_conflict_matrix(s_edge: LatticeEdge) -> Optional[PMatrix]:
    """
    Identifies conflicting pairs of covers between two trees and returns them as a matrix.

    Each row in the returned matrix contains a conflicting pair [t1_cover, t2_cover].

    Args:
        s_edge: A LatticeEdge object containing cover information from both trees

    Returns:
        A matrix (list of lists) of conflicting PartitionSet pairs, or None if
        no conflicts are found or if cover lists are empty.
    """
    t1_covers = s_edge.t1_common_covers
    t2_covers = s_edge.t2_common_covers

    # Early return if either cover list is empty
    if not t1_covers or not t2_covers:
        jt_logger.warning("Cannot build matrix with empty cover lists.")
        return None

    jt_logger.info(
        f"Building Partition Conflict Matrix for edge {format_set(set(s_edge.split))} "
        f"({len(t1_covers)} T1 covers vs {len(t2_covers)} T2 covers)"
    )

    conflict_matrix: PMatrix = []

    # Check for singleton covers (special case handling)
    singleton_index_t1 = find_first_singleton_cover_index(t1_covers)
    singleton_index_t2 = find_first_singleton_cover_index(t2_covers)

    # Case 1: Both trees have singleton covers
    if singleton_index_t1 is not None and singleton_index_t2 is not None:
        return _handle_both_trees_with_singletons(
            t1_covers,
            t2_covers,
            s_edge.t1_unique_partition_sets,
            s_edge.t2_unique_partition_sets,
        )

    # Case 2: General case - find all conflicting pairs
    for t1_cover in t1_covers:
        for t2_cover in t2_covers:
            intersection = t1_cover & t2_cover
            t1_minus_t2 = t1_cover - t2_cover
            t2_minus_t1 = t2_cover - t1_cover

            # A conflict exists when all three sets are non-empty
            if intersection and t1_minus_t2 and t2_minus_t1:
                conflict_matrix.append([t1_cover, t2_cover])

    return conflict_matrix


def _handle_both_trees_with_singletons(
    t1_covers: List[PartitionSet],
    t2_covers: List[PartitionSet],
    t1_unique_splits: List[PartitionSet],
    t2_unique_splits: List[PartitionSet],
) -> PMatrix:
    """Helper function to handle the case when both trees have singleton covers."""
    conflict_matrix: PMatrix = []

    # Get singleton PartitionSets
    t1_singleton = return_singleton_partitionsset(t1_covers)
    t2_singleton = return_singleton_partitionsset(t2_covers)

    if t1_singleton is None or t2_singleton is None:
        raise ValueError("No singleton PartitionSet found in one or both cover lists.")

    jt_logger.info(
        f"Singletons found in both trees: {format_set(set(t1_singleton))} and {format_set(set(t2_singleton))}"
    )

    # Get the first partition from each singleton (since singleton is a PartitionSet of length 1)
    partition1 = next(iter(t1_singleton))
    partition2 = next(iter(t2_singleton))

    # Find conflicting non-singleton PartitionSets
    t1_conflict = find_conflicting_singleton_partition_set(t1_covers, t2_singleton)
    t2_conflict = find_conflicting_singleton_partition_set(t2_covers, t1_singleton)

    if t1_conflict is None or t2_conflict is None:
        raise ValueError(
            "No conflicting singleton PartitionSet found in one or both cover lists."
        )

    index_t1_conflict, t1_conflict_set = t1_conflict
    index_t2_conflict, t2_conflict_set = t2_conflict

    # Count overlaps to determine which conflicts to include
    overlaps_from_t1 = count_full_overlaps(
        partition1, t2_unique_splits[index_t2_conflict]
    )
    overlaps_from_t2 = count_full_overlaps(
        partition2, t1_unique_splits[index_t1_conflict]
    )

    if overlaps_from_t1 > overlaps_from_t2:
        conflict_matrix.append([t1_singleton, t2_conflict_set])
    elif overlaps_from_t1 < overlaps_from_t2:
        conflict_matrix.append([t1_conflict_set, t2_singleton])
    else:  # Equal overlaps
        conflict_matrix.append([t1_conflict_set, t2_singleton])
        conflict_matrix.append([t1_singleton, t2_conflict_set])

    return conflict_matrix


def find_first_singleton_cover_index(
    list_of_covers: List[PartitionSet],
) -> Optional[int]:
    """
    Finds the index of the first PartitionSet in a list that represents a singleton.

    A singleton cover is defined as a PartitionSet containing exactly one Partition,
    and that Partition itself contains exactly one index.

    Args:
        covers: A list of PartitionSet objects to search through.

    Returns:
        The integer index of the first singleton PartitionSet found, or None if
        no singleton is found in the list.
    """
    for index, covers in enumerate(list_of_covers):
        # Check if the PartitionSet has exactly one Partition
        if len(covers) == 1:
            return index  # Found a singleton, return its index
    return None  # No singleton found in the list


def return_singleton_partitionsset(
    covers: List[PartitionSet],
) -> Optional[PartitionSet]:
    """
    Returns the first singleton PartitionSet from a list of PartitionSet objects.

    Args:
        covers: A list of PartitionSet objects to search through.

    Returns:
        The first singleton PartitionSet found in the list.
        Returns None if no singleton is found.
    """
    for cover in covers:
        if len(cover) == 1:
            return cover
    return None


def find_conflicting_singleton_partition_set(
    search_covers_list: List[PartitionSet],
    singleton_partition_set: PartitionSet,
) -> Optional[Tuple[int, PartitionSet]]:
    """
    Finds the first singleton PartitionSet in a list of PartitionSets.

    A singleton PartitionSet is defined as one that contains exactly one Partition.

    Args:
        partition_sets: A list of PartitionSet objects to search through.

    Returns:
        The first singleton PartitionSet found, or None if no singleton is found.
    """
    for i, reference_paritition_set in enumerate(search_covers_list):
        if singleton_partition_set.issubset(reference_paritition_set):
            return i, reference_paritition_set
    return None


def are_cover_lists_equivalent(
    list1: List[PartitionSet], list2: List[PartitionSet]
) -> bool:
    """
    Checks if two lists of PartitionSet objects contain the same elements,
    ignoring order and handling potential duplicates.

    Args:
        list1: The first list of PartitionSet objects.
        list2: The second list of PartitionSet objects.

    Returns:
        True if the lists contain the same PartitionSets (ignoring order),
        False otherwise.
    """
    if len(list1) != len(list2):
        return False

    # Create a mutable copy of the second list to remove items from
    list2_copy = list(list2)

    for item1 in list1:
        found_match_in_list2 = False
        # Iterate through the current state of the copy
        for i in range(
            len(list2_copy) - 1, -1, -1
        ):  # Iterate backwards for safe removal
            item2 = list2_copy[i]
            # Use the equality check defined for PartitionSet
            if item1 == item2:
                list2_copy.pop(i)  # Remove the matched item
                found_match_in_list2 = True
                break  # Found a match for item1, move to the next item1

        # If after checking all items in list2_copy, no match was found for item1
        if not found_match_in_list2:
            return False

    # If the loop completes, it means every item in list1 found a match
    # Since the lengths were initially equal, list2_copy should now be empty
    # The function returns True implicitly if it hasn't returned False yet.
    return True