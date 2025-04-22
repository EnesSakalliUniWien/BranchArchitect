from typing import Optional, List
from brancharchitect.tree import Node
from brancharchitect.partition_set import PartitionSet
from brancharchitect.jumping_taxa.lattice.lattice_edge import LatticeEdge
from pydantic import validate_call
from brancharchitect.jumping_taxa.debug import (
    jt_logger,
    format_set,
)


@validate_call
def get_child_splits(node: Node) -> PartitionSet:
    """
    Compute the set of child splits for a node n.

    Definition:
      If n has children { c₁, c₂, …, cₖ }, then D(n) = { s(c) : s(c) = c.split_indices for each c ∈ children(n) }.
    """
    # Ensure node._encoding is not None by using "or {}"
    return PartitionSet(
        {child.split_indices for child in node.children}, look_up=node._encoding or {}
    )


@validate_call
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
                left_covers: List[PartitionSet] = compute_covet_elements(
                    n_left, D1, node_meet
                )

                right_covers: List[PartitionSet] = compute_covet_elements(
                    n_right, D2, node_meet
                )

                left_unique_atoms: List[PartitionSet] = compute_unique_atoms(
                    n_left, n_right, D1
                )

                rigth_unique_atoms: List[PartitionSet] = compute_unique_atoms(
                    n_right, n_left, D2
                )

                left_unique_covet: List[PartitionSet] = compute_unique_covet(
                    n_left, n_right, D1
                )

                rigth_unique_covet: List[PartitionSet] = compute_unique_covet(
                    n_right, n_left, D2
                )

                sub_lattices.append(
                    LatticeEdge(
                        split=s,
                        left_cover=left_covers,
                        right_cover=right_covers,
                        child_meet=direct_child_meet,
                        left_node=n_left,
                        right_node=n_right,
                        look_up=tree1._encoding,
                        left_unique_atoms=left_unique_atoms,
                        right_unique_atoms=rigth_unique_atoms,
                        left_unique_covet=left_unique_covet,
                        right_unique_covet=rigth_unique_covet,
                    )
                )

    return sub_lattices


def compute_unique_atoms(
    node_target: Node,
    node_reference: Node,
    child_splits: PartitionSet,
) -> List[PartitionSet]:

    unique_atom_target: List[PartitionSet] = []

    for split in child_splits:
        child_node = node_target.find_node_by_split(split)
        if child_node:
            node_splits: PartitionSet = child_node.to_splits()
            unique_splits_target: PartitionSet = (
                node_splits - node_reference.to_splits()
            )
            unique_atom_target.append(unique_splits_target.atom())
    return unique_atom_target


def compute_unique_covet(
    node_target: Node,
    node_reference: Node,
    child_splits: PartitionSet,
) -> List[PartitionSet]:

    unique_atom_target: List[PartitionSet] = []

    for split in child_splits:
        child_node = node_target.find_node_by_split(split)
        if child_node:
            node_splits: PartitionSet = child_node.to_splits()
            unique_splits_target: PartitionSet = (
                node_splits - node_reference.to_splits()
            )
            unique_atom_target.append(unique_splits_target.cover())
    return unique_atom_target


@validate_call
def compute_covet_elements(
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


@validate_call
def is_independent_any(tuple_boolean: tuple[bool, ...]) -> bool:
    """
    Return True if any element in tuple_boolean is True, using the built-in any() function.
    """
    return any(tuple_boolean)


@validate_call
def gather_independent_partitions(
    intersection_map: dict[frozenset, dict[str, PartitionSet]],
    left_minus_right_map: dict[frozenset, dict[str, PartitionSet]],
    right_minus_left_map: dict[frozenset, dict[str, PartitionSet]],
) -> list[dict[str, PartitionSet]]:

    independent_sides: list[dict[str, PartitionSet]] = []

    for common_partition in intersection_map:
        left_entry: dict[str, PartitionSet] = left_minus_right_map[common_partition]
        right_entry: dict[str, PartitionSet] = right_minus_left_map[common_partition]

        independent_left: PartitionSet = left_entry["covet_left"]
        independent_right: PartitionSet = right_entry["covet_right"]

        conditions: tuple[bool, ...] = check_independence_conditions(
            left_entry, right_entry
        )
        is_independent: bool = is_independent_any(conditions)

        jt_logger.info(f"Final independence determination: {is_independent}")

        if is_independent:
            independent_sides.append(
                {
                    "A": independent_left,
                    "B": independent_right,
                }
            )

    return independent_sides


def check_non_subsumption_with_residual(
    primary_set: PartitionSet, comparison_set: PartitionSet, residual: PartitionSet
) -> bool:
    """
    Check if a set is not fully contained in another set
    and has a non-empty residual.

    Args:
        primary_set: The set being checked for subsumption
        comparison_set: The set to compare against
        residual: The complement or difference set

    Returns:
        True if the primary set is not a subset and has a non-empty residual
    """
    return bool(residual) and not primary_set.issubset(comparison_set)


def check_atomic_inclusion(
    primary_set: PartitionSet, comparison_set: PartitionSet
) -> bool:
    """
    Check for atomic inclusion condition.

    An atomic inclusion occurs when:
    1. The primary set is not a subset of the comparison set
    2. The primary set contains exactly one element
    3. The comparison set contains more than one element

    Args:
        primary_set: The set being checked
        comparison_set: The set to compare against

    Returns:
        True if atomic inclusion condition is met
    """
    return (
        not primary_set.issubset(comparison_set)
        and len(primary_set) == 1
        and len(comparison_set) > 1
    )


@validate_call
def check_independence_conditions(
    left: dict[str, PartitionSet], right: dict[str, PartitionSet]
) -> tuple[bool, bool, bool, bool]:
    """
    Evaluate independence conditions between two partition sets.

    Returns a tuple of boolean conditions:
    1. Left non-subsumption with right residual
    2. Right non-subsumption with left residual
    3. Left atomic inclusion
    4. Right atomic inclusion

    Args:
        left: Dictionary containing left partition sets
        right: Dictionary containing right partition sets

    Returns:
        Tuple of four boolean conditions
    """
    # Extract partition sets
    left_arm = left.get("covet_left", PartitionSet())
    right_arm = right.get("covet_right", PartitionSet())

    # Condition 1: Left partition is not fully contained
    # and has a non-empty right-side residual
    left_non_subsumption = check_non_subsumption_with_residual(
        primary_set=left_arm,
        comparison_set=right_arm,
        residual=left.get("b-a", PartitionSet()),
    )

    # Condition 2: Right partition is not fully contained
    # and has a non-empty left-side residual
    right_non_subsumption = check_non_subsumption_with_residual(
        primary_set=right_arm,
        comparison_set=left_arm,
        residual=right.get("a-b", PartitionSet()),
    )

    # Condition 3: Left atomic inclusion
    # (single element set not subset of a larger set)
    left_atomic_inclusion = check_atomic_inclusion(
        primary_set=left_arm, comparison_set=right_arm
    )

    # Condition 4: Right atomic inclusion
    # (single element set not subset of a larger set)
    right_atomic_inclusion = check_atomic_inclusion(
        primary_set=right_arm, comparison_set=left_arm
    )

    return (
        left_non_subsumption,
        right_non_subsumption,
        left_atomic_inclusion,
        right_atomic_inclusion,
    )


# ============================================== Case For Edge Types ====================================================== #


@validate_call
def pairwise_lattice_analysis(
    s_edge: LatticeEdge,
) -> tuple[
    dict[frozenset, dict[str, PartitionSet]],
    dict[frozenset, dict[str, PartitionSet]],
    dict[frozenset, dict[str, PartitionSet]],
]:
    """
    Perform pairwise lattice analysis of the left_cover and right_cover in s_edge.

    Specifically, computes:
      - Intersection (i = a & b)
      - Directional differences (a_without_b = a - b, b_without_a = b - a)
    Accumulates these into tuples and dictionaries for subsequent analysis.
    """
    # Renamed for clarity:
    intersection_map: dict[frozenset, dict[str, PartitionSet]] = {}
    left_minus_right_map: dict[frozenset, dict[str, PartitionSet]] = {}
    right_minus_left_map: dict[frozenset, dict[str, PartitionSet]] = {}

    jt_logger.log_combined_data(
        arms_t_one=s_edge.left_cover,
        arms_t_two=s_edge.right_cover,
        left_unique_atoms=s_edge.left_unique_atoms,
        right_unique_atoms=s_edge.right_unique_atoms,
        left_unique_covet=s_edge.left_unique_covet,
        right_unique_covet=s_edge.right_unique_covet,
        look_up=s_edge.look_up,
    )

    jt_logger.log_cover_cartesian_product(s_edge.left_cover, s_edge.right_cover)

    for x, left in enumerate(s_edge.left_cover, 0):

        for y, right in enumerate(s_edge.right_cover, 0):

            i: PartitionSet = left & right

            left_minus_right: PartitionSet = left - right

            right_minus_left: PartitionSet = right - left

            jt_logger.info(f"Left: {x} {i} Right: {y}")

            if i:

                intersection_map[frozenset(i)] = {
                    "covet_left": left,
                    "covet_right": right,
                    "b-a": right_minus_left,
                    "a-b": left_minus_right,
                }

            if left_minus_right:

                left_minus_right_map[frozenset(left_minus_right)] = {
                    "covet_left": left,
                    "covet_right": right,
                    "b-a": right_minus_left,
                    "a-b": left_minus_right,
                }

            if right_minus_left:
                right_minus_left_map[frozenset(right_minus_left)] = {
                    "covet_left": left,
                    "covet_right": right,
                    "b-a": right_minus_left,
                    "a-b": left_minus_right,
                }

    # Add detailed logging of the maps
    jt_logger.log_map_details(
        intersection_map, left_minus_right_map, right_minus_left_map
    )

    # Return everything as a tuple.
    return (
        intersection_map,
        left_minus_right_map,
        right_minus_left_map,
    )
