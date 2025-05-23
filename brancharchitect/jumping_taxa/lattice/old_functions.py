from typing import Optional, List, Dict, Tuple, Iterable
from brancharchitect.tree import Node
from brancharchitect.partition_set import PartitionSet
from brancharchitect.jumping_taxa.lattice.lattice_edge import LatticeEdge
from brancharchitect.jumping_taxa.debug import (
    jt_logger,
    format_set,
)
from brancharchitect.jumping_taxa.lattice.matrix_ops import PMatrix


"""
def build_cover_pair_records(
    s_edge: LatticeEdge,
) -> tuple[
    dict[frozenset, dict[str, PartitionSet]],
    dict[frozenset, dict[str, PartitionSet]],
    dict[frozenset, dict[str, PartitionSet]],
]:
    # Perform pairwise lattice analysis of the t1_common_covers and t2_common_covers in s_edge.
    # 
    # Specifically, computes:
    #   - Intersection (i = a & b)
    #   - Directional differences (a_without_b = a - b, b_without_a = b - a)
    # Accumulates these into tuples and dictionaries for subsequent analysis.
# Renamed for clarity:
    intersection_map: dict[frozenset, dict[str, PartitionSet]] = {}
    left_minus_right_map: dict[frozenset, dict[str, PartitionSet]] = {}
    right_minus_left_map: dict[frozenset, dict[str, PartitionSet]] = {}

    jt_logger.log_combined_data(
        arms_t_one=s_edge.t1_common_covers,
        arms_t_two=s_edge.t2_common_covers,
        t1_unique_atoms=s_edge.t1_unique_atoms,
        t2_unique_atoms=s_edge.t2_unique_atoms,
        t1_unique_covers=s_edge.t1_unique_covers,
        t2_unique_covers=s_edge.t2_unique_covers,
        look_up=s_edge.look_up,
    )

    jt_logger.log_cover_cartesian_product(
        s_edge.t1_common_covers, s_edge.t2_common_covers
    )

    for x, left in enumerate(s_edge.t1_common_covers, 0):
        for y, right in enumerate(s_edge.t2_common_covers, 0):
            i: PartitionSet = left & right

            left_minus_right: PartitionSet = left - right

            right_minus_left: PartitionSet = right - left

            jt_logger.info(f"Left: {x} {i} Right: {y}")

            if i:
                intersection_map[frozenset(i)] = {
                    "cover_left": left,
                    "cover_right": right,
                }

            if left_minus_right:
                left_minus_right_map[frozenset(left_minus_right)] = {
                    "cover_left": left,
                    "cover_right": right,
                }

            if right_minus_left:
                right_minus_left_map[frozenset(right_minus_left)] = {
                    "cover_left": left,
                    "cover_right": right,
                }

    # Add detailed logging of the maps
    # jt_logger.log_map_details(
    #    intersection_map, left_minus_right_map, right_minus_left_map
    # )
    # Return everything as a tuple.
    return (
        intersection_map,
        left_minus_right_map,
        right_minus_left_map,
    )


def collect_conflicted_cover_partitions(
    intersection_map: dict[frozenset, dict[str, PartitionSet]],
    left_minus_right_map: dict[frozenset, dict[str, PartitionSet]],
    right_minus_left_map: dict[frozenset, dict[str, PartitionSet]],
) -> list[dict[str, PartitionSet]]:
    compatible_sides: list[dict[str, PartitionSet]] = []

    for common_partition in intersection_map:
        left_entry: dict[str, PartitionSet] = left_minus_right_map[common_partition]
        right_entry: dict[str, PartitionSet] = right_minus_left_map[common_partition]

        independent_left: PartitionSet = left_entry["cover_left"]
        independent_right: PartitionSet = right_entry["cover_right"]

        conditions: tuple[bool, ...] = compute_cover_conflict_flags(
            left_entry, right_entry
        )

        has_conflict: bool = any(conditions)

        jt_logger.info(f"Final independence determination: {has_conflict}")

        if has_conflict:
            compatible_sides.append(
                {
                    "A": independent_left,
                    "B": independent_right,
                }
            )

    return compatible_sides



"""


def compute_cover_conflict_flags(
    left: Dict[str, PartitionSet],
    right: Dict[str, PartitionSet],
) -> Tuple[bool, bool, bool, bool]:
    """
    Produce four **conflict flags** for a pair of covers stored in two
    dictionaries (as used throughout *LatticeEdge* records).

    Dictionaries must expose at least these keys
        ── "cover_left"   (left-hand cover)
        ── "cover_right"  (right-hand cover)

    Any missing key is treated as an empty cover.

    Returned tuple
    --------------
    (0) **covers_incomparable**
        • each cover has ≥ 1 block the other lacks
        = (left ⊄ right) ∧ (right ⊄ left)

    (1) **covers_incomparable**  (Same value as 0 – kept for legacy shape)

    (2) **left_singleton_unique**
        • left cover is an atom (size 1)
        • that atom is absent from the right cover

    (3) **right_singleton_unique**
        • right cover is an atom
        • that atom is absent from the left cover
    """
    # ── fetch covers, default to ∅ so the logic never raises ──────────────
    left_cover: PartitionSet = left.get("cover_left", PartitionSet())
    right_cover: PartitionSet = right.get("cover_right", PartitionSet())

    # ── pre-compute the two subset relations only once ────────────────────
    left_subset_right: bool = left_cover.issubset(right_cover)
    right_subset_left: bool = right_cover.issubset(left_cover)

    # flags for “each side owns a private block”
    left_has_private = not left_subset_right
    right_has_private = not right_subset_left

    covers_incomparable = left_has_private and right_has_private

    # ── singleton-uniqueness tests (atomic covers) ────────────────────────
    left_singleton_unique = (
        len(left_cover) == 1  # left is an atom
        and len(right_cover) > 1  # right is not
        and left_has_private  # …and the atom is missing from right
    )
    right_singleton_unique = (
        len(right_cover) == 1 and len(left_cover) > 1 and right_has_private
    )

    return (
        covers_incomparable,
        left_singleton_unique,
        right_singleton_unique,
    )

""""""

def create_matrix(
    independent_directions: list[dict[str, PartitionSet]],
) -> Optional[PMatrix]:
    """
    Create a single matrix from direction analysis results.

    Args:
        direction_by_intersection: list of dictionaries with keys:s
            - "A": frozenset of elements from first set
            - "B": frozenset of elements from second set
            - "direction_a": Tuple indicating direction of first set
            - "direction_b": Tuple indicating direction of second set

    Returns:
        list[list[Set]]: A list containing a single matrix where each row is [A, B]
    """
    if not independent_directions:
        return []

    # Create a single matrix with all rows
    matrix: list = []
    for row in independent_directions:
        a_key: PartitionSet = row["A"]
        b_val: PartitionSet = row["B"]
        matrix.append([a_key, b_val])
    jt_logger.section("Matrix Construction")
    jt_logger.matrix(matrix)
    return matrix  # Return as a list of matrices for compatibility with existing code
