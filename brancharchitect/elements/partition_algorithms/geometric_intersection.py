import itertools
from typing import Dict, Set, Tuple
from brancharchitect.elements.partition import Partition


def compute_geometric_intersection(
    set1_partitions: Dict[int, Partition],
    set2_partitions: Dict[int, Partition],
    encoding: Dict[str, int],
) -> Tuple[Set[int], Dict[int, Partition]]:
    """
    Compute the **Geometric Intersection** (All-Pairs Interaction).
    Synthesize new common substructures by intersecting every cluster.

    If P1 = {{A,B}}, P2 = {{B,C}}:
        P1 & P2 (Set Logic) -> {}
        geometric_intersection(P1, P2) -> {{B}}

    Args:
        set1_partitions: Mapping from bitmask to Partition for the first set.
        set2_partitions: Mapping from bitmask to Partition for the second set.
        encoding: The encoding dictionary to use for new partitions.

    Returns:
        Tuple[Set[int], Dict[int, Partition]]: A tuple containing the set of result bitmasks
        and the mapping from bitmask to Partition objects.
    """
    # Result set of bitmasks
    result_bitmasks: set[int] = set()
    result_partitions: dict[int, Partition] = {}

    # Optimized O(N*M) loop using bitmasks
    for (mask1, p1), (mask2, p2) in itertools.product(
        set1_partitions.items(), set2_partitions.items()
    ):
        intersection_mask = mask1 & mask2
        if intersection_mask != 0:
            if intersection_mask in result_bitmasks:
                continue

            # Determine which partition object to use for metadata/encoding
            # If intersection exactly matches one of the inputs, reuse that object
            if intersection_mask == mask1:
                new_p = p1
            elif intersection_mask == mask2:
                new_p = p2
            else:
                # Create new partition from intersection mask
                # Convert mask back to indices (requires log N per bit or naive scan)
                # Speed optimization: iterate indices of smaller partition
                smaller_p = p1 if len(p1.indices) < len(p2.indices) else p2
                other_mask = mask2 if smaller_p is p1 else mask1
                new_indices = tuple(
                    idx for idx in smaller_p.indices if (other_mask & (1 << idx))
                )
                new_p = Partition(new_indices, encoding)

            result_bitmasks.add(intersection_mask)
            result_partitions[intersection_mask] = new_p

    return result_bitmasks, result_partitions
