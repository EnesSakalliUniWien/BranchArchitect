"""
Generates solution-to-atom mappings for phylogenetic analysis.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree import Node
from brancharchitect.jumping_taxa.lattice.mapping import (
    map_solution_elements_to_minimum_covers,
)


def generate_solution_mappings(
    solutions: Dict[Partition, List[Partition]],
    target: Node,
    reference: Node,
) -> Tuple[
    Dict[Partition, Dict[Partition, Partition]],
    Dict[Partition, Dict[Partition, Partition]],
]:
    """
    Generate per-edge solution-to-atom mappings for phylogenetic analysis.
    """
    # Use UNIQUE splits between target and reference, not all splits.
    # This aligns with minimum_cover_mapping expectations and avoids
    # introducing shared atoms (e.g., common clades) into the cover.
    target_splits: PartitionSet[Partition] = target.to_splits()
    reference_splits: PartitionSet[Partition] = reference.to_splits()
    target_unique_splits: PartitionSet[Partition] = target_splits - reference_splits
    reference_unique_splits: PartitionSet[Partition] = reference_splits - target_splits

    mapping_one, mapping_two = map_solution_elements_to_minimum_covers(
        solutions,
        target_unique_splits,
        reference_unique_splits,
    )

    return mapping_one, mapping_two
