"""
Generates solution-to-atom mappings for phylogenetic analysis.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree import Node
from brancharchitect.jumping_taxa.lattice.mapping import map_solutions_to_atoms


def generate_solution_mappings(
    solutions: Dict[Partition, List[List[Partition]]],
    target: Node,
    reference: Node,
) -> Tuple[
    Dict[Partition, Dict[Partition, Partition]],
    Dict[Partition, Dict[Partition, Partition]],
]:
    """
    Generate per-edge solution-to-atom mappings for phylogenetic analysis.
    """
    target_unique_splits: PartitionSet[Partition] = target.to_splits()
    reference_unique_splits: PartitionSet[Partition] = reference.to_splits()

    mapping_one, mapping_two = map_solutions_to_atoms(
        solutions,
        target_unique_splits,
        reference_unique_splits,
    )

    return mapping_one, mapping_two
