"""
Generates solution-to-atom mappings for phylogenetic analysis.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node
from brancharchitect.jumping_taxa.lattice.mapping import (
    map_solution_elements_via_parent,
)


def generate_solution_mappings(
    solutions: Dict[Partition, List[Partition]],
    source: Node,
    destination: Node,
) -> Tuple[
    Dict[Partition, Dict[Partition, Partition]],
    Dict[Partition, Dict[Partition, Partition]],
]:
    """
    Generate per-edge solution-to-atom mappings for phylogenetic analysis.

    Uses parent relationships to determine where each solution element is attached
    in the source and destination trees.
    """
    mapping_one, mapping_two = map_solution_elements_via_parent(
        solutions,
        source,
        destination,
    )

    return mapping_one, mapping_two
