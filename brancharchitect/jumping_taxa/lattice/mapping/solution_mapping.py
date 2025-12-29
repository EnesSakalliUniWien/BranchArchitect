"""
Solution Mapping: maps pivot edge solutions from pruned to original trees.
"""

from typing import Dict, List

from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition
from brancharchitect.jumping_taxa.lattice.mapping.iterative_pivot_mappings import (
    map_iterative_pivot_edges_to_original,
)
from brancharchitect.logger import jt_logger


def map_solutions_to_original_trees(
    solutions_dict: Dict[Partition, List[Partition]],
    original_tree1: Node,
    original_tree2: Node,
) -> Dict[Partition, List[Partition]]:
    """
    Map pivot edges from pruned trees to their corresponding splits in original trees.

    Args:
        solutions_dict: Flat solution partitions keyed by pivot edges from pruned trees
        original_tree1: Original unpruned tree 1
        original_tree2: Original unpruned tree 2

    Returns:
        Flat solution partitions keyed by pivot edges mapped to original trees
    """
    if not jt_logger.disabled:
        jt_logger.info("[lattice] Mapping pivot edges to original trees...")

    pivot_edges_list = list(solutions_dict.keys())
    solutions_list = [solutions_dict[pivot] for pivot in pivot_edges_list]

    mapped_pivot_edges = map_iterative_pivot_edges_to_original(
        pivot_edges_list,
        original_tree1,
        original_tree2,
        solutions_list,
    )

    # Build new dictionary with mapped pivot edges
    mapped_solutions_dict = {
        mapped_pivot: solutions_dict[pivot_edge]
        for pivot_edge, mapped_pivot in zip(pivot_edges_list, mapped_pivot_edges)
    }

    if not jt_logger.disabled:
        jt_logger.info(
            f"[lattice] Mapped {len(pivot_edges_list)} pivot edges to original trees"
        )

    return mapped_solutions_dict
