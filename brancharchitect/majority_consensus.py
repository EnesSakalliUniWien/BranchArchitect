from typing import List
from brancharchitect.tree import Node
from brancharchitect.newick_parser import parse_newick
from brancharchitect.consensus_tree import (create_star_tree, 
                                            apply_split_in_tree, 
                                            get_taxa_circular_order, 
                                            collect_count_of_splits, 
                                            filter_by_occurrence
                                        )


def _create_majority_consensus_tree(tree_order: List[int], number_of_splits: List[int], eq: float = 0.50) -> Node:
    star_tree = create_star_tree(tree_order)    
    over_fifty_splits = filter_by_occurrence(number_of_splits, len(tree_order), eq)    
    for split in over_fifty_splits:
        apply_split_in_tree(split, star_tree)
    return star_tree


def create_majority_consensus_tree(newick_tree_list : str, eq: float = 0.50) -> Node:
            
    trees = parse_newick(newick_tree_list)    
    order_of_first_tree = get_taxa_circular_order(trees[0])
    observed_number_of_splits = collect_count_of_splits(trees)    
    tree = _create_majority_consensus_tree(tree_order=order_of_first_tree, number_of_splits=observed_number_of_splits)
    return tree
