from typing import List
from brancharchitect.tree import Node
from brancharchitect.newick_parser import parse_newick
from brancharchitect.consensus_tree import sort_splits_by_occurrence, create_star_tree, filter_incompatible_splits, apply_split_in_tree, get_taxa_circular_order, collect_count_of_splits


def _create_majority_consensus_tree_extended(tree_order: List[int], number_of_splits: List[int]) -> Node:    
    indices_list = [i for i in range(len(tree_order))]        
    star_tree = create_star_tree(tree_order)    
        
    sorted_splits_by_occurrence, _, __ = sort_splits_by_occurrence(number_of_splits)     
    compatible_split_memory = filter_incompatible_splits(sorted_splits_by_occurrence, indices_list)        
    compatible_split_memory.sort(key=lambda x: len(x))    

    for split in compatible_split_memory:        
        if len(split) != len(tree_order):
            apply_split_in_tree(split, star_tree)  
    return star_tree

def create_majority_consensus_tree_extended(newick_tree_list : str):
    trees = parse_newick(newick_tree_list)    
    order_of_first_tree = get_taxa_circular_order(trees[0])
    observed_number_of_splits = collect_count_of_splits(trees)    
    tree = _create_majority_consensus_tree_extended(tree_order=order_of_first_tree, number_of_splits=observed_number_of_splits)
    return tree
