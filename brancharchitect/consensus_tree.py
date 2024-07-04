from typing import List, Dict, Tuple
from pprint import pprint
from brancharchitect.tree import Node
from brancharchitect.distances import collect_splits

def get_taxa_circular_order(node: Node):
    taxa_order : list = []    
    _get_taxa_circular_order(node, taxa_order)
    return taxa_order

def _get_taxa_circular_order(node: Node, taxa_order: list[str] = []):
    if not node.children:
        taxa_order.append(node.name)                    
    for node in node.children:
        _get_taxa_circular_order(node, taxa_order)

def incorporate_split_counts(split_list: List[Tuple[int]], number_of_splits: dict[Tuple[int],dict]):
    for split in split_list:
        if len(split) >  1:        
            if split in number_of_splits:
                number_of_splits[split]['count'] += 1
            else:
                number_of_splits[split] = {'count': 1}

def collect_count_of_splits(list_of_trees: List[Node]) -> Dict[tuple[int], Dict]:
    count_of_splits = {}    
    for tree in list_of_trees:        
        splits, split_lenghts = collect_splits(tree)            
        incorporate_split_counts(splits, count_of_splits)

    for split in count_of_splits.keys():        
        count_of_splits[split]['occurrence'] = count_of_splits[split]['count'] / len(list_of_trees)
    return count_of_splits

def sort_splits_by_occurrence(splits: Dict[Tuple[int], Dict[str, float]]) -> Tuple[List[Tuple[int]], List[float], List[float]]:
    """
    Sorts the splits by their number of occurrences in descending order and returns the sorted splits, occurrences, and counts.

    Args:
    - splits (Dict[Tuple[int], Dict[str, float]]): A dictionary where the key is a tuple representing a split,
      and the value is a dictionary containing information about the split, including 'occurrence' and 'count'.

    Returns:
    - Tuple[List[Tuple[int]], List[float], List[float]]: A tuple containing three lists:
      - List of splits (tuples) sorted by 'occurrence' in descending order.
      - List of corresponding 'occurrence' values sorted in descending order.
      - List of corresponding 'count' values sorted in descending order.
    """
    # Sort splits by their 'occurrence' value in descending order
    sorted_splits_with_counts = sorted(splits.items(), key=lambda item: item[1]['occurrence'], reverse=True)
    
    # Extract sorted splits, occurrences, and counts into separate lists
    sorted_splits = [split for split, _ in sorted_splits_with_counts]
    occurrences = [details['occurrence'] for _, details in sorted_splits_with_counts]
    counts = [details['count'] for _, details in sorted_splits_with_counts]
    
    return sorted_splits, occurrences, counts

def create_star_tree(taxon_order: List[int])-> Node:
    star_tree : Node = Node()    
    star_tree.split_indices = [i for i in range(len(taxon_order))]
    star_tree.name = 'R'
    for taxon in taxon_order:
        new_node = Node(name=taxon,split_indices=taxon_order.index(taxon), length=1)
        star_tree.children.append(new_node)                
    return star_tree

def filter_by_occurrence(splits: Dict[Tuple[int], Dict[str, float]], number_of_taxa: int, eq : float) -> List[Tuple[int]]:
        
    
    # List to store splits with occurrence over 50%    
    over_fifty_split = []
    
    # Filter splits with occurrence over 50%    
    for split in splits:
        
        if splits[split]['occurrence'] > eq and len(split) > 1 and number_of_taxa != len(split):
            
            over_fifty_split.append(split)
            
    # Sort the splits by the size of their tuples    
    over_fifty_split.sort(key=lambda x: len(x))    
    
    return over_fifty_split

def check_split_memory_compatibility(split_1: Tuple[int], tree_order: List[int], memory: List[int]):
    if memory:
        for memory_split in memory:
            if(check_split_compatibility(memory_split, split_1 ,tree_order)):                
                return False
    return True

def check_split_compatibility(split_1: List[int], split_2: List[int], tree_order: List[int]):
    x1 = set(split_1)
    x2 = set(split_1).difference(tree_order)
    
    y1 = set(split_2)
    y2 = set(split_2).difference(tree_order)

    x1y1 = x1.intersection(y1)
    x1y2 = x1.intersection(y2)
    
    y2x1 = y2.intersection(x1)
    y2x2 = y2.intersection(x2)

    if len(x1y2) == 0 and len(x1y1) == 0 and len(y2x1) == 0 and len(y2x2) == 0:
        return True
    else:
        return False

def filter_incompatible_splits(sorted_splits_by_occurrence: List[Tuple[int]],tree_order : List[int]):
    compatible_split_memory = []
    for split in sorted_splits_by_occurrence:
        if check_split_memory_compatibility(split, tree_order, compatible_split_memory):
            compatible_split_memory.append(split)
    return compatible_split_memory

def apply_split_in_tree(split: Tuple[int], node: Node) -> Node:
    split_set = set(map(int, split))  # Convert split to set of integers

    # Check if split_set is a subset of node.split_indices
    if split_set.issubset(set(node.split_indices)) and split_set != set(node.split_indices) or node.split_indices.issubset(set(split_set)) and split_set != set(node.split_indices):
        
        remaining_children, reassigned_children = [], []
        
        # Reassign children
        for child in node.children:
            
            if isinstance(child.split_indices, int):
                # Handle integer case                
                if child.split_indices in split_set:
                    reassigned_children.append(child)
                else:
                    remaining_children.append(child)
                    
            elif isinstance(child.split_indices, (list, set)):
                                
                # Handle list or set case
                if set(child.split_indices).issubset(split_set) or set(split_set).issubset(child.indices) :
                    reassigned_children.append(child)
                else:
                    remaining_children.append(child)

        if reassigned_children:            
            new_node = Node(name='', split_indices=split_set, children=reassigned_children)
            # Update original node's children and append new node
            node.children = remaining_children        
            node.children.append(new_node)

    # Recursively apply to children
    for child in node.children:
        if child.children:
            apply_split_in_tree(split, child)

def print_count_of_splits(count_of_splits: Dict[Tuple[int], Dict]):
    """
    Pretty-prints the count_of_splits dictionary.
    
    Args:
    - count_of_splits (Dict[Tuple[int], Dict]): The dictionary containing split counts and occurrences.
    """
    pprint(count_of_splits)