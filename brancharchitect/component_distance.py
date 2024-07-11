from brancharchitect.tree import Node
from typing import Tuple

def get_components_distances(tree1: Node, tree2: Node, components: Tuple) -> dict[tuple,int]:
    component_distance_map = {}
    for component in components:
        distance_one_two = _get_component_distance(reference_node=tree1, tree_to_check=tree2, component=component)
        distance_two_one = _get_component_distance(reference_node=tree2, tree_to_check=tree1, component=component)        
        component_distance_map[component] = (distance_one_two + distance_two_one)
    return component_distance_map

def _get_component_distance(reference_node: Node, tree_to_check: Node, component: list, counter : int=0) -> int:
    # Checking for proper subset condition
    for child in reference_node.children:
        if set(component) < set(child.split_indices):
            counter = _get_component_distance(child, tree_to_check, component, counter)
    # Check if the current split_indices exists in tree_to_check
    if not check_if_split_exists(tree_to_check, reference_node.split_indices):
        counter += 1  # Increment counter when split not found
    return counter  # Return the updated counter value

def _get_weighted_component_distance(reference_node: Node, tree_to_check: Node, component: list, weight : int = 0) -> int:
    # Checking for proper subset condition
    for child in reference_node.children:
        if set(component) < set(child.split_indices):
            weight = _get_weighted_component_distance(child, tree_to_check, component, weight)
    # Check if the current split_indices exists in tree_to_check
    if not check_if_split_exists(tree_to_check, reference_node.split_indices):
        weight += reference_node.length  # Increment counter when split not found
    return weight  # Return the updated counter value

def get_weighted_components_distances(tree1: Node, tree2: Node, components: Tuple) -> dict[tuple,int]:
    component_distance_map = {}
    for component in components:
        distance_one_two = _get_weighted_component_distance(reference_node=tree1, tree_to_check=tree2, component=component)
        distance_two_one = _get_weighted_component_distance(reference_node=tree2, tree_to_check=tree1, component=component)        
        component_distance_map[component] = (distance_one_two + distance_two_one)
    return component_distance_map


def check_if_split_exists(tree_to_check: Node, split_indices: list) -> bool: 
    # Convert split_indices to set once to avoid repeated conversions
    split_indices_set = set(split_indices)
    
    # Recursive function to check each node
    def check_node(node: Node):
        # Check current node's split indices
        if set(node.split_indices) == split_indices_set:
            return True
        # Recursively check each child
        for child in node.children:
            if check_node(child):  # If a child has the split, return True immediately
                return True
        return False  # Return False if no children match and current node doesn't match

    # Start the check from the root of tree_to_check
    return check_node(tree_to_check)