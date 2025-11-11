# Test for jump_path_component_to_pivot_edge
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.distances.component_distance import jump_path_component_to_pivot_edge


def test_jump_path_component_to_pivot_edge_basic():
    trees = parse_newick("(((A,B),(C,D)),E);")
    if not isinstance(trees, list):
        trees = [trees]
    tree = trees[0]
    # Get Partition for component ('A', 'B') and pivot_edge ('A', 'B', 'C', 'D')
    component = tree.names_to_partition(("A", "B"))
    pivot_edge_split = tree.names_to_partition(("A", "B", "C", "D"))
    path = jump_path_component_to_pivot_edge(tree, component, pivot_edge_split)
    # The path should go from the ('A', 'B') node up to the ('A', 'B', 'C', 'D') node
    assert path[0].split_indices == component
    assert path[-1].split_indices == pivot_edge_split
    # The path should be in order from component up to pivot_edge
    assert all(
        path[i].parent == path[i + 1] or path[i + 1] is None
        for i in range(len(path) - 1)
    )


def test_jump_path_component_to_pivot_edge_not_descendant():
    trees = parse_newick("(((A,B),(C,D)),E);")
    if not isinstance(trees, list):
        trees = [trees]
    tree = trees[0]
    # Get Partition for component ('E',) and pivot_edge ('A', 'B', 'C', 'D')
    component = tree.names_to_partition(("E",))
    pivot_edge_split = tree.names_to_partition(("A", "B", "C", "D"))
    path = jump_path_component_to_pivot_edge(tree, component, pivot_edge_split)
    # E is not a descendant of ('A', 'B', 'C', 'D'), so path should be empty
    assert path == []
