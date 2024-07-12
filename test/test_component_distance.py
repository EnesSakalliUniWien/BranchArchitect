from brancharchitect.newick_parser import parse_newick
from brancharchitect.component_distance import component_distance, jump_path_distance
from brancharchitect.distances import robinson_foulds_distance
from brancharchitect.jumping_taxa.bruteforce_algorithm import algorithm as brute_force_algo


def test_component_distance():
    trees = parse_newick("(((6:1,5:1),(1:1,(2:1,X:1):1):1):1,O:1);" +"(((6:1,(X:1,5:1)),(1:1,2:1):1),O:1);")
    components = [('X',)]    
    expected_distance = [4]

    observed_distance = component_distance(trees[0], trees[1], components)

    assert expected_distance == observed_distance


def test_two_component_distance():
    trees = parse_newick("(((6:1,5:1),(1:1,(2:1,X:1):1):1):1,O:1);" + "(((6:1,5:1),(1:1,(2:1,X:1):1):1):1,O:1);")
    components = [('X',)]    
    expected_distance = [0]

    observed_distance = component_distance(trees[0], trees[1], components)

    assert expected_distance == observed_distance


def test_cherry_is_hopping_component_distance():
    trees = parse_newick("(((6:1,5:1),((1:1,2:1):1,(X:1,Y:1):1):1):1,O:1);" + "(((6:1,5:1),((1:1,2:1):1,(X:1,Y:1):1):1):1,O:1);")
    components = [('X', 'Y')]    
    expected_distance = [0]

    observed_distance = component_distance(trees[0], trees[1], components)

    assert expected_distance == observed_distance

    
def test_taxa_switch_from_outgroup():
    trees = parse_newick("(((A:1,B:1),(C:1,D:1)),X:1);" + "(((X:1,B:1):1,(C:1,D:1):1),A:1);")
    components = [('X',)]    
    expected_distance = [2]

    observed_distance = component_distance(trees[0], trees[1], components)

    assert expected_distance == observed_distance


def test_weighted_component_distance():
    trees = parse_newick("(((6,5),(1,(2,X):2)),O);" +"(((6,(X,5)),(1,2)),O);")
    components = [('X',)]    
    expected_distance = [5]

    observed_distance = component_distance(trees[0], trees[1], components, weighted=True)

    assert expected_distance == observed_distance


def test_same_tree():
    trees = parse_newick("(((6:1,5:1),(1:1,(2:1,X:1):2):1):1,O:1);" +"(((6:1,5:1),(1:1,(2:1,X:1):2):1):1,O:1);")
    components = [('X',)]    
    expected_distance = [0]

    observed_distance = component_distance(trees[0], trees[1], components, weighted=True)

    assert expected_distance == observed_distance


def test_zero_edge_high_in_hierarchy():
    trees = parse_newick('(((1,2),(3,4)),5);' + '((2,(3,4)),(5,1));')
    components = [('4',), ('1',)]    
    expected_distance = [0, 3]

    observed_distance = component_distance(trees[0], trees[1], components)

    assert expected_distance == observed_distance


def test_jump_path_distance():
    trees = parse_newick("(((6:1,5:1),(1:1,(2:1,X:1):1):1):1,O:1);" +"(((6:1,(X:1,5:1)),(1:1,2:1):1),O:1);")
    components = [('X',)]    
    expected_distance = [4]

    observed_distance = jump_path_distance(trees[0], trees[1], components)

    assert expected_distance == observed_distance


def test_two_jump_path_distance():
    trees = parse_newick("(((6:1,5:1),(1:1,(2:1,X:1):1):1):1,O:1);" + "(((6:1,5:1),(1:1,(2:1,X:1):1):1):1,O:1);")
    components = [('X',)]    
    expected_distance = [0]

    observed_distance = jump_path_distance(trees[0], trees[1], components)

    assert expected_distance == observed_distance


def test_cherry_is_hopping_jump_path_distance():
    trees = parse_newick("(((6:1,5:1),((1:1,2:1):1,(X:1,Y:1):1):1):1,O:1);" + "(((6:1,5:1),((1:1,2:1):1,(X:1,Y:1):1):1):1,O:1);")
    components = [('X', 'Y')]    
    expected_distance = [0]

    observed_distance = jump_path_distance(trees[0], trees[1], components)

    assert expected_distance == observed_distance

    
def test_taxa_switch_from_outgroup_jump_path_distance():
    trees = parse_newick("(((A:1,B:1),(C:1,D:1)),X:1);" + "(((X:1,B:1):1,(C:1,D:1):1),A:1);")
    components = [('X',)]    
    expected_distance = [2]

    observed_distance = jump_path_distance(trees[0], trees[1], components)

    assert expected_distance == observed_distance


def test_weighted_jump_path_distance():
    trees = parse_newick("(((6,5),(1,(2,X):2)),O);" +"(((6,(X,5)),(1,2)),O);")
    components = [('X',)]    
    expected_distance = [5]

    observed_distance = jump_path_distance(trees[0], trees[1], components, weighted=True)

    assert expected_distance == observed_distance


def test_same_tree_jump_path_distance():
    trees = parse_newick("(((6:1,5:1),(1:1,(2:1,X:1):2):1):1,O:1);" +"(((6:1,5:1),(1:1,(2:1,X:1):2):1):1,O:1);")
    components = [('X',)]    
    expected_distance = [0]

    observed_distance = jump_path_distance(trees[0], trees[1], components, weighted=True)

    assert expected_distance == observed_distance


def test_zero_edge_high_in_hierarchy_jump_path_distance():
    trees = parse_newick('(((1,2),(3,4)),5);' + '((2,(3,4)),(5,1));')
    components = [('4',), ('1',)]    
    expected_distance = [0, 3]

    observed_distance = jump_path_distance(trees[0], trees[1], components)

    assert expected_distance == observed_distance
