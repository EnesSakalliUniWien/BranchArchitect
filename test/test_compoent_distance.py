from brancharchitect.newick_parser import parse_newick
from brancharchitect.component_distance import get_components_distances, get_weighted_components_distances
from brancharchitect.distances import robinson_foulds_distance

def test_component_distance():
    trees = parse_newick("(((6:1,5:1),(1:1,(2:1,X:1):1):1):1,O:1);" +"(((6:1,(X:1,5:1)),(1:1,2:1):1),O:1);")
    components = [(4,)]    
    expected_distance = {(4,): 4}
    observed_distance = get_components_distances(tree1=trees[0], tree2=trees[1], components=components)
    assert expected_distance == observed_distance
    
def test_two_component_distance():
    trees = parse_newick("(((6:1,5:1),(1:1,(2:1,X:1):1):1):1,O:1);" + "(((6:1,5:1),(1:1,(2:1,X:1):1):1):1,O:1);")
    components = [(4,)]    
    expected_distance = {(4,): 0}
    observed_distance = get_components_distances(tree1=trees[0], tree2=trees[1], components=components)
    assert expected_distance == observed_distance
    
def test_cherry_is_hopping_component_distance():
    trees = parse_newick("(((6:1,5:1),((1:1,2:1):1,(X:1,Y:1):1):1):1,O:1);" + "(((6:1,5:1),((1:1,2:1):1,(X:1,Y:1):1):1):1,O:1);")
    components = [(4,5)]    
    expected_distance = {(4,5): 0}
    observed_distance = get_components_distances(tree1=trees[0], tree2=trees[1], components=components)
    assert expected_distance == observed_distance
    
def test_taxa_switch_from_outgroup():
    trees = parse_newick("(((A:1,B:1),(C:1,D:1)),X:1);" + "(((X:1,B:1):1,(C:1,D:1):1),A:1);")
    components = [(3,)]    
    expected_distance = {(3,): 2}
    observed_distance = get_components_distances(tree1=trees[0], tree2=trees[1], components=components)
    assert expected_distance == observed_distance
    
def test_cherry_is_hopping_component():
    trees = parse_newick("(((6:1,5:1),(1:1,(2:1,(X:1,Y:1)):1):1):1,O:1);" +"(((6:1,((X:1,Y:1),5:1)),(1:1,2:1):1),O:1);")
    components = [(4,5), (2,3), (0,1)]    
    expected_distance = {
            (4,5): 4,
            (2,3): 1,
            (0,1): 1
        }
    
    rfd = robinson_foulds_distance(trees[0], trees[1])    
    observed_distances = get_components_distances(tree1=trees[0], tree2=trees[1], components=components)
    assert expected_distance == observed_distances
    assert rfd == 6/2

def test_relative_tree_component_distance():
    trees = parse_newick("(((6:1,5:1),(1:1,(2:1,X:1):2):1):1,O:1);" +"(((6:1,(X:1,5:1):1):1,(1:1,2:1):1),O:1);")
    components = [(4,)]    
    expected_distance = {(4,): 5}
    observed_distance = get_weighted_components_distances(tree1=trees[0], tree2=trees[1], components=components)
    assert expected_distance == observed_distance
