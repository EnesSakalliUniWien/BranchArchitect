from brancharchitect.newick_parser import parse_newick
from brancharchitect.distances import collect_splits, sort_splits
from brancharchitect.distances import calculate_relative_tree_distance, calculate_relative_distances, calculate_weighted_distance, calculate_weighted_distances

def test_collect_splits():
    s = ('(A:1,(B:1,C:1):1);' + '(B:1,(A:1,C:1):1);')
    
    t1, t2  = parse_newick(s)
    
    expected_list_split_list_one = [tuple([0,1,2]), tuple([1,2]), tuple([0]), tuple([1]), tuple([2])] 
    expected_list_split_list_two = [tuple([1,0,2]), tuple([0,2]), tuple([1]), tuple([0]), tuple([2])] 
    
    # Sort the expected lists
    expected_list_split_list_one = sort_splits(expected_list_split_list_one)
    expected_list_split_list_two = sort_splits(expected_list_split_list_two)

    # Collect splits and sort the observed lists
    observed_split_list_one, split_lengths_one = collect_splits(t1)
    observed_split_list_two, split_lengths_two = collect_splits(t2)

    observed_split_list_one = sort_splits(observed_split_list_one)
    observed_split_list_two = sort_splits(observed_split_list_two)

    assert expected_list_split_list_one == observed_split_list_one    
    assert expected_list_split_list_two == observed_split_list_two 

def test_collected_splits():
    s = ('(A:1,(B:1,C:1):1);' + '(B:1,(A:1,C:1):1);')
    
    t1, t2  = parse_newick(s)
    
    expected_list_split_list_one = [tuple([0,1,2]), tuple([1,2]), tuple([0]), tuple([1]), tuple([2])] 
    expected_list_split_list_two = [tuple([1,0,2]), tuple([0,2]), tuple([1]), tuple([0]), tuple([2])] 
    
    # Sort the expected lists
    expected_list_split_list_one = sort_splits(expected_list_split_list_one)
    expected_list_split_list_two = sort_splits(expected_list_split_list_two)

    # Collect splits and sort the observed lists
    observed_split_list_one, split_lengths_one = collect_splits(t1)
    observed_split_list_two, split_lengths_two = collect_splits(t2)
    
    observed_split_list_one = sort_splits(observed_split_list_one)
    observed_split_list_two = sort_splits(observed_split_list_two)

    assert expected_list_split_list_one == observed_split_list_one    
    assert expected_list_split_list_two == observed_split_list_two 

def test_distances_1():
    s = ('(A:1,(B:1,C:1):1);' + '(B:1,(A:1,C:1):1);')
    t1, t2  = parse_newick(s)
    expected_distances = 1 
    relative_distance = calculate_relative_tree_distance(t1, t2)
    assert relative_distance == expected_distances

def test_distances_2():
    s = ('((A:1,B:1):1,(C:1,D:1):1);' + '((C:1,A:1):1,(B:1,D:1):1);')
    t1, t2  = parse_newick(s)
    expected_distances = 2
    relative_distance = calculate_relative_tree_distance(t1, t2)
    assert relative_distance == expected_distances

def test_distances_3():
    s = ('((A:1,B:1):1,(C:1,D:1):1);' + '((A:1,B:1):1,(C:1,D:1):1);')
    t1, t2  = parse_newick(s)
    expected_distances = 0
    relative_distance = calculate_relative_tree_distance(t1, t2)
    assert relative_distance == expected_distances

def test_trajectory_distances_test_one():
    s = ('((A:1,B:1):1,(C:1,D:1):1);' + '((A:1,B:1):1,(C:1,D:1):1);' + '((A:1,B:1):1,(C:1,D:1):1);')
    trees  = parse_newick(s)
    observed_distances = calculate_relative_distances(trees)
    expected_distances = [0,0]     
    assert observed_distances == expected_distances

def test_trajectory_distances_test_two():
    s = ('((A:1,B:1):1,(C:1,D:1):1);' + '((C:1,A:1):1,(B:1,D:1):1);' + '((C:1,A:1):1,(B:1,D:1):1);')
    trees  = parse_newick(s)
    expected_distances = [2,0]
    observed_relative_distances = calculate_relative_distances(trees)
    assert expected_distances == observed_relative_distances

def test_trajectory_distances_test_three():
    s = ('((A:1,B:1):1,(C:1,D:1):1);' + '((C:1,A:1):1,(B:1,D:1):1);' + '((A:1,B:1):1,(C:1,D:1):1);')
    trees  = parse_newick(s)
    expected_distances = [2,2]
    observed_relative_distances = calculate_relative_distances(trees)
    assert expected_distances == observed_relative_distances

def test_pair_weights_distances_test_one():
    s = ('((A:1,B:1):1,(C:1,D:1):1);' + '((A:1,B:1):1,(C:1,D:1):1);')
    trees  = parse_newick(s)    
    observed_distance = calculate_weighted_distance(trees[0], trees[1])    
    assert observed_distance == 0

def test_pair_weights_distances_test_one():
    s = ('((A:1,B:1):1,(C:1,D:1):1);' + '((A:1,B:1):1,(C:1,D:1):1);')
    trees  = parse_newick(s)    
    observed_distance = calculate_weighted_distance(trees[0], trees[1])    
    assert observed_distance == 0

def test_trajectory_distances_test_two():
    s = ('((A:1,B:1):1,(C:1,D:1):1);' + '((C:1,A:1):1,(B:1,D:1):1);' + '((C:1,A:1):1,(B:1,D:1):1);')
    trees  = parse_newick(s)
    expected_distances = [4,0]
    observed_relative_distances = calculate_weighted_distances(trees)
    
    assert expected_distances == observed_relative_distances
