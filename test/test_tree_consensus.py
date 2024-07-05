from brancharchitect.consensus_tree import *
from brancharchitect.newick_parser import parse_newick
from brancharchitect.consensus_tree import get_taxa_circular_order
from brancharchitect.majority_consensus_extended import create_majority_consensus_tree_extended
from brancharchitect.majority_consensus import create_majority_consensus_tree

def test_check_split_compatibility():
    assert check_split_compatibility(set([1,2]), set([3,4]), tree_order=[1,2,3,4])
    assert check_split_compatibility(set([1]), set([3,4]), tree_order=[1,2,3,4])


    assert check_split_compatibility(set([2,4]), set([1,3]), tree_order=[1,2,3,4])    
    assert check_split_compatibility(set([1]), set([2,3,4]), tree_order=[1,2,3,4])


    assert not check_split_compatibility(set([2]), set([2,3,4]), tree_order=[1,2,3,4])
    assert not check_split_compatibility(set([3]), set([2,3,4]), tree_order=[1,2,3,4])
    assert not check_split_compatibility(set([4]), set([2,3,4]), tree_order=[1,2,3,4])    
    assert not check_split_compatibility(set([2,3]), set([3,4]), tree_order=[1,2,3,4])
    assert not check_split_compatibility(set([1,3]), set([3,4]), tree_order=[1,2,3,4])


def test_single_split_in_dict():
    split_list = [(1, 2)]
    number_of_splits = {(1, 2): {'count': 3}}
    expected_result = {(1, 2): {'count': 4}}
    incorporate_split_counts(split_list, number_of_splits)
    assert number_of_splits == expected_result

    split_list = [(3, 4)]
    number_of_splits = {(1, 2): {'count': 3}}
    expected_result = {(1, 2): {'count': 3}, (3, 4): {'count': 1}}
    incorporate_split_counts(split_list, number_of_splits)
    assert number_of_splits == expected_result

    split_list = [(1, 2), (2, 3), (1, 2)]
    number_of_splits = {(1, 2): {'count': 3}, (2, 3): {'count': 2}}
    expected_result = {(1, 2): {'count': 5}, (2, 3): {'count': 3}}
    incorporate_split_counts(split_list, number_of_splits)
    assert number_of_splits == expected_result


def test_sort_splits_by_occurrence_empty_dict():
    splits = {}
    expected_result = ([], [], [])
    assert sort_splits_by_occurrence(splits) == expected_result


def test_sort_splits_by_occurrence_multiple_splits():
    splits = {
        (1, 2): {'occurrence': 0.7, 'count': 10},
        (3, 4): {'occurrence': 0.5, 'count': 5},
        (1, 2, 3): {'occurrence': 0.8, 'count': 15},
        (4, 5): {'occurrence': 0.6, 'count': 8}
    }
    expected_result = ([(1, 2, 3), (1, 2), (4, 5), (3, 4)], [0.8, 0.7, 0.6, 0.5], [15, 10, 8, 5])
    assert sort_splits_by_occurrence(splits) == expected_result


def test_sort_splits_by_occurrence_same_occurrence():
    splits = {
        (1, 2): {'occurrence': 0.5, 'count': 10},
        (3, 4): {'occurrence': 0.5, 'count': 5},
        (1, 2, 3): {'occurrence': 0.5, 'count': 15},
        (4, 5): {'occurrence': 0.5, 'count': 8}
    }
    expected_result = ([(1, 2), (3, 4), (1, 2, 3), (4, 5)], [0.5, 0.5, 0.5, 0.5], [10, 5, 15, 8])
    assert sort_splits_by_occurrence(splits) == expected_result


def test_check_split_memory_compatibility_empty_memory():
    split_1 = (1, 2, 3)
    tree_order = [1, 2, 3, 4, 5]
    memory = []
    assert check_split_memory_compatibility(split_1, tree_order, memory) == True


def test_check_split_memory_compatibility_compatible_split():
    split_1 = (1, 2, 3)
    tree_order = [1, 2, 3, 4, 5]
    memory = [(1, 2), (3, 4)]
    assert check_split_memory_compatibility(split_1, tree_order, memory) == True


def test_check_split_memory_compatibility_incompatible_split():
    split_1 = (1, 5)
    tree_order = [1, 2, 3, 4, 5]
    memory = [(1, 2), (3, 4)]
    assert check_split_memory_compatibility(split_1, tree_order, memory) == False


def test_tree_order_tree_extended():
    s = (
        '((A:1,(B:1,C:1):1):1,(D:1,(E:1,F:1):1):1);' 
        '((A:1,(B:1,C:1):1):1,(D:1,(E:1,F:1):1):1);'         
    )
    trees = parse_newick(s)    
    taxa_order = get_taxa_circular_order(trees[0])        
    assert taxa_order == ['A', 'B', 'C', 'D', 'E', 'F']


def test_create_majority_consensus_tree_extended():
    s = (
        '((A,(B,C)),(D,(E,F)));' 
        + 
        '((A,(B,C)),(D,(E,F)));'
        +
        '(((A,B),C),((D,E),F));'
        +        
        '((A,(B,C)),(D,(E,F)));'
        +
        '((E,(C,D)),(A,(B,F)));'        
    )

    trees = parse_newick(s)    
    observed_number_of_splits = collect_count_of_splits(trees)    
    taxa_order = get_taxa_circular_order(trees[0])

    expected_number_of_splits = {
        (0, 1, 2, 3, 4, 5): {'count': 5, 'occurrence': 1.0},
        (0, 1, 2): {'count': 4, 'occurrence': 0.8},
        (0, 1): {'count': 1, 'occurrence': 0.2},        
        (1, 2): {'count': 3, 'occurrence': 0.6},
        (3, 4, 5): {'count': 4, 'occurrence': 0.8},
        (4, 5): {'count': 3, 'occurrence': 0.6},
        (2, 3, 4): {'count': 1, 'occurrence': 0.2},        
        (3, 4): {'count': 1, 'occurrence': 0.2},
        (2, 3): {'count': 1, 'occurrence': 0.2},
        (1, 5): {'count': 1, 'occurrence': 0.2},        
        (0, 1, 5): {'count': 1, 'occurrence': 0.2},                
    }
    
    assert expected_number_of_splits == observed_number_of_splits    
    tree = create_majority_consensus_tree_extended(tree_order=taxa_order, number_of_splits=observed_number_of_splits)
        
    assert expected_number_of_splits == observed_number_of_splits
    assert '(D:1,E:1,((B:1,C:1),A:1),F:1)R;' == tree.to_newick()


def test_create_majority_consensus_tree_extended():
    tree = create_majority_consensus_tree_extended(
        '((A,(B,C)),(D,(E,F)));' + 
        '((A,(B,C)),(D,(E,F)));' +
        '(((A,B),C),((D,E),F));' +        
        '((A,(B,C)),(D,(E,F)));' +
        '((E,(C,D)),(A,(B,F)));'        
    )
    
    assert '(D:1,E:1,((B:1,C:1),A:1),F:1)R;' == tree.to_newick()

def test_create_majority_consensus():
    
    tree = create_majority_consensus_tree(
        '((((A:1,B:1),C:1),D:1):1);'+         
        '(((A:1,B:1,C:1),D:1):1);'+  
        '(((A:1,B:1,D:1),C:1):1);'        
    )

    assert '(D:1,(A:1,B:1,C:1))R;' == tree.to_newick()

