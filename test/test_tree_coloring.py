# from phylomovie.services.coloring_algorithm import identification_algorithm_with_anti_s_edge_grafting
# from phylomovie.services.tree.Treere import Treere
from typing import Dict, List

import pytest
import math
import itertools
from os.path import join
import os.path

from brancharchitect.jumping_taxa import call_jumping_taxa
from brancharchitect.io import read_newick

# from phylomovie.services.coloring_algorithm.algorithm_5 import algorithm_5, algorithm1 as functional_alg1
DATA_ROOT = "test/data/"


def all_combinations(a, b):
    return [sorted(l) for l in itertools.product(a, b)]


test_data_list = [
    (join(DATA_ROOT, "simon_test_tree_6.tree"), "Subtree (X1, X2) and (X3, X4) exchange", [[("X1", "X2", "X3", "X4")]]),
    (join(DATA_ROOT, "simon_test_tree_7.tree"), "50% of taxa jumps", [[("B", "C"), ("A", "E"), ("A", "B"), ("A", "C"),("C","E")]]),
    (join(DATA_ROOT, "small-test-one-leave-moving.tree"), "one taxon jumps", [[["C"], ["E"], ["D"]]]),
    (join(DATA_ROOT, "small-test-two-leaves-moving.tree"), "Two taxa jump independently", [all_combinations(["D", "E", "F"], ["A", "B", "C"])]),
    (join(DATA_ROOT, "small-test-sub-tree-moving.tree"), "one taxon jumps", [[["D"], ["C"],["E","F"]]]),
    (join(DATA_ROOT, "small-test-one-leave-crossing-moving.tree"), "A and D exchange", [[["D", "A"]]]),
    (join(DATA_ROOT, "small-test-subtree-crossing-moving.tree"), "No topological change, only order change", [[[]]]),
    (join(DATA_ROOT, "small-change-in-subtree.tree"), "E jumps into G", [[["E"]]]),
    (join(DATA_ROOT, "heiko_5_test_tree.tree"), "C1/C2 jumps into A2, E1/E2 jumps into", [[["C1", "C2", "E1", "E2"]]]),
    (join(DATA_ROOT, "heiko_6_test_tree.tree"), "b1/b2 jumps to A2, E jumps into F1", [[["B1", "B2", "E", "A1"],["A3", "B1", "B2", "E"]]]),
    (join(DATA_ROOT, "test_tree_moving_downwards.tree"), "Subtree is moving one step downwards", [[["X1", "X2"]]]),
    (join(DATA_ROOT, "test_tree_moving_updwards.tree"), "Subtree is moving one step downwards", [[["X1", "X2"]]]),
    (join(DATA_ROOT, "test_tree_moving_downwards_easier.tree"), "Subtree is moving one step downwards", [[["X1", "X2"]]]),
    (join(DATA_ROOT, "test_tree_moving_downwards_easier2.tree"), "Subtree is moving one step downwards", [[["X1", "X2"]]]),
    (join(DATA_ROOT, "simon_test_tree_2.tree"), "two leaves jump independently but crossing; 1 jumps into 7, 5 into 4", [[["1", "5"]]]),
    (join(DATA_ROOT, "simon_test_tree_4.tree"), "two leaves jump independently but crossing; 1 jumps into 7, 5 into 4", [[["C", "D"]]]),
    (join(DATA_ROOT, "simon_test_tree_5.tree"), "two leaves jump independently but crossing; 1 jumps into 7, 5 into 4", [[["A", "E"],["A","B","C","D"]]]),
    (join(DATA_ROOT, "simon_test_tree_2_renamed.tree"), "two leaves jump independently but crossing; 1 jumps into 7, 5 into 4", [[["A", "E"]]]),
    (join(DATA_ROOT, "alltrees-112.tree"), "two leaves jump independently but crossing; 1 jumps into 7, 5 into 4", [[["C", "B2", "A2"],["A1","X"],["B1","B2"]]]),
    (join(DATA_ROOT, "simon_test_tree_3.tree"), "Leave 1 jumps into subtree and builds multifurication", [[["1"]]]),
    (join(DATA_ROOT, "multifurcation.tree"), "Cherry (b1, b2) disolves", [[["b1"], ["b2"],["b1","b2"]]]),
    (join(DATA_ROOT, "multifurcation2.tree"), "Cherry (b1, b2) disolves and cherry (b3, b4) disolves", [all_combinations(["b1", "b2"], ["b3", "b4"]),["b1","b2","b3","b4"],["b1", "b2", "b3", "b4"]]),
    (join(DATA_ROOT, "test_tree_moving_downwards_with_multifurcations.tree"), "Subtree is moving one step downwards", [[["X1", "X2"]]]),
    (join(DATA_ROOT, "hiv_index_2.tree"), "HIV Test", [[["A1AF004885", "BAY423387", "BAY970949"], ["A1AF004885", "BAY423387"]]]),
    (join(DATA_ROOT, "hiv_index_86.tree"), "HIV Test", [["A1AF069670", "A1AF484509", "A1U51190"]]),
    (join(DATA_ROOT, "hiv_index_86_half.tree"), "HIV Test", [[["X1", "X2"]]]),
    (join(DATA_ROOT, "rampage_tree.tree"), "HIV Test", [[["G"]]]),
    (join(DATA_ROOT, "fullNone1.tree"), "Full None", [[["3","5"],["4","6"]]]),
    (join(DATA_ROOT, "rampage_tree2.tree"), "HIV Test", [[["G"]]]),
    (join(DATA_ROOT, "rampage_tree3.tree"), "(G, F) jumps to root", [[["G", "F"]]]),
    (join(DATA_ROOT, "rampage_tree4.tree"), "(G) jumps into F", [[["G"], ["F"]]]),
    (join(DATA_ROOT, "alltrees-0-1.tree"), "All Trees Test", [[["G"]]]),
    (join(DATA_ROOT, "subtree_hiv_86.tree"), "HIV Test", [[["X1", "X2"]]]),
    (join(DATA_ROOT, "subtree_2_hiv_86.tree"), "HIV Test", [[["X1", "X2"]]]),
    (join(DATA_ROOT, "subtree_2_hiv_86.tree"), "HIV Test", [[["X1", "X2"]]]),
    (join(DATA_ROOT, "reverse_simon_test_tree_6.tree"), "REVERSE Subtree (X1, X2) and (X3, X4) exchange", [[["X1", "X2", "X3", "X4"]]]),
    (join(DATA_ROOT, "reverse_simon_test_tree_7.tree"), "REVERSE 50% of taxa jumps", [[["B", "C"], ["A", "E"], ["A", "B"], ["A", "C"],["C","E"]]]),
    (join(DATA_ROOT, "reverse_small-test-one-leave-moving.tree"), "REVERSE one taxon jumps", [[["C"], ["E"], ["D"]]]),
    (join(DATA_ROOT, "reverse_small-test-two-leaves-moving.tree"), "REVERSE Two taxa jump independently", [all_combinations(["D", "E", "F"], ["A", "B", "C"])]),
    (join(DATA_ROOT, "reverse_small-test-sub-tree-moving.tree"), "REVERSE one taxon jumps", [[["D"], ["C"],["E","F"]]]),
    (join(DATA_ROOT, "reverse_small-test-one-leave-crossing-moving.tree"), "REVERSE A and D exchange", [[["D", "A"]]]),
    (join(DATA_ROOT, "reverse_small-test-subtree-crossing-moving.tree"), "REVERSE No topological change, only order change", [[[]]]),
    (join(DATA_ROOT, "reverse_small-change-in-subtree.tree"), "REVERSE E jumps into G", [[["E"]]]),
    (join(DATA_ROOT, "reverse_heiko_5_test_tree.tree"), "REVERSE C1/C2 jumps into A2, E1/E2 jumps into", [[["C1", "C2", "E1", "E2"]]]),
    (join(DATA_ROOT, "reverse_heiko_6_test_tree.tree"), "REVERSE b1/b2 jumps to A2, E jumps into F1", [[["B1", "B2", "E", "A1"],["A3", "B1", "B2", "E"]]]),
    (join(DATA_ROOT, "reverse_test_tree_moving_downwards.tree"), "REVERSE Subtree is moving one step downwards", [[["X1", "X2"]]]),
    (join(DATA_ROOT, "reverse_test_tree_moving_updwards.tree"), "REVERSE Subtree is moving one step downwards", [[["X1", "X2"]]]),
    (join(DATA_ROOT, "reverse_test_tree_moving_downwards_easier.tree"), "REVERSE Subtree is moving one step downwards", [[["X1", "X2"]]]),
    (join(DATA_ROOT, "reverse_test_tree_moving_downwards_easier2.tree"), "REVERSE Subtree is moving one step downwards", [[["X1", "X2"]]]),
    (join(DATA_ROOT, "reverse_simon_test_tree_2.tree"), "REVERSE two leaves jump independently but crossing; 1 jumps into 7, 5 into 4", [[["1", "5"]]]),
    (join(DATA_ROOT, "reverse_simon_test_tree_4.tree"), "REVERSE two leaves jump independently but crossing; 1 jumps into 7, 5 into 4", [[["C", "D"]]]),
    (join(DATA_ROOT, "reverse_simon_test_tree_5.tree"), "REVERSE two leaves jump independently but crossing; 1 jumps into 7, 5 into 4", [[["A", "E"],["A","B","C","D"]]]),
    (join(DATA_ROOT, "reverse_simon_test_tree_2_renamed.tree"), "REVERSE two leaves jump independently but crossing; 1 jumps into 7, 5 into 4", [[["A", "E"]]]),
    (join(DATA_ROOT, "reverse_alltrees-112.tree"), "REVERSE two leaves jump independently but crossing; 1 jumps into 7, 5 into 4", [[["C", "B2", "A2"],["A1","X"],["B1","B2"]]]),
    (join(DATA_ROOT, "reverse_simon_test_tree_3.tree"), "REVERSE Leave 1 jumps into subtree and builds multifurication", [[["1"]]]),
    (join(DATA_ROOT, "reverse_multifurcation.tree"), "REVERSE Cherry (b1, b2) disolves", [[["b1"], ["b2"],["b1","b2"]]]),
    (join(DATA_ROOT, "reverse_multifurcation2.tree"), "REVERSE Cherry (b1, b2) disolves and cherry (b3, b4) disolves", [all_combinations(["b1", "b2"], ["b3", "b4"]),["b1","b2","b3","b4"],["b1", "b2", "b3", "b4"]]),
    (join(DATA_ROOT, "reverse_test_tree_moving_downwards_with_multifurcations.tree"), "REVERSE Subtree is moving one step downwards", [[["X1", "X2"]]]),
    (join(DATA_ROOT, "reverse_hiv_index_2.tree"), "REVERSE HIV Test", [[["A1AF004885", "BAY423387", "BAY970949"], ["A1AF004885", "BAY423387"]]]),
    (join(DATA_ROOT, "reverse_hiv_index_86.tree"), "REVERSE HIV Test", [["A1AF069670", "A1AF484509", "A1U51190"]]),
    (join(DATA_ROOT, "reverse_hiv_index_86_half.tree"), "REVERSE HIV Test", [[["X1", "X2"]]]),
    (join(DATA_ROOT, "reverse_rampage_tree.tree"), "REVERSE HIV Test", [[["G"]]]),
    (join(DATA_ROOT, "reverse_fullNone1.tree"), "REVERSE Full None", [[["3","5"],["4","6"]]]),
    (join(DATA_ROOT, "reverse_rampage_tree2.tree"), "REVERSE HIV Test", [[["G"]]]),
    (join(DATA_ROOT, "reverse_rampage_tree3.tree"), "REVERSE (G, F) jumps to root", [[["G", "F"]]]),
    (join(DATA_ROOT, "reverse_rampage_tree4.tree"), "REVERSE (G) jumps into F", [[["G"], ["F"]]]),
    (join(DATA_ROOT, "reverse_alltrees-0-1.tree"), "REVERSE All Trees Test", [[["G"]]]),
    (join(DATA_ROOT, "reverse_subtree_hiv_86.tree"), "REVERSE HIV Test", [[["X1", "X2"]]]),
    (join(DATA_ROOT, "reverse_subtree_2_hiv_86.tree"), "REVERSE HIV Test", [[["X1", "X2"]]]),
    (join(DATA_ROOT, "reverse_subtree_2_hiv_86.tree"), "REVERSE HIV Test", [[["X1", "X2"]]]),
]


def find_to_be_highlighted_leaves(
    json_consensus_tree_list: List,
    sorted_nodes: List,
    find_highlight_tax,
    file_name=None,
    newick_string_list: List = [],
) -> List[Dict]:
    highlights_every_tree_list = []

    for i in range(0, len(json_consensus_tree_list) - 5, 5):
        first_tree_index = math.floor(i / 5)
        second_tree_index = math.floor(i / 5) + 1

        pair_of_newick_string = [
            newick_string_list[first_tree_index],
            newick_string_list[second_tree_index],
        ]

        set_of_trees = json_consensus_tree_list[i : i + 6]

        highlights_every_tree_list.append(
            find_highlight_tax(
                set_of_trees, sorted_nodes, file_name, pair_of_newick_string
            )
        )

    return highlights_every_tree_list


def _test_highlighting_algorithm(
    file_name,
    description,
    find_to_be_highlighted_taxa_function,
    results,
    strict=False,
    output=True,
):
    with open(file_name) as f:
        txt = f.read()

    treeRe = Treere()
    tree_list = treeRe.input_manager(txt, file_name)

    newick_string = treeRe.newick_purification(txt)

    newick_string_list = newick_string.split("\n")

    to_be_highlighted_leaves = find_to_be_highlighted_leaves(
        tree_list,
        treeRe.sorted_nodes,
        find_to_be_highlighted_taxa_function,
        file_name,
        newick_string_list,
    )

    for j in range(len(results)):
        for i in range(len(results[j])):
            results[j][i] = sorted(results[j][i])

    for i in range(len(to_be_highlighted_leaves)):
        to_be_highlighted_leaves[i] = sorted(to_be_highlighted_leaves[i])

    worked = True
    for rs, tree_result in zip(results, to_be_highlighted_leaves):
        worked = worked and tree_result in rs

    if not worked:
        functionname = find_to_be_highlighted_taxa_function.__name__
        treename = file_name
        if output:
            print(
                f"Algorithm {functionname} produced incorrect results on tree {treename}:"
            )
            print(f"Expected: {results}")
            print(f"Got: {to_be_highlighted_leaves}")
    if strict:
        for rs, tree_result in zip(results, to_be_highlighted_leaves):
            assert worked and tree_result in rs

    return worked, to_be_highlighted_leaves

    # assert to_be_highlighted_leaves == results,  ("Expected Results %", "".join(results))


@pytest.fixture(params=test_data_list, ids=[test_data[1] for test_data in test_data_list])
def test_data(request):
    return request.param

@pytest.mark.timeout(5)
def test_algorithm(test_data):
    filepath, description, result = test_data

    assert os.path.exists(filepath)

    tree1, tree2 = read_newick(filepath)

    candidates = call_jumping_taxa(tree1, tree2)

    assert candidates in result[0]
