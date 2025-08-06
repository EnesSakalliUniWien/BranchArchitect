import pytest
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.jumping_taxa.lattice.lattice_solver import lattice_algorithm
from brancharchitect.jumping_taxa.lattice.mapping import (
    map_s_edges_to_original_by_index,
)


def test_map_s_edges_to_original_by_index():
    # Example trees from notebook
    tree1 = "(O1:1,(O2:1,((((C1:1,X:1)44:1,B1:1)45:1,C2:1)39:1,(D1:1,((D2:1,A1:1)75:1,A2:1)29:1)20:1)31:1)10:1,B2:1);"
    tree2 = "(O1:1,O2:1,(((((C1:1,X:1)39:1,(B1:1,B2:1)6:1)21:1,C2:1)28:1,D1:1)23:1,((D2:1,A1:1)57:1,A2:1)42:1)53:1);"
    trees = parse_newick(tree1 + tree2)
    current_t1, current_t2 = trees[0], trees[1]
    input_tree1, input_tree2 = trees[0], trees[1]

    # Run lattice algorithm for current iteration
    solution_sets_this_iter, s_edges_this_iter_unmapped = lattice_algorithm(
        current_t1, current_t2, None
    )

    # Map s-edges to original
    mapped = map_s_edges_to_original_by_index(
        s_edges_this_iter_unmapped, input_tree1, input_tree2, current_t1, current_t2, 0
    )

    # Print diagnostics
    print("S-edges from current iteration:")
    for s_edge in s_edges_this_iter_unmapped:
        print(f"S-edge: {s_edge}, Indices: {getattr(s_edge, 'indices', None)}")

    print("Mapping results:")
    for i, (s_edge, original) in enumerate(zip(s_edges_this_iter_unmapped, mapped)):
        print(
            f"{i + 1}. S-edge: {getattr(s_edge, 'indices', None)} -> Original: {getattr(original, 'indices', None) if original else None}"
        )

    # Assert that at least one mapping is successful
    assert any(original is not None for original in mapped), (
        "No s-edges were mapped to original partitions"
    )
