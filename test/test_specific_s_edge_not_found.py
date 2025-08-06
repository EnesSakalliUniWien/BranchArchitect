from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.jumping_taxa.lattice.lattice_solver import lattice_algorithm
from brancharchitect.jumping_taxa.lattice.mapping import (
    map_s_edges_to_original_by_index,
)


def test_specific_s_edge_not_found_debug():
    tree1_str = "(O1:1,(O2:1,((((C1:1,X:1)44:1,B1:1)45:1,C2:1)39:1,(D1:1,((D2:1,A1:1)75:1,A2:1)29:1)20:1)31:1)10:1,B2:1);"
    tree2_str = "(O1:1,O2:1,(((((C1:1,X:1)39:1,(B1:1,B2:1)6:1)21:1,C2:1)28:1,D1:1)23:1,((D2:1,A1:1)57:1,A2:1)42:1)53:1);"
    tree1 = parse_newick(tree1_str)
    tree2 = parse_newick(tree2_str)
    if isinstance(tree1, list):
        tree1 = tree1[0]
    if isinstance(tree2, list):
        tree2 = tree2[0]

    current_t1, current_t2 = tree1, tree2
    input_t1, input_t2 = tree1, tree2

    # Run lattice algorithm for current iteration
    solution_sets_this_iter, s_edges_this_iter_unmapped = lattice_algorithm(
        current_t1, current_t2, []
    )

    print("All s-edges produced in this iteration:")
    for s_edge in s_edges_this_iter_unmapped:
        indices = getattr(s_edge, "indices", [])
        taxa_names = (
            [current_t1.get_leaves()[i].name for i in indices]
            if hasattr(current_t1, "get_leaves")
            else []
        )
        print(f"Indices: {indices}, Taxa: {taxa_names}")

    # Print context leaf indices
    t1_leaf_indices = getattr(current_t1, "split_indices", [])
    t1_leaf_names = (
        [current_t1.get_leaves()[i].name for i in t1_leaf_indices]
        if hasattr(current_t1, "get_leaves")
        else []
    )
    print(f"Current t1 leaf indices: {t1_leaf_indices}, names: {t1_leaf_names}")

    # Print and collect common splits between trees
    common_splits = input_t1.to_splits() & input_t2.to_splits()
    print("Common splits between trees:")
    common_indices_set = set()
    for split in common_splits:
        indices = tuple(sorted(getattr(split, "indices", [])))
        taxa_names = (
            [current_t1.get_leaves()[i].name for i in indices]
            if hasattr(current_t1, "get_leaves")
            else []
        )
        print(f"Indices: {indices}, Taxa: {taxa_names}")
        common_indices_set.add(indices)

    # Filter s-edges to only those matching a common split
    filtered_s_edges = []
    for s_edge in s_edges_this_iter_unmapped:
        indices = tuple(sorted(getattr(s_edge, "indices", [])))
        if indices in common_indices_set:
            filtered_s_edges.append(s_edge)

    print(f"Filtered s-edges (matching common splits): {len(filtered_s_edges)}")
    for s_edge in filtered_s_edges:
        indices = getattr(s_edge, "indices", [])
        taxa_names = (
            [current_t1.get_leaves()[i].name for i in indices]
            if hasattr(current_t1, "get_leaves")
            else []
        )
        print(f"Indices: {indices}, Taxa: {taxa_names}")

    # Map only filtered s-edges to original
    mapped = map_s_edges_to_original_by_index(
        filtered_s_edges,
        input_t1,
        input_t2,
        current_t1,
        current_t2,
        2,  # Iteration number for diagnostic
    )

    # Find the s-edge with indices [2, 3, 4, 5, 6, 7, 8, 9]
    target_indices = tuple(sorted([2, 3, 4, 5, 6, 7, 8, 9]))
    found = False
    for s_edge, original in zip(filtered_s_edges, mapped):
        indices = tuple(sorted(getattr(s_edge, "indices", [])))
        taxa_names = (
            [current_t1.get_leaves()[i].name for i in indices]
            if hasattr(current_t1, "get_leaves")
            else []
        )
        if indices == target_indices:
            found = True
            print(f"S-edge found: {s_edge}, Indices: {indices}, Taxa: {taxa_names}")
            print(
                f"Mapped to original: {getattr(original, 'indices', None) if original else None}"
            )
            assert original is not None, (
                "S-edge was not mapped to any original partition!"
            )
    if not found:
        print(
            "Target s-edge with indices [2, 3, 4, 5, 6, 7, 8, 9] not found in filtered_s_edges."
        )
        print(f"All common split indices: {sorted(list(common_indices_set))}")
        assert False, "Target s-edge not found among common s-edges!"
