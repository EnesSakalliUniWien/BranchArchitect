from brancharchitect.elements.partition import Partition
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree_interpolation.subtree_paths.execution.layout.mover_ordering import (
    build_destination_mover_order_key,
)
from brancharchitect.tree_interpolation.subtree_paths.execution.phases.subtree_microsteps import (
    build_subtree_interpolation_frames,
)
from brancharchitect.tree_interpolation.subtree_paths.planning import PivotTransitionStep


def test_reorder_highlight_group_does_not_make_all_group_members_move():
    source = parse_newick("(A:1,M1:1,M2:1,B:1);")
    destination = parse_newick("(A:1,B:1,M1:1,M2:1);")
    destination.initialize_split_indices(source.taxa_encoding)

    encoding = source.taxa_encoding
    pivot_edge = Partition(tuple(sorted(encoding.values())), encoding)
    mover = Partition((encoding["M1"],), encoding)
    sibling = Partition((encoding["M2"],), encoding)

    trees, _edges, _final_tree, subtree_highlights = build_subtree_interpolation_frames(
        interpolation_state=source,
        destination_tree=destination,
        current_pivot_edge=pivot_edge,
        selection=PivotTransitionStep(
            subtree=mover,
            collapse_path=(),
            expand_path=(),
        ),
        all_mover_partitions=[mover, sibling],
        collapse_sibling_groups={mover: [mover, sibling]},
        expand_sibling_groups={mover: [mover, sibling]},
    )

    assert len(trees) >= 2
    assert list(trees[1].get_current_order()) == ["A", "M2", "B", "M1"]
    assert all(group == [mover, sibling] for group in subtree_highlights)


def test_reorder_highlight_group_excludes_unrelated_movers():
    source = parse_newick("(A:1,M1:1,M2:1,X:1,B:1);")
    destination = parse_newick("(A:1,B:1,M1:1,M2:1,X:1);")
    destination.initialize_split_indices(source.taxa_encoding)

    encoding = source.taxa_encoding
    pivot_edge = Partition(tuple(sorted(encoding.values())), encoding)
    mover = Partition((encoding["M1"],), encoding)
    sibling = Partition((encoding["M2"],), encoding)
    unrelated = Partition((encoding["X"],), encoding)

    trees, _edges, _final_tree, subtree_highlights = build_subtree_interpolation_frames(
        interpolation_state=source,
        destination_tree=destination,
        current_pivot_edge=pivot_edge,
        selection=PivotTransitionStep(
            subtree=mover,
            collapse_path=(),
            expand_path=(),
        ),
        all_mover_partitions=[mover, sibling, unrelated],
        collapse_sibling_groups={mover: [mover, sibling]},
        expand_sibling_groups={mover: [mover, sibling]},
    )

    assert len(trees) >= 2
    assert all(group == [mover, sibling] for group in subtree_highlights)


def test_destination_mover_order_key_follows_destination_pivot_order():
    source = parse_newick("(A:1,B:1,C:1,D:1);")
    destination = parse_newick("(D:1,C:1,B:1,A:1);", encoding=source.taxa_encoding)
    destination.reorder_taxa(["D", "C", "B", "A"])

    encoding = source.taxa_encoding
    pivot_edge = Partition(tuple(sorted(encoding.values())), encoding)
    subtree_a = Partition((encoding["A"],), encoding)
    subtree_d = Partition((encoding["D"],), encoding)

    order_key = build_destination_mover_order_key(
        destination,
        pivot_edge,
        {subtree_a, subtree_d},
    )

    assert order_key[subtree_d] < order_key[subtree_a]
