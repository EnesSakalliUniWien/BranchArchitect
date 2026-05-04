from brancharchitect.elements.partition import Partition
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree_interpolation.subtree_paths.execution.pivot_edge_interpolation_frame_builder import (
    build_frames_for_subtree,
)


def test_reorder_frames_track_all_movers_shifted_by_global_reorder():
    source = parse_newick("(A:1,M1:1,M2:1,B:1);")
    destination = parse_newick("(A:1,B:1,M1:1,M2:1);")
    destination.initialize_split_indices(source.taxa_encoding)

    encoding = source.taxa_encoding
    pivot_edge = Partition(tuple(sorted(encoding.values())), encoding)
    mover = Partition((encoding["M1"],), encoding)
    sibling = Partition((encoding["M2"],), encoding)

    trees, _edges, _final_tree, subtree_tracker = build_frames_for_subtree(
        interpolation_state=source,
        destination_tree=destination,
        current_pivot_edge=pivot_edge,
        selection={
            "subtree": mover,
            "collapse": {"path_segment": []},
            "expand": {"path_segment": []},
        },
        all_mover_partitions=[mover, sibling],
        collapse_sibling_groups={mover: [mover, sibling]},
        expand_sibling_groups={mover: [mover, sibling]},
    )

    assert len(trees) >= 2
    assert list(trees[1].get_current_order()) == ["A", "B", "M1", "M2"]
    assert all(group == [mover, sibling] for group in subtree_tracker)
