from pathlib import Path
from typing import get_type_hints

import pytest

from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.jumping_taxa.lattice.frontiers.child_frontiers import (
    compute_child_frontiers,
)
from brancharchitect.jumping_taxa.lattice.matrices.conflict_matrix_builder import (
    _nesting_solution_rank_key,
    build_conflict_matrix,
)
from brancharchitect.jumping_taxa.lattice.matrices.meet_product_solvers import (
    union_split_matrix_results,
)
from brancharchitect.jumping_taxa.lattice.solvers.lattice_solver import LatticeSolver
from brancharchitect.jumping_taxa.lattice.types.child_frontiers import ChildFrontiers
from brancharchitect.jumping_taxa.lattice.types.pivot_edge_subproblem import (
    PivotEdgeSubproblem,
)
from brancharchitect.jumping_taxa.lattice.types.registry import SolutionRegistry
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree import Node
from brancharchitect.tree_interpolation.pair_interpolation import (
    process_tree_pair_interpolation,
)


def _ps(*parts: Partition, encoding: dict[str, int]) -> PartitionSet[Partition]:
    return PartitionSet(set(parts), encoding=encoding)


def _assert_child_frontiers_valid(
    child_frontiers_by_split: dict[Partition, ChildFrontiers],
) -> None:
    for child_frontiers in child_frontiers_by_split.values():
        expected_frontiers = {
            partition.bitmask
            for partition in child_frontiers.shared_top_splits.maximal_elements()
        }
        actual_frontiers = {
            partition.bitmask for partition in child_frontiers.shared_top_splits
        }
        assert actual_frontiers == expected_frontiers

        for bottom, frontiers in child_frontiers.bottom_partition_map.items():
            for frontier in frontiers:
                assert frontier in child_frontiers.shared_top_splits
                assert (frontier.bitmask & ~bottom.bitmask) == 0


def test_handle_pivot_solutions_accepts_one_candidate_before_requeueing():
    """
    Candidate solution sets from one solve pass are alternatives. The solver should
    accept one best candidate, remove only that candidate from covers, and requeue
    the pivot when residual frontier conflicts remain.
    """
    encoding = {"A": 0, "B": 1, "C": 2}
    pivot = Partition((0, 1, 2), encoding=encoding)
    a = Partition((0,), encoding=encoding)
    b = Partition((1,), encoding=encoding)

    child_frontiers = {
        pivot: ChildFrontiers(
            shared_top_splits=_ps(a, b, encoding=encoding),
            bottom_partition_map={
                a: _ps(a, encoding=encoding),
                b: _ps(b, encoding=encoding),
            },
        )
    }
    subproblem = PivotEdgeSubproblem(
        pivot_split=pivot,
        tree1_node=Node(name="t1", split_indices=pivot, taxa_encoding=encoding),
        tree2_node=Node(name="t2", split_indices=pivot, taxa_encoding=encoding),
        tree1_child_frontiers=child_frontiers,
        tree2_child_frontiers={
            pivot: ChildFrontiers(
                shared_top_splits=_ps(a, b, encoding=encoding),
                bottom_partition_map={
                    a: _ps(a, encoding=encoding),
                    b: _ps(b, encoding=encoding),
                },
            )
        },
        child_subtree_splits_across_trees=_ps(a, b, encoding=encoding),
        encoding=encoding,
    )
    _assert_child_frontiers_valid(subproblem.tree1_child_frontiers)
    _assert_child_frontiers_valid(subproblem.tree2_child_frontiers)
    subproblem.visits = 1

    solver = object.__new__(LatticeSolver)
    solver.registry = SolutionRegistry()
    solver.processing_stack = []

    solver._handle_pivot_solutions(
        subproblem,
        [
            _ps(a, encoding=encoding),
            _ps(b, encoding=encoding),
        ],
    )

    selected = solver.registry.select_best_solutions()

    assert selected[pivot] == [a]
    assert solver.processing_stack == [subproblem]
    assert b in subproblem.tree1_child_frontiers[pivot].shared_top_splits
    assert b in subproblem.tree2_child_frontiers[pivot].shared_top_splits


def test_handle_pivot_solutions_rejects_pivot_split_candidate():
    """
    The pivot split is the local root of the subproblem, not a valid jumping
    subtree. It must not be selected even when it would clear all covers.
    """
    encoding = {"A": 0, "B": 1}
    pivot = Partition((0, 1), encoding=encoding)
    a = Partition((0,), encoding=encoding)
    b = Partition((1,), encoding=encoding)

    child_frontiers = {
        pivot: ChildFrontiers(
            shared_top_splits=_ps(a, b, encoding=encoding),
            bottom_partition_map={
                a: _ps(a, encoding=encoding),
                b: _ps(b, encoding=encoding),
            },
        )
    }
    subproblem = PivotEdgeSubproblem(
        pivot_split=pivot,
        tree1_node=Node(name="t1", split_indices=pivot, taxa_encoding=encoding),
        tree2_node=Node(name="t2", split_indices=pivot, taxa_encoding=encoding),
        tree1_child_frontiers=child_frontiers,
        tree2_child_frontiers={
            pivot: ChildFrontiers(
                shared_top_splits=_ps(a, b, encoding=encoding),
                bottom_partition_map={
                    a: _ps(a, encoding=encoding),
                    b: _ps(b, encoding=encoding),
                },
            )
        },
        child_subtree_splits_across_trees=_ps(a, b, encoding=encoding),
        encoding=encoding,
    )
    _assert_child_frontiers_valid(subproblem.tree1_child_frontiers)
    _assert_child_frontiers_valid(subproblem.tree2_child_frontiers)
    subproblem.visits = 1

    solver = object.__new__(LatticeSolver)
    solver.registry = SolutionRegistry()
    solver.processing_stack = []

    solver._handle_pivot_solutions(
        subproblem,
        [
            _ps(a, encoding=encoding),
            _ps(pivot, encoding=encoding),
        ],
    )

    selected = solver.registry.select_best_solutions()

    assert selected[pivot] == [a]
    assert solver.processing_stack == [subproblem]
    assert b in subproblem.tree1_child_frontiers[pivot].shared_top_splits
    assert b in subproblem.tree2_child_frontiers[pivot].shared_top_splits


def test_handle_pivot_solutions_rejects_current_tree_root_candidate():
    """A lattice witness is not a mover if it is the current tree root."""
    tree1 = parse_newick("((A,B),C);")
    tree2 = parse_newick("((A,B),C);", encoding=tree1.taxa_encoding)
    encoding = tree1.taxa_encoding
    root = Partition((0, 1, 2), encoding=encoding)
    pivot = Partition((0, 1), encoding=encoding)
    a = Partition((0,), encoding=encoding)
    b = Partition((1,), encoding=encoding)

    subproblem = PivotEdgeSubproblem(
        pivot_split=pivot,
        tree1_node=tree1.find_node_by_split(pivot),
        tree2_node=tree2.find_node_by_split(pivot),
        tree1_child_frontiers={
            pivot: ChildFrontiers(
                shared_top_splits=_ps(a, b, encoding=encoding),
                bottom_partition_map={
                    a: _ps(a, encoding=encoding),
                    b: _ps(b, encoding=encoding),
                },
            )
        },
        tree2_child_frontiers={
            pivot: ChildFrontiers(
                shared_top_splits=_ps(a, b, encoding=encoding),
                bottom_partition_map={
                    a: _ps(a, encoding=encoding),
                    b: _ps(b, encoding=encoding),
                },
            )
        },
        child_subtree_splits_across_trees=_ps(a, b, encoding=encoding),
        encoding=encoding,
    )
    subproblem.visits = 1

    solver = object.__new__(LatticeSolver)
    solver.current_t1 = tree1
    solver.current_t2 = tree2
    solver.registry = SolutionRegistry()
    solver.processing_stack = []

    solver._handle_pivot_solutions(
        subproblem,
        [
            _ps(root, encoding=encoding),
            _ps(a, b, encoding=encoding),
        ],
    )

    selected = solver.registry.select_best_solutions()

    assert selected[pivot] == [a, b]


def test_completion_sequence_rejects_pivot_only_candidates():
    """Recursive completion cannot use the pivot split as a residual witness."""
    encoding = {"A": 0, "B": 1}
    pivot = Partition((0, 1), encoding=encoding)
    a = Partition((0,), encoding=encoding)
    b = Partition((1,), encoding=encoding)

    subproblem = PivotEdgeSubproblem(
        pivot_split=pivot,
        tree1_node=Node(name="t1", split_indices=pivot, taxa_encoding=encoding),
        tree2_node=Node(name="t2", split_indices=pivot, taxa_encoding=encoding),
        tree1_child_frontiers={
            pivot: ChildFrontiers(
                shared_top_splits=_ps(a, b, encoding=encoding),
                bottom_partition_map={
                    a: _ps(a, encoding=encoding),
                    b: _ps(b, encoding=encoding),
                },
            )
        },
        tree2_child_frontiers={
            pivot: ChildFrontiers(
                shared_top_splits=_ps(a, b, encoding=encoding),
                bottom_partition_map={
                    a: _ps(a, encoding=encoding),
                    b: _ps(b, encoding=encoding),
                },
            )
        },
        child_subtree_splits_across_trees=_ps(a, b, encoding=encoding),
        encoding=encoding,
    )
    _assert_child_frontiers_valid(subproblem.tree1_child_frontiers)
    _assert_child_frontiers_valid(subproblem.tree2_child_frontiers)

    solver = object.__new__(LatticeSolver)

    assert (
        solver._best_completion_sequence_for_pivot(
            subproblem, [_ps(pivot, encoding=encoding)]
        )
        == []
    )


def test_union_split_matrix_results_enumerates_independent_combinations():
    """
    Independent split matrices should combine their candidate solutions
    exhaustively, not only by reverse-index or position-index pairing.
    """
    encoding = {"A": 0, "B": 1, "C": 2, "D": 3}
    a = Partition((0,), encoding=encoding)
    b = Partition((1,), encoding=encoding)
    c = Partition((2,), encoding=encoding)
    d = Partition((3,), encoding=encoding)

    matrix_1 = [
        [_ps(a, encoding=encoding), _ps(b, encoding=encoding)],
        [_ps(b, encoding=encoding), _ps(a, encoding=encoding)],
    ]
    matrix_2 = [
        [_ps(c, encoding=encoding), _ps(d, encoding=encoding)],
        [_ps(d, encoding=encoding), _ps(c, encoding=encoding)],
    ]

    solutions = union_split_matrix_results([matrix_1, matrix_2])

    assert {
        tuple(sorted(part.bitmask for part in solution)) for solution in solutions
    } == {
        tuple(sorted((a.bitmask, c.bitmask))),
        tuple(sorted((a.bitmask, d.bitmask))),
        tuple(sorted((b.bitmask, c.bitmask))),
        tuple(sorted((b.bitmask, d.bitmask))),
    }


def test_union_split_matrix_results_unions_witnesses_from_each_submatrix():
    """Cartesian recombination keeps one witness set from each component."""
    encoding = {"A": 0, "B": 1, "C": 2}
    a = Partition((0,), encoding=encoding)
    b = Partition((1,), encoding=encoding)
    c = Partition((2,), encoding=encoding)

    matrix_1 = [[_ps(a, b, encoding=encoding), _ps(a, b, encoding=encoding)]]
    matrix_2 = [[_ps(c, encoding=encoding), _ps(c, encoding=encoding)]]

    solutions = union_split_matrix_results([matrix_1, matrix_2])

    assert {
        tuple(sorted(part.bitmask for part in solution)) for solution in solutions
    } == {tuple(sorted((a.bitmask, b.bitmask, c.bitmask)))}


def test_conflict_matrix_keeps_nesting_and_overlap_constraints():
    """Nesting and proper-overlap rows both remain visible when requested."""
    encoding = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    pivot = Partition((0, 1, 2, 3, 4), encoding=encoding)
    a = Partition((0,), encoding=encoding)
    b = Partition((1,), encoding=encoding)
    c = Partition((2,), encoding=encoding)
    d = Partition((3,), encoding=encoding)
    e = Partition((4,), encoding=encoding)
    ab = Partition((0, 1), encoding=encoding)
    bc = Partition((1, 2), encoding=encoding)
    de = Partition((3, 4), encoding=encoding)

    left_cover = _ps(a, b, encoding=encoding)
    right_cover = _ps(b, c, encoding=encoding)
    left_nested_cover = _ps(d, encoding=encoding)
    right_nested_cover = _ps(d, e, encoding=encoding)
    nesting_solution = _ps(d, encoding=encoding)

    subproblem = PivotEdgeSubproblem(
        pivot_split=pivot,
        tree1_node=Node(name="t1", split_indices=pivot, taxa_encoding=encoding),
        tree2_node=Node(name="t2", split_indices=pivot, taxa_encoding=encoding),
        tree1_child_frontiers={
            ab: ChildFrontiers(
                shared_top_splits=left_cover,
                bottom_partition_map={ab: left_cover},
            ),
            d: ChildFrontiers(
                shared_top_splits=left_nested_cover,
                bottom_partition_map={d: left_nested_cover},
            ),
        },
        tree2_child_frontiers={
            bc: ChildFrontiers(
                shared_top_splits=right_cover,
                bottom_partition_map={bc: right_cover},
            ),
            de: ChildFrontiers(
                shared_top_splits=right_nested_cover,
                bottom_partition_map={de: right_nested_cover},
            ),
        },
        child_subtree_splits_across_trees=_ps(a, b, c, d, e, encoding=encoding),
        encoding=encoding,
    )
    _assert_child_frontiers_valid(subproblem.tree1_child_frontiers)
    _assert_child_frontiers_valid(subproblem.tree2_child_frontiers)

    matrix = build_conflict_matrix(subproblem, include_top_containment=False)

    assert len(matrix) == 2
    assert any(row == [left_cover, right_cover] for row in matrix)
    assert any(row == [nesting_solution, nesting_solution] for row in matrix)


def test_solver_keeps_top_containment_candidates_when_overlap_exists():
    """
    Top-cover containment candidates can be valid alternatives in the same local
    component as an overlap row. They must not be suppressed just because the
    component also has a proper-overlap witness.
    """
    source = parse_newick("((O,T),(L,G));")
    destination = parse_newick("(((O,L),T),G);", encoding=source.taxa_encoding)
    encoding = source.taxa_encoding
    lbpenguin = Partition((encoding["L"],), encoding=encoding)

    solutions, _deleted = LatticeSolver(source, destination).solve_iteratively()
    movers = {
        partition for partitions in solutions.values() for partition in partitions
    }

    assert movers == {lbpenguin}


def test_mixed_direct_rows_do_not_turn_overlap_row_into_square_matrix():
    """
    A direct row [S, S] is an already-solved candidate, not a second column for
    the meet-product shape classifier. One overlap row plus one direct row must
    still expose the overlap row meet as a candidate.
    """
    encoding = {"O": 0, "T": 1, "L": 2, "G": 3}
    pivot = Partition((0, 1, 2, 3), encoding=encoding)
    o = Partition((0,), encoding=encoding)
    t = Partition((1,), encoding=encoding)
    l = Partition((2,), encoding=encoding)
    g = Partition((3,), encoding=encoding)
    ot = Partition((0, 1), encoding=encoding)
    lg = Partition((2, 3), encoding=encoding)
    ol = Partition((0, 2), encoding=encoding)
    olt = Partition((0, 1, 2), encoding=encoding)

    subproblem = PivotEdgeSubproblem(
        pivot_split=pivot,
        tree1_node=Node(name="t1", split_indices=pivot, taxa_encoding=encoding),
        tree2_node=Node(name="t2", split_indices=pivot, taxa_encoding=encoding),
        tree1_child_frontiers={
            lg: ChildFrontiers(
                shared_top_splits=_ps(l, g, encoding=encoding),
                bottom_partition_map={lg: _ps(l, g, encoding=encoding)},
            ),
            ot: ChildFrontiers(
                shared_top_splits=_ps(o, t, encoding=encoding),
                bottom_partition_map={ot: _ps(o, t, encoding=encoding)},
            ),
        },
        tree2_child_frontiers={
            g: ChildFrontiers(
                shared_top_splits=_ps(g, encoding=encoding),
                bottom_partition_map={g: _ps(g, encoding=encoding)},
            ),
            olt: ChildFrontiers(
                shared_top_splits=_ps(o, t, l, encoding=encoding),
                bottom_partition_map={ol: _ps(o, l, encoding=encoding)},
            ),
        },
        child_subtree_splits_across_trees=_ps(o, t, l, g, encoding=encoding),
        encoding=encoding,
    )

    solver = object.__new__(LatticeSolver)
    candidates = solver._solve_pivot_edge(subproblem)

    assert {
        tuple(sorted(partition.bitmask for partition in candidate))
        for candidate in candidates
    } == {
        (l.bitmask,),
        (g.bitmask,),
        tuple(sorted((o.bitmask, t.bitmask))),
    }


def test_conflict_matrix_keeps_all_nesting_witnesses_with_overlap_constraints():
    """Nesting witnesses remain visible when proper-overlap rows also exist."""
    encoding = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
    pivot = Partition((0, 1, 2, 3, 4, 5, 6), encoding=encoding)
    a = Partition((0,), encoding=encoding)
    b = Partition((1,), encoding=encoding)
    c = Partition((2,), encoding=encoding)
    d = Partition((3,), encoding=encoding)
    e = Partition((4,), encoding=encoding)
    f = Partition((5,), encoding=encoding)
    g = Partition((6,), encoding=encoding)
    ab = Partition((0, 1), encoding=encoding)
    bc = Partition((1, 2), encoding=encoding)
    de = Partition((3, 4), encoding=encoding)
    fg = Partition((5, 6), encoding=encoding)

    d_solution = _ps(d, encoding=encoding)
    f_solution = _ps(f, encoding=encoding)

    subproblem = PivotEdgeSubproblem(
        pivot_split=pivot,
        tree1_node=Node(name="t1", split_indices=pivot, taxa_encoding=encoding),
        tree2_node=Node(name="t2", split_indices=pivot, taxa_encoding=encoding),
        tree1_child_frontiers={
            ab: ChildFrontiers(
                shared_top_splits=_ps(a, b, encoding=encoding),
                bottom_partition_map={ab: _ps(a, b, encoding=encoding)},
            ),
            d: ChildFrontiers(
                shared_top_splits=d_solution,
                bottom_partition_map={d: d_solution},
            ),
            f: ChildFrontiers(
                shared_top_splits=f_solution,
                bottom_partition_map={f: f_solution},
            ),
        },
        tree2_child_frontiers={
            bc: ChildFrontiers(
                shared_top_splits=_ps(b, c, encoding=encoding),
                bottom_partition_map={bc: _ps(b, c, encoding=encoding)},
            ),
            de: ChildFrontiers(
                shared_top_splits=_ps(d, e, encoding=encoding),
                bottom_partition_map={de: _ps(d, e, encoding=encoding)},
            ),
            fg: ChildFrontiers(
                shared_top_splits=_ps(f, g, encoding=encoding),
                bottom_partition_map={fg: _ps(f, g, encoding=encoding)},
            ),
        },
        child_subtree_splits_across_trees=_ps(a, b, c, d, e, f, g, encoding=encoding),
        encoding=encoding,
    )

    matrix = build_conflict_matrix(subproblem)

    assert [_ps(a, b, encoding=encoding), _ps(b, c, encoding=encoding)] in matrix
    assert [d_solution, d_solution] in matrix
    assert [f_solution, f_solution] in matrix


def test_conflict_matrix_collects_strict_top_cover_containment_alternatives():
    """Strict top-cover containment exposes both sides of the frontier cut."""
    encoding = {"O": 0, "A": 1, "B": 2, "C": 3}
    pivot = Partition((0, 1, 2, 3), encoding=encoding)
    o = Partition((0,), encoding=encoding)
    a = Partition((1,), encoding=encoding)
    b = Partition((2,), encoding=encoding)
    c = Partition((3,), encoding=encoding)
    abc = Partition((1, 2, 3), encoding=encoding)
    full_child = Partition((0, 1, 2, 3), encoding=encoding)

    complement_witness = _ps(o, encoding=encoding)
    contained_witness = _ps(a, b, c, encoding=encoding)
    subproblem = PivotEdgeSubproblem(
        pivot_split=pivot,
        tree1_node=Node(name="t1", split_indices=pivot, taxa_encoding=encoding),
        tree2_node=Node(name="t2", split_indices=pivot, taxa_encoding=encoding),
        tree1_child_frontiers={
            full_child: ChildFrontiers(
                shared_top_splits=_ps(o, a, b, c, encoding=encoding),
                bottom_partition_map={},
            )
        },
        tree2_child_frontiers={
            abc: ChildFrontiers(
                shared_top_splits=_ps(a, b, c, encoding=encoding),
                bottom_partition_map={},
            )
        },
        child_subtree_splits_across_trees=_ps(o, a, b, c, encoding=encoding),
        encoding=encoding,
    )

    matrix = build_conflict_matrix(subproblem)

    assert {
        tuple(sorted(partition.bitmask for partition in row[0])) for row in matrix
    } == {
        tuple(sorted(partition.bitmask for partition in complement_witness)),
        tuple(sorted(partition.bitmask for partition in contained_witness)),
    }


def test_conflict_matrix_can_disable_top_cover_containment_for_residual_probes():
    """Mutated residual covers are not actual tree frontiers."""
    encoding = {"O": 0, "A": 1, "B": 2, "C": 3}
    pivot = Partition((0, 1, 2, 3), encoding=encoding)
    o = Partition((0,), encoding=encoding)
    a = Partition((1,), encoding=encoding)
    b = Partition((2,), encoding=encoding)
    c = Partition((3,), encoding=encoding)
    abc = Partition((1, 2, 3), encoding=encoding)

    subproblem = PivotEdgeSubproblem(
        pivot_split=pivot,
        tree1_node=Node(name="t1", split_indices=pivot, taxa_encoding=encoding),
        tree2_node=Node(name="t2", split_indices=pivot, taxa_encoding=encoding),
        tree1_child_frontiers={
            pivot: ChildFrontiers(
                shared_top_splits=_ps(o, a, b, c, encoding=encoding),
                bottom_partition_map={},
            )
        },
        tree2_child_frontiers={
            abc: ChildFrontiers(
                shared_top_splits=_ps(a, b, c, encoding=encoding),
                bottom_partition_map={},
            )
        },
        child_subtree_splits_across_trees=_ps(o, a, b, c, encoding=encoding),
        encoding=encoding,
    )

    assert (
        build_conflict_matrix(
            subproblem,
            include_top_containment=False,
        )
        == []
    )


def test_conflict_matrix_type_hints_are_resolvable():
    """Deferred annotations should still resolve under typing introspection."""
    hints = get_type_hints(_nesting_solution_rank_key)

    assert hints["solution"] == PartitionSet[Partition]


def test_empty_unique_child_frontiers_use_maximal_shared_splits():
    """The no-unique-children path should still construct frontier antichains."""
    encoding = {"A": 0, "B": 1, "C": 2}
    a = Partition((0,), encoding=encoding)
    ab = Partition((0, 1), encoding=encoding)
    c = Partition((2,), encoding=encoding)
    shared_splits = _ps(a, ab, c, encoding=encoding)

    frontiers = compute_child_frontiers(
        parent=Node(
            name="pivot",
            split_indices=Partition((0, 1, 2), encoding=encoding),
            taxa_encoding=encoding,
        ),
        children_to_process=PartitionSet(encoding=encoding),
        shared_splits=shared_splits,
    )

    assert set(frontiers.keys()) == {ab, c}
    assert a not in frontiers


def test_completion_sequence_reuses_identical_residual_state_solutions():
    """Equivalent accepted candidates should share residual solve work."""
    encoding = {"A": 0, "B": 1, "C": 2}
    pivot = Partition((0, 1, 2), encoding=encoding)
    a = Partition((0,), encoding=encoding)
    b = Partition((1,), encoding=encoding)
    c = Partition((2,), encoding=encoding)
    ab = Partition((0, 1), encoding=encoding)

    subproblem = PivotEdgeSubproblem(
        pivot_split=pivot,
        tree1_node=Node(name="t1", split_indices=pivot, taxa_encoding=encoding),
        tree2_node=Node(name="t2", split_indices=pivot, taxa_encoding=encoding),
        tree1_child_frontiers={
            pivot: ChildFrontiers(
                shared_top_splits=_ps(a, b, c, encoding=encoding),
                bottom_partition_map={
                    a: _ps(a, encoding=encoding),
                    b: _ps(b, encoding=encoding),
                    c: _ps(c, encoding=encoding),
                },
            )
        },
        tree2_child_frontiers={
            pivot: ChildFrontiers(
                shared_top_splits=_ps(a, b, c, encoding=encoding),
                bottom_partition_map={
                    a: _ps(a, encoding=encoding),
                    b: _ps(b, encoding=encoding),
                    c: _ps(c, encoding=encoding),
                },
            )
        },
        child_subtree_splits_across_trees=_ps(a, b, c, encoding=encoding),
        encoding=encoding,
    )

    solver = object.__new__(LatticeSolver)
    solve_calls = 0

    def solve_residual(
        _branch: PivotEdgeSubproblem,
        include_top_containment: bool = True,
    ) -> list[PartitionSet[Partition]]:
        nonlocal solve_calls
        solve_calls += 1
        return [_ps(c, encoding=encoding)]

    solver._solve_pivot_edge = solve_residual

    sequence = solver._best_completion_sequence_for_pivot(
        subproblem,
        [
            _ps(ab, encoding=encoding),
            _ps(a, b, encoding=encoding),
        ],
    )

    assert solve_calls == 1
    assert sequence == [_ps(ab, encoding=encoding), _ps(c, encoding=encoding)]


def test_completion_sequence_does_not_exhaust_current_tree_root():
    """A residual sequence cannot select movers covering the whole current tree."""
    tree1 = parse_newick("(A,(B,C));")
    tree2 = parse_newick("(A,(B,C));", encoding=tree1.taxa_encoding)
    encoding = tree1.taxa_encoding
    pivot = Partition((0, 1, 2), encoding=encoding)
    a = Partition((0,), encoding=encoding)
    b = Partition((1,), encoding=encoding)
    c = Partition((2,), encoding=encoding)
    bc = Partition((1, 2), encoding=encoding)

    subproblem = PivotEdgeSubproblem(
        pivot_split=pivot,
        tree1_node=tree1,
        tree2_node=tree2,
        tree1_child_frontiers={
            pivot: ChildFrontiers(
                shared_top_splits=_ps(a, b, c, encoding=encoding),
                bottom_partition_map={
                    a: _ps(a, encoding=encoding),
                    b: _ps(b, encoding=encoding),
                    c: _ps(c, encoding=encoding),
                },
            )
        },
        tree2_child_frontiers={
            pivot: ChildFrontiers(
                shared_top_splits=_ps(a, b, c, encoding=encoding),
                bottom_partition_map={
                    a: _ps(a, encoding=encoding),
                    b: _ps(b, encoding=encoding),
                    c: _ps(c, encoding=encoding),
                },
            )
        },
        child_subtree_splits_across_trees=_ps(a, b, c, encoding=encoding),
        encoding=encoding,
    )

    solver = object.__new__(LatticeSolver)
    solver.current_t1 = tree1
    solver.current_t2 = tree2
    solver._solve_pivot_edge = lambda _branch, include_top_containment=True: [
        _ps(bc, encoding=encoding)
    ]

    sequence = solver._best_completion_sequence_for_pivot(
        subproblem, [_ps(a, encoding=encoding)]
    )

    assert sequence == [_ps(a, encoding=encoding)]


def test_solver_prefers_atomic_residual_witnesses_for_equal_repairs():
    """
    For (((F,G),I),M) -> (((F,M),I),G), deleting {F,I} and deleting
    {G,M} both have two singleton movers. The atomic residual witnesses are
    {G} and {M}; {F,I} is the proper-overlap meet of shared anchors.
    """
    source = parse_newick("(((F,G),I),M);")
    destination = parse_newick("(((F,M),I),G);", encoding=source.taxa_encoding)

    solutions, _deleted = LatticeSolver(source, destination).solve_iteratively()
    movers = {
        tuple(partition.resolve_to_indices())
        for partitions in solutions.values()
        for partition in partitions
    }

    assert movers == {(1,), (3,)}


def test_publication_pair_183_resolves_strict_top_cover_containment():
    """The 24-taxa residual pair with Ostrich as a direct child must complete."""
    pytest.skip("historical FastTree-specific 24-taxa fixture is not retained in publication_data")
    source = parse_newick(newicks[183])
    destination = parse_newick(newicks[184], encoding=source.taxa_encoding)

    ostrich_index = source.taxa_encoding["Ostrich"]
    solutions, _deleted = LatticeSolver(source, destination).solve_iteratively()

    assert any(
        partition.resolve_to_indices() == (ostrich_index,)
        for partitions in solutions.values()
        for partition in partitions
    )

    source = parse_newick(newicks[183])
    destination = parse_newick(newicks[184], encoding=source.taxa_encoding)
    result = process_tree_pair_interpolation(source, destination, pair_index=183)

    assert result.trees
