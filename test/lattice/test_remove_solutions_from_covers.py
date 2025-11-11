"""
Test for PivotEdgeSubproblem.remove_solutions_from_covers method.

This test verifies that the method correctly removes partitions from cover sets
using PartitionSet.discard() rather than the non-existent .remove() method.
"""

from brancharchitect.elements.partition_set import PartitionSet, Partition
from brancharchitect.jumping_taxa.lattice.pivot_edge_subproblem import (
    PivotEdgeSubproblem,
)
from brancharchitect.jumping_taxa.lattice.types import TopToBottom
from brancharchitect.tree import Node


def create_simple_tree(newick: str, name: str = "tree") -> Node:
    """Helper to create a tree from Newick string."""
    from brancharchitect.parser import parse_newick

    result = parse_newick(newick)
    # parse_newick returns a single Node for single trees (not a list)
    tree = result if isinstance(result, Node) else result[0]
    tree.name = name
    return tree


class TestRemoveSolutionsFromCovers:
    """Test suite for PivotEdgeSubproblem.remove_solutions_from_covers method."""

    def test_remove_partitions_from_covers_basic(self):
        """Test basic removal of partitions from cover sets."""
        # Setup encoding
        encoding = {"A": 0, "B": 1, "C": 2, "D": 3}

        # Create partitions
        p1 = Partition((0, 1), encoding)  # A, B
        p2 = Partition((2, 3), encoding)  # C, D
        p3 = Partition((0, 2), encoding)  # A, C
        p4 = Partition((1, 3), encoding)  # B, D

        # Create cover sets
        cover1 = PartitionSet({p1, p2, p3}, encoding=encoding)
        cover2 = PartitionSet({p2, p4}, encoding=encoding)

        # Create trees (minimal setup)
        t1 = create_simple_tree("((A,B),(C,D));")
        t2 = create_simple_tree("((A,C),(B,D));")

        # Initialize split_indices for nodes
        t1._initialize_split_indices(encoding)
        t2._initialize_split_indices(encoding)

        # Create TopToBottom structures for the frontiers
        t1_top_to_bottom = TopToBottom(
            shared_top_splits=cover1.copy(), bottom_to_frontiers={}
        )
        t2_top_to_bottom = TopToBottom(
            shared_top_splits=cover2.copy(), bottom_to_frontiers={}
        )

        # Create a PivotEdgeSubproblem with these frontiers
        edge = PivotEdgeSubproblem(
            pivot_split=p1,
            tree1_node=t1,
            tree2_node=t2,
            tree1_child_frontiers={p1: t1_top_to_bottom},
            tree2_child_frontiers={p1: t2_top_to_bottom},
            child_subtree_splits_across_trees=PartitionSet(set(), encoding=encoding),
            encoding=encoding,
        )

        # Verify initial state
        assert len(edge.tree1_child_frontiers[p1].shared_top_splits) == 3
        assert len(edge.tree2_child_frontiers[p1].shared_top_splits) == 2
        assert p1 in edge.tree1_child_frontiers[p1].shared_top_splits
        assert p2 in edge.tree1_child_frontiers[p1].shared_top_splits
        assert p2 in edge.tree2_child_frontiers[p1].shared_top_splits

        # Create solutions to remove
        solution1 = PartitionSet({p1, p2}, encoding=encoding)
        solutions = [solution1]

        # Remove solutions
        edge.remove_solutions_from_covers(solutions)

        # Verify p1 and p2 were removed from covers
        assert p1 not in edge.tree1_child_frontiers[p1].shared_top_splits
        assert p2 not in edge.tree1_child_frontiers[p1].shared_top_splits
        assert p2 not in edge.tree2_child_frontiers[p1].shared_top_splits

        # Verify p3 and p4 remain (were not in solutions)
        assert p3 in edge.tree1_child_frontiers[p1].shared_top_splits
        assert p4 in edge.tree2_child_frontiers[p1].shared_top_splits

        # Verify final counts
        assert (
            len(edge.tree1_child_frontiers[p1].shared_top_splits) == 1
        )  # only p3 remains
        assert (
            len(edge.tree2_child_frontiers[p1].shared_top_splits) == 1
        )  # only p4 remains

    def test_remove_from_multiple_cover_sets(self):
        """Test removal from multiple cover sets in the same list."""
        encoding = {"A": 0, "B": 1, "C": 2}

        p1 = Partition((0, 1), encoding)  # A, B
        p2 = Partition((1, 2), encoding)  # B, C
        p3 = Partition((0, 2), encoding)  # A, C

        # Create multiple cover sets, some containing the same partitions
        cover1 = PartitionSet({p1, p2}, encoding=encoding)
        cover2 = PartitionSet({p1, p3}, encoding=encoding)
        cover3 = PartitionSet({p2, p3}, encoding=encoding)

        t1 = create_simple_tree("(A,(B,C));")
        t2 = create_simple_tree("(B,(A,C));")

        t1._initialize_split_indices(encoding)
        t2._initialize_split_indices(encoding)

        # Create TopToBottom structures for multiple covers
        edge = PivotEdgeSubproblem(
            pivot_split=p1,
            tree1_node=t1,
            tree2_node=t2,
            tree1_child_frontiers={
                p1: TopToBottom(
                    shared_top_splits=cover1.copy(), bottom_to_frontiers={}
                ),
                p2: TopToBottom(
                    shared_top_splits=cover2.copy(), bottom_to_frontiers={}
                ),
                p3: TopToBottom(
                    shared_top_splits=cover3.copy(), bottom_to_frontiers={}
                ),
            },
            tree2_child_frontiers={},
            child_subtree_splits_across_trees=PartitionSet(set(), encoding=encoding),
            encoding=encoding,
        )

        # Remove p1 from all covers
        solution = PartitionSet({p1}, encoding=encoding)
        edge.remove_solutions_from_covers([solution])

        # Verify p1 was removed from covers that contained it
        assert (
            p1 not in edge.tree1_child_frontiers[p1].shared_top_splits
        )  # was in cover1
        assert (
            p1 not in edge.tree1_child_frontiers[p2].shared_top_splits
        )  # was in cover2
        # cover3 never had p1, so it should remain unchanged
        assert len(edge.tree1_child_frontiers[p3].shared_top_splits) == 2

        # Verify other partitions remain
        assert p2 in edge.tree1_child_frontiers[p1].shared_top_splits
        assert p3 in edge.tree1_child_frontiers[p2].shared_top_splits
        assert p2 in edge.tree1_child_frontiers[p3].shared_top_splits
        assert p3 in edge.tree1_child_frontiers[p3].shared_top_splits

    def test_remove_nonexistent_partition(self):
        """Test that removing a partition not in any cover doesn't cause errors."""
        encoding = {"A": 0, "B": 1, "C": 2}

        p1 = Partition((0, 1), encoding)  # A, B
        p2 = Partition((1, 2), encoding)  # B, C
        p3 = Partition((0, 2), encoding)  # A, C (not in any cover)

        cover1 = PartitionSet({p1}, encoding=encoding)
        cover2 = PartitionSet({p2}, encoding=encoding)

        t1 = create_simple_tree("(A,(B,C));")
        t2 = create_simple_tree("(B,(A,C));")

        t1._initialize_split_indices(encoding)
        t2._initialize_split_indices(encoding)

        edge = PivotEdgeSubproblem(
            pivot_split=p1,
            tree1_node=t1,
            tree2_node=t2,
            tree1_child_frontiers={
                p1: TopToBottom(shared_top_splits=cover1.copy(), bottom_to_frontiers={})
            },
            tree2_child_frontiers={
                p2: TopToBottom(shared_top_splits=cover2.copy(), bottom_to_frontiers={})
            },
            child_subtree_splits_across_trees=PartitionSet(set(), encoding=encoding),
            encoding=encoding,
        )

        # Try to remove p3 which is not in any cover - should not raise error
        solution = PartitionSet({p3}, encoding=encoding)
        edge.remove_solutions_from_covers([solution])

        # Verify covers unchanged
        assert p1 in edge.tree1_child_frontiers[p1].shared_top_splits
        assert p2 in edge.tree2_child_frontiers[p2].shared_top_splits
        assert len(edge.tree1_child_frontiers[p1].shared_top_splits) == 1
        assert len(edge.tree2_child_frontiers[p2].shared_top_splits) == 1

    def test_remove_multiple_solutions(self):
        """Test removal of multiple solution sets."""
        encoding = {"A": 0, "B": 1, "C": 2, "D": 3}

        p1 = Partition((0, 1), encoding)  # A, B
        p2 = Partition((2, 3), encoding)  # C, D
        p3 = Partition((0, 2), encoding)  # A, C
        p4 = Partition((1, 3), encoding)  # B, D

        cover = PartitionSet({p1, p2, p3, p4}, encoding=encoding)

        t1 = create_simple_tree("((A,B),(C,D));")
        t2 = create_simple_tree("((A,C),(B,D));")

        t1._initialize_split_indices(encoding)
        t2._initialize_split_indices(encoding)

        edge = PivotEdgeSubproblem(
            pivot_split=p1,
            tree1_node=t1,
            tree2_node=t2,
            tree1_child_frontiers={
                p1: TopToBottom(shared_top_splits=cover.copy(), bottom_to_frontiers={})
            },
            tree2_child_frontiers={},
            child_subtree_splits_across_trees=PartitionSet(set(), encoding=encoding),
            encoding=encoding,
        )

        # Create multiple solutions
        solution1 = PartitionSet({p1}, encoding=encoding)
        solution2 = PartitionSet({p2, p3}, encoding=encoding)

        # Remove all solutions
        edge.remove_solutions_from_covers([solution1, solution2])

        # Verify p1, p2, p3 were removed
        assert p1 not in edge.tree1_child_frontiers[p1].shared_top_splits
        assert p2 not in edge.tree1_child_frontiers[p1].shared_top_splits
        assert p3 not in edge.tree1_child_frontiers[p1].shared_top_splits

        # Verify p4 remains
        assert p4 in edge.tree1_child_frontiers[p1].shared_top_splits
        assert len(edge.tree1_child_frontiers[p1].shared_top_splits) == 1

    def test_empty_solutions_list(self):
        """Test that empty solutions list doesn't modify covers."""
        encoding = {"A": 0, "B": 1}

        p1 = Partition((0, 1), encoding)
        cover = PartitionSet({p1}, encoding=encoding)

        t1 = create_simple_tree("(A,B);")
        t2 = create_simple_tree("(A,B);")

        t1._initialize_split_indices(encoding)
        t2._initialize_split_indices(encoding)

        edge = PivotEdgeSubproblem(
            pivot_split=p1,
            tree1_node=t1,
            tree2_node=t2,
            tree1_child_frontiers={
                p1: TopToBottom(shared_top_splits=cover.copy(), bottom_to_frontiers={})
            },
            tree2_child_frontiers={},
            child_subtree_splits_across_trees=PartitionSet(set(), encoding=encoding),
            encoding=encoding,
        )

        # Remove empty solutions list
        edge.remove_solutions_from_covers([])

        # Verify cover unchanged
        assert p1 in edge.tree1_child_frontiers[p1].shared_top_splits
        assert len(edge.tree1_child_frontiers[p1].shared_top_splits) == 1

    def test_uses_discard_not_remove(self):
        """
        Verify that the method uses discard() which doesn't raise KeyError,
        rather than remove() which would raise an error for missing elements.
        """
        encoding = {"A": 0, "B": 1, "C": 2}

        p1 = Partition((0, 1), encoding)
        p2 = Partition((1, 2), encoding)  # Not in cover

        cover = PartitionSet({p1}, encoding=encoding)

        t1 = create_simple_tree("(A,(B,C));")
        t2 = create_simple_tree("(A,(B,C));")

        t1._initialize_split_indices(encoding)
        t2._initialize_split_indices(encoding)

        edge = PivotEdgeSubproblem(
            pivot_split=p1,
            tree1_node=t1,
            tree2_node=t2,
            tree1_child_frontiers={
                p1: TopToBottom(shared_top_splits=cover.copy(), bottom_to_frontiers={})
            },
            tree2_child_frontiers={},
            child_subtree_splits_across_trees=PartitionSet(set(), encoding=encoding),
            encoding=encoding,
        )

        # This should NOT raise an error even though p2 is not in the cover
        # because discard() is used instead of remove()
        solution = PartitionSet({p2}, encoding=encoding)

        # Should complete without error
        edge.remove_solutions_from_covers([solution])

        # Cover should remain unchanged
        assert p1 in edge.tree1_child_frontiers[p1].shared_top_splits
        assert len(edge.tree1_child_frontiers[p1].shared_top_splits) == 1

    def test_remove_from_frontier_sets(self):
        """Test removal of partitions from frontier sets in bottom_to_frontiers."""
        encoding = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

        # Create partitions
        p1 = Partition((0, 1), encoding)  # A, B
        p2 = Partition((2, 3), encoding)  # C, D
        p3 = Partition((4,), encoding)  # E
        p4 = Partition((0, 2), encoding)  # A, C

        # Create cover with frontier mappings
        cover = PartitionSet({p1, p2, p3}, encoding=encoding)
        bottom_to_frontiers = {
            p4: PartitionSet({p1, p2}, encoding=encoding),  # p4 maps to {p1, p2}
            p3: PartitionSet({p3}, encoding=encoding),  # p3 maps to itself
        }

        t1 = create_simple_tree("((A,B),(C,D),E);")
        t2 = create_simple_tree("((A,C),(B,D),E);")

        t1._initialize_split_indices(encoding)
        t2._initialize_split_indices(encoding)

        # Create edge with frontier mappings
        edge = PivotEdgeSubproblem(
            pivot_split=p1,
            tree1_node=t1,
            tree2_node=t2,
            tree1_child_frontiers={
                p1: TopToBottom(
                    shared_top_splits=cover.copy(),
                    bottom_to_frontiers=bottom_to_frontiers.copy(),
                )
            },
            tree2_child_frontiers={},
            child_subtree_splits_across_trees=PartitionSet(set(), encoding=encoding),
            encoding=encoding,
        )

        # Verify initial frontier state
        assert p1 in edge.tree1_child_frontiers[p1].bottom_to_frontiers[p4]
        assert p2 in edge.tree1_child_frontiers[p1].bottom_to_frontiers[p4]
        assert len(edge.tree1_child_frontiers[p1].bottom_to_frontiers[p4]) == 2

        # Remove p1 from solutions
        solution = PartitionSet({p1}, encoding=encoding)
        edge.remove_solutions_from_covers([solution])

        # Verify p1 removed from cover
        assert p1 not in edge.tree1_child_frontiers[p1].shared_top_splits

        # Verify p1 removed from frontier sets
        assert p1 not in edge.tree1_child_frontiers[p1].bottom_to_frontiers[p4]
        assert p2 in edge.tree1_child_frontiers[p1].bottom_to_frontiers[p4]
        assert len(edge.tree1_child_frontiers[p1].bottom_to_frontiers[p4]) == 1

    def test_remove_bottom_key_when_partition_is_bottom(self):
        """Test that partition is removed from bottom_to_frontiers keys when it's a bottom."""
        encoding = {"A": 0, "B": 1, "C": 2, "D": 3}

        p1 = Partition((0, 1), encoding)  # A, B
        p2 = Partition((2, 3), encoding)  # C, D
        p3 = Partition((0, 2), encoding)  # A, C

        # p1 is both in cover and is a bottom key
        cover = PartitionSet({p1, p2}, encoding=encoding)
        bottom_to_frontiers = {
            p1: PartitionSet({p1}, encoding=encoding),  # p1 as bottom
            p3: PartitionSet({p2}, encoding=encoding),  # p3 as bottom
        }

        t1 = create_simple_tree("((A,B),(C,D));")
        t2 = create_simple_tree("((A,C),(B,D));")

        t1._initialize_split_indices(encoding)
        t2._initialize_split_indices(encoding)

        edge = PivotEdgeSubproblem(
            pivot_split=p1,
            tree1_node=t1,
            tree2_node=t2,
            tree1_child_frontiers={
                p1: TopToBottom(
                    shared_top_splits=cover.copy(),
                    bottom_to_frontiers=bottom_to_frontiers.copy(),
                )
            },
            tree2_child_frontiers={},
            child_subtree_splits_across_trees=PartitionSet(set(), encoding=encoding),
            encoding=encoding,
        )

        # Verify p1 exists as a bottom key
        assert p1 in edge.tree1_child_frontiers[p1].bottom_to_frontiers

        # Remove p1
        solution = PartitionSet({p1}, encoding=encoding)
        edge.remove_solutions_from_covers([solution])

        # Verify p1 removed from cover
        assert p1 not in edge.tree1_child_frontiers[p1].shared_top_splits

        # Verify p1 removed as bottom key
        assert p1 not in edge.tree1_child_frontiers[p1].bottom_to_frontiers

        # Verify p3 remains as bottom key and p2 remains in its frontier
        assert p3 in edge.tree1_child_frontiers[p1].bottom_to_frontiers
        assert p2 in edge.tree1_child_frontiers[p1].bottom_to_frontiers[p3]

    def test_excluded_partitions_tracking(self):
        """Test that removed partitions are tracked in excluded_partitions."""
        encoding = {"A": 0, "B": 1, "C": 2, "D": 3}

        p1 = Partition((0, 1), encoding)  # A, B
        p2 = Partition((2, 3), encoding)  # C, D
        p3 = Partition((0, 2), encoding)  # A, C

        cover = PartitionSet({p1, p2, p3}, encoding=encoding)

        t1 = create_simple_tree("((A,B),(C,D));")
        t2 = create_simple_tree("((A,C),(B,D));")

        t1._initialize_split_indices(encoding)
        t2._initialize_split_indices(encoding)

        edge = PivotEdgeSubproblem(
            pivot_split=p1,
            tree1_node=t1,
            tree2_node=t2,
            tree1_child_frontiers={
                p1: TopToBottom(shared_top_splits=cover.copy(), bottom_to_frontiers={})
            },
            tree2_child_frontiers={},
            child_subtree_splits_across_trees=PartitionSet(set(), encoding=encoding),
            encoding=encoding,
        )

        # Verify excluded_partitions initially empty
        assert len(edge.excluded_partitions) == 0

        # Remove p1 and p2
        solution1 = PartitionSet({p1}, encoding=encoding)
        solution2 = PartitionSet({p2}, encoding=encoding)
        edge.remove_solutions_from_covers([solution1, solution2])

        # Verify excluded_partitions contains p1 and p2
        assert p1 in edge.excluded_partitions
        assert p2 in edge.excluded_partitions
        assert p3 not in edge.excluded_partitions
        assert len(edge.excluded_partitions) == 2

    def test_has_remaining_conflicts_after_removal(self):
        """Test has_remaining_conflicts() method after removing solutions."""
        encoding = {"A": 0, "B": 1, "C": 2, "D": 3}

        p1 = Partition((0, 1), encoding)  # A, B
        p2 = Partition((2, 3), encoding)  # C, D

        cover = PartitionSet({p1, p2}, encoding=encoding)

        t1 = create_simple_tree("((A,B),(C,D));")
        t2 = create_simple_tree("((A,C),(B,D));")

        t1._initialize_split_indices(encoding)
        t2._initialize_split_indices(encoding)

        edge = PivotEdgeSubproblem(
            pivot_split=p1,
            tree1_node=t1,
            tree2_node=t2,
            tree1_child_frontiers={
                p1: TopToBottom(shared_top_splits=cover.copy(), bottom_to_frontiers={})
            },
            tree2_child_frontiers={},
            child_subtree_splits_across_trees=PartitionSet(set(), encoding=encoding),
            encoding=encoding,
        )

        # Initially has conflicts
        assert edge.has_remaining_conflicts() is True

        # Remove all partitions
        solution = PartitionSet({p1, p2}, encoding=encoding)
        edge.remove_solutions_from_covers([solution])

        # Should have no remaining conflicts
        assert edge.has_remaining_conflicts() is False

    def test_sequential_removal_iterations(self):
        """Test removing solutions across multiple iterations (simulating algorithm behavior)."""
        encoding = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}

        # Create a complex set of partitions
        p1 = Partition((0, 1), encoding)  # A, B
        p2 = Partition((2, 3), encoding)  # C, D
        p3 = Partition((4, 5), encoding)  # E, F
        p4 = Partition((0, 2), encoding)  # A, C
        p5 = Partition((1, 3), encoding)  # B, D
        p6 = Partition((4,), encoding)  # E

        # Create covers for both trees
        cover1 = PartitionSet({p1, p2, p3, p4}, encoding=encoding)
        cover2 = PartitionSet({p2, p3, p5, p6}, encoding=encoding)

        t1 = create_simple_tree("((A,B),(C,D),(E,F));")
        t2 = create_simple_tree("((A,C),(B,D),(E,F));")

        t1._initialize_split_indices(encoding)
        t2._initialize_split_indices(encoding)

        edge = PivotEdgeSubproblem(
            pivot_split=p1,
            tree1_node=t1,
            tree2_node=t2,
            tree1_child_frontiers={
                p1: TopToBottom(shared_top_splits=cover1.copy(), bottom_to_frontiers={})
            },
            tree2_child_frontiers={
                p1: TopToBottom(shared_top_splits=cover2.copy(), bottom_to_frontiers={})
            },
            child_subtree_splits_across_trees=PartitionSet(set(), encoding=encoding),
            encoding=encoding,
        )

        # Iteration 1: Remove p1 and p2
        solution_iter1 = PartitionSet({p1, p2}, encoding=encoding)
        edge.remove_solutions_from_covers([solution_iter1])

        # Verify state after iteration 1
        assert p1 not in edge.tree1_child_frontiers[p1].shared_top_splits
        assert p2 not in edge.tree1_child_frontiers[p1].shared_top_splits
        assert p3 in edge.tree1_child_frontiers[p1].shared_top_splits
        assert p4 in edge.tree1_child_frontiers[p1].shared_top_splits

        assert p2 not in edge.tree2_child_frontiers[p1].shared_top_splits
        assert p3 in edge.tree2_child_frontiers[p1].shared_top_splits
        assert p5 in edge.tree2_child_frontiers[p1].shared_top_splits

        assert len(edge.excluded_partitions) == 2

        # Iteration 2: Remove p3 and p4
        solution_iter2 = PartitionSet({p3, p4}, encoding=encoding)
        edge.remove_solutions_from_covers([solution_iter2])

        # Verify state after iteration 2
        assert p3 not in edge.tree1_child_frontiers[p1].shared_top_splits
        assert p4 not in edge.tree1_child_frontiers[p1].shared_top_splits
        assert len(edge.tree1_child_frontiers[p1].shared_top_splits) == 0

        assert p3 not in edge.tree2_child_frontiers[p1].shared_top_splits
        assert p5 in edge.tree2_child_frontiers[p1].shared_top_splits
        assert p6 in edge.tree2_child_frontiers[p1].shared_top_splits

        assert len(edge.excluded_partitions) == 4
        assert p1 in edge.excluded_partitions
        assert p2 in edge.excluded_partitions
        assert p3 in edge.excluded_partitions
        assert p4 in edge.excluded_partitions

    def test_complex_frontier_removal_with_nested_bottoms(self):
        """Test removal from complex frontier structures with nested bottom mappings."""
        encoding = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}

        # Create a complex hierarchy of partitions
        p_ab = Partition((0, 1), encoding)  # A, B
        p_cd = Partition((2, 3), encoding)  # C, D
        p_ef = Partition((4, 5), encoding)  # E, F
        p_g = Partition((6,), encoding)  # G
        p_abc = Partition((0, 1, 2), encoding)  # A, B, C
        p_def = Partition((3, 4, 5), encoding)  # D, E, F

        # Create complex cover and frontier structure
        cover = PartitionSet({p_ab, p_cd, p_ef, p_g}, encoding=encoding)
        bottom_to_frontiers = {
            p_abc: PartitionSet({p_ab, p_cd}, encoding=encoding),
            p_def: PartitionSet({p_cd, p_ef}, encoding=encoding),
            p_g: PartitionSet({p_g}, encoding=encoding),
        }

        t1 = create_simple_tree("((A,B),(C,D),(E,F),G);")
        t2 = create_simple_tree("((A,C),(B,D),(E,F),G);")

        t1._initialize_split_indices(encoding)
        t2._initialize_split_indices(encoding)

        edge = PivotEdgeSubproblem(
            pivot_split=p_ab,
            tree1_node=t1,
            tree2_node=t2,
            tree1_child_frontiers={
                p_ab: TopToBottom(
                    shared_top_splits=cover.copy(),
                    bottom_to_frontiers=bottom_to_frontiers.copy(),
                )
            },
            tree2_child_frontiers={},
            child_subtree_splits_across_trees=PartitionSet(set(), encoding=encoding),
            encoding=encoding,
        )

        # Verify initial complex state
        assert len(edge.tree1_child_frontiers[p_ab].shared_top_splits) == 4
        assert len(edge.tree1_child_frontiers[p_ab].bottom_to_frontiers) == 3
        assert p_cd in edge.tree1_child_frontiers[p_ab].bottom_to_frontiers[p_abc]
        assert p_cd in edge.tree1_child_frontiers[p_ab].bottom_to_frontiers[p_def]

        # Remove p_cd (appears in multiple frontiers)
        solution = PartitionSet({p_cd}, encoding=encoding)
        edge.remove_solutions_from_covers([solution])

        # Verify p_cd removed from cover
        assert p_cd not in edge.tree1_child_frontiers[p_ab].shared_top_splits
        assert len(edge.tree1_child_frontiers[p_ab].shared_top_splits) == 3

        # Verify p_cd removed from all frontier sets where it appeared
        assert p_cd not in edge.tree1_child_frontiers[p_ab].bottom_to_frontiers[p_abc]
        assert p_cd not in edge.tree1_child_frontiers[p_ab].bottom_to_frontiers[p_def]

        # Verify p_ab and p_ef remain in appropriate frontiers
        assert p_ab in edge.tree1_child_frontiers[p_ab].bottom_to_frontiers[p_abc]
        assert p_ef in edge.tree1_child_frontiers[p_ab].bottom_to_frontiers[p_def]

        # Verify all bottom keys still exist
        assert len(edge.tree1_child_frontiers[p_ab].bottom_to_frontiers) == 3

    def test_overlapping_solutions_removal(self):
        """Test removal when multiple solutions contain overlapping partitions."""
        encoding = {"A": 0, "B": 1, "C": 2, "D": 3}

        p1 = Partition((0, 1), encoding)  # A, B
        p2 = Partition((2, 3), encoding)  # C, D
        p3 = Partition((0, 2), encoding)  # A, C

        cover = PartitionSet({p1, p2, p3}, encoding=encoding)

        t1 = create_simple_tree("((A,B),(C,D));")
        t2 = create_simple_tree("((A,C),(B,D));")

        t1._initialize_split_indices(encoding)
        t2._initialize_split_indices(encoding)

        edge = PivotEdgeSubproblem(
            pivot_split=p1,
            tree1_node=t1,
            tree2_node=t2,
            tree1_child_frontiers={
                p1: TopToBottom(shared_top_splits=cover.copy(), bottom_to_frontiers={})
            },
            tree2_child_frontiers={},
            child_subtree_splits_across_trees=PartitionSet(set(), encoding=encoding),
            encoding=encoding,
        )

        # Create overlapping solutions: both contain p2
        solution1 = PartitionSet({p1, p2}, encoding=encoding)
        solution2 = PartitionSet({p2, p3}, encoding=encoding)

        # Remove both solutions (p2 should only be removed once)
        edge.remove_solutions_from_covers([solution1, solution2])

        # Verify all partitions removed
        assert p1 not in edge.tree1_child_frontiers[p1].shared_top_splits
        assert p2 not in edge.tree1_child_frontiers[p1].shared_top_splits
        assert p3 not in edge.tree1_child_frontiers[p1].shared_top_splits
        assert len(edge.tree1_child_frontiers[p1].shared_top_splits) == 0

        # Verify excluded_partitions contains all unique partitions
        assert len(edge.excluded_partitions) == 3
        assert p1 in edge.excluded_partitions
        assert p2 in edge.excluded_partitions
        assert p3 in edge.excluded_partitions
