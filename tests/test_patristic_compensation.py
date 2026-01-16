import unittest
import logging
from brancharchitect.tree import Node
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree_interpolation.topology_ops.weights import (
    distribute_path_weights,
    get_virtual_collapse_weights,
)
from brancharchitect.elements.partition import Partition

# Disable logging for tests
logging.disable(logging.CRITICAL)


class TestPatristicCompensation(unittest.TestCase):
    def test_distribute_path_weights_add(self):
        """Test 'add' operation for distribute_path_weights."""
        # Tree: (A:1,B:2):5;
        tree = parse_newick("(A:1.0,B:2.0):5.0;")
        # Root is at (A,B). Internal branch is the one leading to (A,B) clade.

        path = [tree.split_indices]

        distribute_path_weights(tree, path, operation="add")

        leaf_a = [leaf for leaf in tree.get_leaves() if leaf.name == "A"][0]
        leaf_b = [leaf for leaf in tree.get_leaves() if leaf.name == "B"][0]

        # A: 1.0 + 5.0 = 6.0
        # B: 2.0 + 5.0 = 7.0
        self.assertEqual(leaf_a.length, 6.0)
        self.assertEqual(leaf_b.length, 7.0)

    def test_distribute_path_weights_subtract(self):
        """Test 'subtract' operation for distribute_path_weights."""
        # Tree: (A:6,B:7):5;
        tree = parse_newick("(A:6.0,B:7.0):5.0;")

        path = [tree.split_indices]

        distribute_path_weights(tree, path, operation="subtract")

        leaf_a = [leaf for leaf in tree.get_leaves() if leaf.name == "A"][0]
        leaf_b = [leaf for leaf in tree.get_leaves() if leaf.name == "B"][0]

        # A: 6.0 - 5.0 = 1.0
        # B: 7.0 - 5.0 = 2.0
        self.assertEqual(leaf_a.length, 1.0)
        self.assertEqual(leaf_b.length, 2.0)

    def test_distribute_path_weights_ladder(self):
        """Test mass conservation over a deep nested path (Ladder)."""
        # Tree: (((A:1.0,B:1.0):1.0,C:2.0):1.0,D:3.0);
        tree = parse_newick("(((A:1.0,B:1.0):1.0,C:2.0):1.0,D:3.0);")

        # All internal splits including root for full conservation test
        internal_splits = [nd.split_indices for nd in tree.traverse() if nd.children]

        # Measure starting radial distances
        def get_radial(name):
            leaf_node = [leaf for leaf in tree.get_leaves() if leaf.name == name][0]
            dist = 0.0
            curr = leaf_node
            while curr:
                dist += curr.length or 0.0
                curr = curr.parent
            return dist

        start_a = get_radial("A")
        start_c = get_radial("C")

        distribute_path_weights(tree, internal_splits, operation="add")

        # Set internals to 0
        for nd in tree.traverse():
            if nd.children:
                nd.length = 0.0

        end_a = get_radial("A")
        end_c = get_radial("C")

        # Radial distances should be invariant
        self.assertAlmostEqual(end_a, start_a)
        self.assertAlmostEqual(end_c, start_c)

    def test_get_virtual_collapse_weights(self):
        """Test virtual weight calculation for collapsed state."""
        # Tree: (A:1, (B:1, C:1):5):0;
        tree = parse_newick("(A:1.0,(B:1.0,C:1.0):5.0):0.0;")

        # Node for (B,C)
        node_bc = [
            nd for nd in tree.traverse() if nd.children and len(nd.get_leaves()) == 2
        ][0]
        split_bc = node_bc.split_indices

        virtual_weights = get_virtual_collapse_weights(tree, [split_bc])

        # B virtual = 1 (pendant) + 5 (internal) = 6
        # C virtual = 1 (pendant) + 5 (internal) = 6
        # A virtual = 1 (unchanged)
        # Split BC virtual = 0

        # Find leaf partitions
        leaf_b_part = [leaf for leaf in tree.get_leaves() if leaf.name == "B"][
            0
        ].split_indices
        leaf_c_part = [leaf for leaf in tree.get_leaves() if leaf.name == "C"][
            0
        ].split_indices
        leaf_a_part = [leaf for leaf in tree.get_leaves() if leaf.name == "A"][
            0
        ].split_indices

        self.assertEqual(virtual_weights[leaf_b_part], 6.0)
        self.assertEqual(virtual_weights[leaf_c_part], 6.0)
        self.assertEqual(virtual_weights[leaf_a_part], 1.0)
        self.assertEqual(virtual_weights[split_bc], 0.0)

    def test_full_trace_integration(self):
        """Test the full 5-microstep lifecycle for patristic conservation."""
        from brancharchitect.tree_interpolation.subtree_paths.execution.step_executor import (
            build_microsteps_for_selection,
        )

        # Source: (A, (B,C))
        source = parse_newick("(A:1.0,(B:1.0,C:1.0):1.0):0.0;")
        # Dest: ((A,B), C)
        dest = parse_newick("((A:1.0,B:1.0):1.0,C:1.0):0.0;")

        # Pivot edge is the one that contains all taxa (root)
        pivot_edge = source.split_indices

        # Selection for B moving
        # In this transition, B moves from (B,C) clade to (A,B) clade.
        # This is a small enough change to trace.

        # We'll use a simplified selection for testing
        # B's collapse path in source: (B,C) split
        # B's expand path in dest: (A,B) split

        node_bc_src = [
            nd for nd in source.traverse() if nd.children and len(nd.get_leaves()) == 2
        ][0]
        node_ab_dest = [
            nd for nd in dest.traverse() if nd.children and len(nd.get_leaves()) == 2
        ][0]

        selection = {
            "subtree": [l for l in source.get_leaves() if l.name == "B"][
                0
            ].split_indices,
            "collapse": {"path_segment": [node_bc_src.split_indices]},
            "expand": {"path_segment": [node_ab_dest.split_indices]},
        }

        trees, edges, final_state, subtrees = build_microsteps_for_selection(
            interpolation_state=source,
            destination_tree=dest,
            current_pivot_edge=pivot_edge,
            selection=selection,
            source_tree=source,
            step_progress=0.5,
        )

        def get_radial(tree, name):
            target = [leaf for leaf in tree.get_leaves() if leaf.name == name][0]
            dist = 0.0
            curr = target
            while curr:
                dist += curr.length or 0.0
                curr = curr.parent
            return dist

        # Trace leaf B across steps
        # Step 0: Original Source (dist = 1.0 + 1.0 = 2.0)
        # Step 1: IT_Down (dist = 2.0 - internal 1.0 removed, but pendant increased by 1.0 = 2.0)
        # Step 2: Collapsed (dist = 2.0)
        # Step 3: Reordered (dist = 2.0)
        # Step 4: Grafted (dist = 2.0)
        # Step 5: Snapped (Averaging happens. B virtual source=2.0, B virtual dest=2.0. Mean=2.0)

        # print("TRACE DISTANCES FOR B:")
        for i, t in enumerate(trees):
            d = get_radial(t, "B")
            # print(f"Step {i}: {d}")
            self.assertAlmostEqual(d, 2.0, places=5)

        # Final state check
        self.assertAlmostEqual(get_radial(final_state, "B"), 2.0)

if __name__ == "__main__":
    unittest.main()
