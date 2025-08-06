from typing import List, Tuple, Dict, FrozenSet, Any

# Assuming these imports point to valid modules in your project structure
from brancharchitect.tree import Node
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.elements.partition import Partition
from brancharchitect.leaforder.rotation_functions import (
    get_unique_splits,
    get_s_edge_splits,
    get_common_splits,
    optimize_splits,
    clear_split_pair_cache,
)
from brancharchitect.leaforder.fieldler_order import (
    fiedler_ordering_for_tree_pair,
)
from brancharchitect.leaforder.tree_order_utils import (
    build_orientation_map,
    reorder_tree_if_full_common,
)


# Correctly defined type alias
SplitChildOrderMap = Dict[Partition, List[FrozenSet[str]]]


def final_pairwise_alignment_pass(trees: List[Node]) -> None:
    """
    Performs a final alignment pass on a list of trees to ensure that for
    any adjacent pair, subtrees that are topologically identical are also
    oriented identically.

    This function iterates through the tree sequence and, for each pair, uses
    the first tree as a template to orient the common splits in the second tree.

    Args:
        trees: The list of optimized phylogenetic trees to be aligned. This list is modified in-place.
    """
    if len(trees) < 2:
        return  # No pairs to align

    for i in range(len(trees) - 1):
        tree1 = trees[i]
        tree2 = trees[i + 1]

        # Find splits that are common to this specific adjacent pair
        common_splits_pair = get_common_splits(tree1, tree2)

        if not common_splits_pair:
            # If there are no common splits, there's nothing to align for this pair
            continue

        # Use tree1 as the template to get the desired orientation
        orientation_map = build_orientation_map(tree1, common_splits_pair)

        # Apply this orientation to tree2, but only for subtrees that are
        # topologically identical ("full-common")
        reorder_tree_if_full_common(tree1, tree2, orientation_map)

    # Clear any cached split information after modifications are complete
    clear_split_pair_cache()


class TreeOrderOptimizer:
    """
    Optimizes the leaf order of a list of phylogenetic trees using local and
    propagated split operations. The primary entry point is the `optimize` method.

    Attributes:
        trees (List[Node]): The list of trees to optimize.
    """

    def __init__(self, trees: List[Node]):
        """
        Initialize the optimizer with a list of trees.

        Args:
            trees (List[Node]): The trees to optimize.
        """
        self.trees: List[Node] = trees
        self.split_rotation_history: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._history_counter = 0

    def apply_fiedler_ordering(self):
        """
        Apply consensus Fiedler (spectral) ordering to all trees in self.trees.
        """
        fiedler_ordering_for_tree_pair(self.trees)
        clear_split_pair_cache()

    def _optimize_tree_pair_splits(
        self, reference_tree: Node, target_tree: Node
    ) -> Tuple[bool, PartitionSet[Partition]]:
        """
        Optimize the target tree by performing local split operations based on the reference tree.
        """
        rotated_splits_in_target: PartitionSet[Partition] = PartitionSet()
        ref_order = reference_tree.get_current_order()

        s_edge_splits: PartitionSet[Partition] = get_s_edge_splits(
            reference_tree, target_tree
        )
        improved_sedge: bool = optimize_splits(
            target_tree, s_edge_splits, ref_order, rotated_splits_in_target
        )

        unique_splits: PartitionSet[Partition] = get_unique_splits(
            reference_tree, target_tree
        )
        improved_unique: bool = optimize_splits(
            target_tree, unique_splits, ref_order, rotated_splits_in_target
        )

        return (improved_unique or improved_sedge), rotated_splits_in_target

    def _propagate_child_order_forward(
        self, reference_index: int, splits_to_propagate: PartitionSet[Partition]
    ) -> None:
        """Propagate child orderings forward from tree at `reference_index + 1`."""
        if not splits_to_propagate or reference_index >= len(self.trees) - 1:
            return

        ref_tree: Node = self.trees[reference_index + 1]
        split_child_order_map = build_orientation_map(ref_tree, splits_to_propagate)

        for j in range(reference_index + 2, len(self.trees)):
            split_child_order_map = reorder_tree_if_full_common(
                ref_tree, self.trees[j], split_child_order_map
            )
            if not split_child_order_map:
                break

    def _propagate_child_order_backward(
        self, reference_index: int, splits_to_propagate: PartitionSet[Partition]
    ) -> None:
        """Propagate child orderings backward from tree at `reference_index`."""
        if not splits_to_propagate or reference_index <= 0:
            return

        ref_tree: Node = self.trees[reference_index]
        split_child_order_map = build_orientation_map(ref_tree, splits_to_propagate)

        for j in range(reference_index - 1, -1, -1):
            split_child_order_map = reorder_tree_if_full_common(
                ref_tree, self.trees[j], split_child_order_map
            )
            if not split_child_order_map:
                break

    def _run_optimization_pass(self, direction: str, bidirectional_mode: bool) -> bool:
        """
        [CORRECTED] Run a single optimization pass. This version uses the correct
        indexing to ensure propagation always originates from the modified tree.
        """
        improved_any = False
        n = len(self.trees)
        if direction == "forward":
            for i in range(n - 1):
                improved, splits_rotated = self._optimize_tree_pair_splits(
                    self.trees[i], self.trees[i + 1]
                )
                if improved:
                    improved_any = True
                    self.split_rotation_history[
                        (self._history_counter, i)
                    ] = {  # Store with a unique key
                        "improved": improved,
                        "splits_rotated": splits_rotated,
                        "direction": direction,
                    }
                    self._history_counter += 1
                    if bidirectional_mode:
                        # Modified tree is at i+1. Propagate from it.
                        # Forward propagation helper needs index i to use tree i+1 as ref.
                        self._propagate_child_order_forward(i, splits_rotated)
                        # Backward propagation helper needs index i+1 to use tree i+1 as ref.
                        self._propagate_child_order_backward(i + 1, splits_rotated)
                    else:
                        # Standard mode: only propagate forward from the modified tree.
                        self._propagate_child_order_forward(i, splits_rotated)
                    clear_split_pair_cache()
        elif direction == "backward":
            for i in range(n - 1, 0, -1):
                improved, splits_rotated = self._optimize_tree_pair_splits(
                    self.trees[i], self.trees[i - 1]
                )
                if improved:
                    improved_any = True
                    self.split_rotation_history[
                        (self._history_counter, i)
                    ] = {  # Store with a unique key
                        "improved": improved,
                        "splits_rotated": splits_rotated,
                        "direction": direction,
                    }
                    self._history_counter += 1
                    # A backward pass only runs in bidirectional mode, which demands
                    # full global propagation from the modified tree.
                    # The modified tree is at i-1.

                    # Propagate backward from i-1. Helper needs index i-1.
                    self._propagate_child_order_backward(i - 1, splits_rotated)
                    # Propagate forward from i-1. Helper needs index i-2.
                    self._propagate_child_order_forward(i - 2, splits_rotated)
                    clear_split_pair_cache()
        else:
            raise ValueError(f"Unknown direction: {direction}")
        return improved_any

    def _optimize_forward(self, bidirectional_mode: bool) -> bool:
        """Perform a forward pass, passing on the mode flag."""
        return self._run_optimization_pass("forward", bidirectional_mode)

    def _optimize_backward(self) -> bool:
        """Perform a backward pass, which always assumes bidirectional mode."""
        return self._run_optimization_pass("backward", bidirectional_mode=True)

    def optimize(self, n_iterations: int = 3, bidirectional: bool = False) -> None:
        """
        Main entry point for optimization. Repeatedly runs optimization passes.

        Args:
            n_iterations (int): Number of optimization iterations to perform.
            bidirectional (bool): If True, enables a more powerful optimization
                strategy with global change propagation. If False, uses a
                simpler, faster forward-only approach.
        """
        for _ in range(n_iterations):
            forward_impr = self._optimize_forward(bidirectional_mode=bidirectional)

            back_impr = False
            if bidirectional:
                back_impr = self._optimize_backward()

            if not (forward_impr or back_impr):
                break

            final_pairwise_alignment_pass(self.trees)
