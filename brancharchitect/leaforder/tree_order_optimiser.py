import logging
from typing import List, Tuple, Dict
from brancharchitect.tree import Node
from brancharchitect.partition_set import PartitionSet
from brancharchitect.leaforder.old.tree_order_optimisation_local import (
    build_orientation_map,
    reorder_tree_if_full_common,
)
from brancharchitect.leaforder.rotation_functions import (
    get_unique_splits,
    get_s_edge_splits,
    optimize_splits,
)
from brancharchitect.leaforder.fieldler_order import (
    fiedler_ordering_for_tree_pair,
)

SplitChildOrderMap = Dict  # Dict[Partition, List[frozenset]]


class TreeOrderOptimizer:
    """
    Optimizes the order of a list of phylogenetic trees using local and propagated split operations.

    Attributes:
        trees (List[Node]): The list of trees to optimize.
        split_rotation_history (dict): Tracks which splits were rotated and whether improvement occurred.

    Example:
        >>> optimizer = TreeOrderOptimizer(trees)
        >>> optimizer.optimize(n_iterations=5, bidirectional=True)
    """

    def __init__(self, trees: List[Node]):
        """
        Initialize the optimizer with a list of trees.

        Args:
            trees (List[Node]): The trees to optimize.
        """
        self.trees: List[Node] = trees
        self.split_rotation_history: dict = dict()

    def apply_fiedler_ordering(self):
        """
        Apply consensus Fiedler (spectral) ordering to all trees in self.trees.
        This reorders internal nodes for a good initial layout using the consensus Fiedler vector.
        """
        fiedler_ordering_for_tree_pair(self.trees)

    def _optimize_tree_pair_splits(
        self, reference_tree: Node, target_tree: Node
    ) -> Tuple[bool, PartitionSet]:
        """
        Optimize the target tree by performing local split operations based on the reference tree.
        Operates on unique splits and s-edge splits, as determined by helper functions.

        Args:
            reference_tree (Node): The reference tree for comparison.
            target_tree (Node): The tree to optimize.

        Returns:
            Tuple[bool, PartitionSet]: (improved_any, splits_rotated_in_target)
        """
        rotated_splits_in_target: PartitionSet = PartitionSet()
        ref_order = reference_tree.get_current_order()

        # Optimize unique splits
        unique_splits = get_unique_splits(reference_tree, target_tree)
        improved_unique = optimize_splits(
            target_tree, unique_splits, ref_order, rotated_splits_in_target
        )

        # Optimize s-edge splits
        s_edge_splits = get_s_edge_splits(reference_tree, target_tree)

        improved_sedge = optimize_splits(
            target_tree, s_edge_splits, ref_order, rotated_splits_in_target
        )

        return (improved_unique or improved_sedge), rotated_splits_in_target

    def _propagate_child_order_forward(
        self, reference_index: int, splits_to_propagate: PartitionSet
    ) -> None:
        """
        Propagate the child orderings for the given splits forward through the tree list.

        Args:
            reference_index (int): Index of the reference tree.
            splits_to_propagate (PartitionSet): Splits whose child orderings are to be propagated.

        Side Effects:
            Mutates the order of child nodes in subsequent trees if the split is present and full-common.
        """

        if reference_index < 0 or reference_index >= len(self.trees) - 1:
            return

        ref_tree = self.trees[reference_index + 1]
        split_child_order_map = build_orientation_map(ref_tree, splits_to_propagate)

        for j in range(reference_index + 2, len(self.trees)):
            split_child_order_map = reorder_tree_if_full_common(
                ref_tree, self.trees[j], split_child_order_map
            )
            if not split_child_order_map:
                break

    def _propagate_child_order_backward(
        self, reference_index: int, splits_to_propagate: PartitionSet
    ) -> None:
        """
        Propagate the child orderings for the given splits backward through the tree list.

        Args:
            reference_index (int): Index of the reference tree.
            splits_to_propagate (PartitionSet): Splits whose child orderings are to be propagated.

        Side Effects:
            Mutates the order of child nodes in previous trees if the split is present and full-common.
        """
        if reference_index <= 0 or reference_index >= len(self.trees):
            return
        ref_tree = self.trees[reference_index]

        split_child_order_map = build_orientation_map(ref_tree, splits_to_propagate)
        for j in range(reference_index - 1, -1, -1):
            split_child_order_map = reorder_tree_if_full_common(
                ref_tree, self.trees[j], split_child_order_map
            )
            if not split_child_order_map:
                break

    def _run_optimization_pass(self, direction: str) -> bool:
        """
        Run a single optimization pass in the specified direction ('forward' or 'backward').
        Returns True if any improvement was made.

        Args:
            direction (str): 'forward' or 'backward'.

        Returns:
            bool: True if any improvement was made, False otherwise.

        Raises:
            ValueError: If direction is not 'forward' or 'backward'.
        """
        improved_any = False
        n = len(self.trees)
        if direction == "forward":
            for reference_index in range(n - 1):
                improved, splits_rotated = self._optimize_tree_pair_splits(
                    self.trees[reference_index], self.trees[reference_index + 1]
                )

                self.split_rotation_history[(reference_index, reference_index + 1)] = {
                    "splits_rotated": splits_rotated,
                    "improved": improved,
                }

                if improved:
                    improved_any = True
                    self._propagate_child_order_forward(reference_index, splits_rotated)
        elif direction == "backward":
            for reference_index in range(n - 1, 0, -1):
                improved, splits_rotated = self._optimize_tree_pair_splits(
                    self.trees[reference_index], self.trees[reference_index - 1]
                )
                if improved:
                    improved_any = True
                    self._propagate_child_order_backward(
                        reference_index, splits_rotated
                    )
        else:
            raise ValueError(f"Unknown direction: {direction}")
        return improved_any

    def optimize_forward(self) -> bool:
        """
        Perform a forward optimization pass (left-to-right) over the tree list.

        Returns:
            bool: True if any improvement was made, False otherwise.
        """
        result = self._run_optimization_pass("forward")
        return result

    def optimize_backward(self) -> bool:
        """
        Perform a backward optimization pass (right-to-left) over the tree list.

        Returns:
            bool: True if any improvement was made, False otherwise.
        """
        result = self._run_optimization_pass("backward")
        return result

    def optimize_bidirectional(self, n_iterations: int = 3) -> None:
        """
        Run both forward and backward optimization passes for several iterations.
        Each iteration consists of forward, backward, then reversed forward and backward passes.

        Args:
            n_iterations (int): Number of bidirectional optimization iterations to perform.
        """
        for i in range(n_iterations):
            self.optimize_forward()
            self.optimize_backward()
            self.trees.reverse()
            self.optimize_forward()
            self.optimize_backward()
            self.trees.reverse()

    def optimize(self, n_iterations: int = 3, bidirectional: bool = False) -> None:
        """
        Main entry point for optimization. Repeatedly runs forward passes, and if bidirectional=True, also runs reverse passes.
        Stops early if no improvement is made in an iteration.

        Args:
            n_iterations (int): Number of optimization iterations to perform.
            bidirectional (bool): Whether to use bidirectional optimization.
        """
        for i in range(n_iterations):
            forward_impr = self.optimize_forward()
            back_impr = False
            if bidirectional:
                self.trees.reverse()
                back_impr = self.optimize_forward()
                self.trees.reverse()
            if not (forward_impr or back_impr):
                break
