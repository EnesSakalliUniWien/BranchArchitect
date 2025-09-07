from typing import List, Tuple, Dict, FrozenSet, Any, Optional
import logging

# Assuming these imports point to valid modules in your project structure
from brancharchitect.tree import Node
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.elements.partition import Partition
from brancharchitect.leaforder.rotation_functions import (
    get_unique_splits,
    get_s_edge_splits,
    optimize_splits,
    clear_split_pair_cache,
)
from brancharchitect.leaforder.tree_order_utils import (
    build_orientation_map,
    reorder_tree_if_full_common,
)
from brancharchitect.leaforder.final_alignment import final_pairwise_alignment_pass


# Correctly defined type alias
SplitChildOrderMap = Dict[Partition, List[FrozenSet[str]]]


# final_pairwise_alignment_pass moved to brancharchitect.leaforder.final_alignment


class TreeOrderOptimizer:
    """
    Optimizes the leaf order of a list of phylogenetic trees using local and
    propagated split operations. The primary entry point is the `optimize` method.

    Attributes:
        trees (List[Node]): The list of trees to optimize.
    """

    def __init__(
        self,
        trees: List[Node],
        precomputed_s_edges: Optional[List[Optional[PartitionSet[Partition]]]] = None,
    ):
        """
        Initialize the optimizer with a list of trees.

        Args:
            trees (List[Node]): The trees to optimize.
        """
        self.trees: List[Node] = trees
        self.precomputed_s_edges = precomputed_s_edges or []
        self.split_rotation_history: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._history_counter = 0
        self.logger = logging.getLogger(__name__)

    # Removed: apply_fiedler_ordering (Fiedler spectral ordering deprecated)

    def _optimize_tree_pair_splits(
        self, pair_index: int, reference_tree: Node, target_tree: Node
    ) -> Tuple[bool, PartitionSet[Partition]]:
        """
        Optimize the target tree by performing local split operations based on the reference tree.
        """
        rotated_splits_in_target: PartitionSet[Partition] = PartitionSet()
        ref_order = reference_tree.get_current_order()

        if (
            pair_index < len(self.precomputed_s_edges)
            and self.precomputed_s_edges[pair_index] is not None
        ):
            s_edge_splits: PartitionSet[Partition] = self.precomputed_s_edges[
                pair_index
            ]  # type: ignore[assignment]
            self.logger.debug(f"Using precomputed s-edges for pair {pair_index}: {len(s_edge_splits)} edges")
        else:
            s_edge_splits = get_s_edge_splits(reference_tree, target_tree)
            self.logger.debug(f"Computed s-edges for pair {pair_index}: {len(s_edge_splits)} edges")
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

    def _propagate_from_index_forward(
        self, ref_index: int, splits_to_propagate: PartitionSet[Partition]
    ) -> None:
        """Propagate child orderings forward starting from `ref_index`.

        Uses `self.trees[ref_index]` as the reference orientation and applies
        to trees at indices ref_index+1 .. end where subtrees are full-common.
        """
        if not splits_to_propagate or ref_index >= len(self.trees) - 1:
            return

        ref_tree: Node = self.trees[ref_index]
        split_child_order_map = build_orientation_map(ref_tree, splits_to_propagate)

        for j in range(ref_index + 1, len(self.trees)):
            split_child_order_map = reorder_tree_if_full_common(
                ref_tree, self.trees[j], split_child_order_map
            )
            if not split_child_order_map:
                break

    def _propagate_from_index_backward(
        self, ref_index: int, splits_to_propagate: PartitionSet[Partition]
    ) -> None:
        """Propagate child orderings backward starting from `ref_index`.

        Uses `self.trees[ref_index]` as the reference orientation and applies
        to trees at indices ref_index-1 down to 0 where subtrees are full-common.
        """
        if not splits_to_propagate or ref_index <= 0:
            return

        ref_tree: Node = self.trees[ref_index]
        split_child_order_map = build_orientation_map(ref_tree, splits_to_propagate)

        for j in range(ref_index - 1, -1, -1):
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
                    i, self.trees[i], self.trees[i + 1]
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
                        # Modified tree is at index i+1. Propagate from that index.
                        self._propagate_from_index_forward(i + 1, splits_rotated)
                        self._propagate_from_index_backward(i + 1, splits_rotated)
                    else:
                        # Standard mode: only propagate forward from the modified tree (i+1).
                        self._propagate_from_index_forward(i + 1, splits_rotated)
                    clear_split_pair_cache()
        elif direction == "backward":
            for i in range(n - 1, 0, -1):
                improved, splits_rotated = self._optimize_tree_pair_splits(
                    i - 1, self.trees[i], self.trees[i - 1]
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
                    # The modified tree is at index i-1. Propagate from that index.
                    self._propagate_from_index_backward(i - 1, splits_rotated)
                    self._propagate_from_index_forward(i - 1, splits_rotated)
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
        self.logger.info(f"Starting tree order optimization with {n_iterations} iterations, bidirectional={bidirectional}")
        self.logger.info(f"Optimizing {len(self.trees)} trees")
        
        for iteration in range(n_iterations):
            self.logger.info(f"Starting optimization iteration {iteration + 1}/{n_iterations}")
            forward_impr = self._optimize_forward(bidirectional_mode=bidirectional)

            back_impr = False
            if bidirectional:
                back_impr = self._optimize_backward()
                self.logger.debug(f"Backward pass improved: {back_impr}")

            if not (forward_impr or back_impr):
                self.logger.info(f"No improvements found in iteration {iteration + 1}, stopping early")
                break

            # Align adjacent pairs at the end of each improving iteration
            final_pairwise_alignment_pass(self.trees)
            self.logger.info(f"Completed iteration {iteration + 1} with improvements")
        
        # Unconditional final alignment to ensure global coherence
        final_pairwise_alignment_pass(self.trees)
        self.logger.info("Tree order optimization completed (final alignment applied)")
