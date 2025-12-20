from typing import List, Tuple, Dict, Any, Optional
import logging

from brancharchitect.tree import Node
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.elements.partition import Partition
from brancharchitect.leaforder.rotation_functions import optimize_splits
from brancharchitect.leaforder.split_analysis import (
    get_unique_splits,
    get_active_changing_splits,
    clear_split_pair_cache,
    get_common_splits,
)
from brancharchitect.leaforder.pairwise_alignment import final_pairwise_alignment_pass
from brancharchitect.leaforder.anchor_order import derive_order_for_pair
from brancharchitect.leaforder.tree_order_utils import build_orientation_map
from brancharchitect.leaforder.tree_order_utils import reorder_tree_if_full_common


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
        precomputed_active_changing_splits: Optional[
            List[Optional[PartitionSet[Partition]]]
        ] = None,
        precomputed_pair_solutions: Optional[
            List[Optional[Dict[Partition, List[Partition]]]]
        ] = None,
    ):
        """
        Initialize the optimizer with a list of trees.

        Args:
            trees (List[Node]): The trees to optimize.
        """
        self.trees: List[Node] = trees
        self.precomputed_active_changing_splits = (
            precomputed_active_changing_splits or []
        )
        self.precomputed_pair_solutions = precomputed_pair_solutions or []
        self.split_rotation_history: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._history_counter = 0
        self.logger = logging.getLogger(__name__)

    def _optimize_tree_pair_splits(
        self, pair_index: int, destination_tree: Node, source_tree: Node
    ) -> Tuple[bool, PartitionSet[Partition]]:
        """
        Optimize the source tree by performing local split operations based on the destination tree.
        """
        rotated_splits_in_source: PartitionSet[Partition] = PartitionSet()
        destination_order = destination_tree.get_current_order()

        if (
            pair_index < len(self.precomputed_active_changing_splits)
            and self.precomputed_active_changing_splits[pair_index] is not None
        ):
            s_edge_splits: PartitionSet[Partition] = (
                self.precomputed_active_changing_splits[pair_index]
            )  # type: ignore[assignment]
            self.logger.debug(
                f"Using precomputed s-edges for pair {pair_index}: {len(s_edge_splits)} edges"
            )
        else:
            s_edge_splits = get_active_changing_splits(destination_tree, source_tree)

            self.logger.debug(
                f"Computed s-edges for pair {pair_index}: {len(s_edge_splits)} edges"
            )

        improved_sedge: bool = optimize_splits(
            source_tree, s_edge_splits, destination_order, rotated_splits_in_source
        )

        unique_splits: PartitionSet[Partition] = get_unique_splits(
            destination_tree, source_tree
        )

        improved_unique: bool = optimize_splits(
            source_tree, unique_splits, destination_order, rotated_splits_in_source
        )

        return (improved_unique or improved_sedge), rotated_splits_in_source

    def _run_optimization_pass(self, direction: str) -> bool:
        """Run a single optimization pass in the specified direction."""
        improved_any = False
        n = len(self.trees)

        if direction == "forward":
            for i in range(n - 1):
                improved, splits_rotated = self._optimize_tree_pair_splits(
                    i, self.trees[i], self.trees[i + 1]
                )

                if improved:
                    improved_any = True
                    self.split_rotation_history[(self._history_counter, i)] = {
                        "improved": improved,
                        "splits_rotated": splits_rotated,
                        "direction": direction,
                    }
                    self._history_counter += 1

                    clear_split_pair_cache()

        elif direction == "backward":
            for i in range(n - 1, 0, -1):
                improved, splits_rotated = self._optimize_tree_pair_splits(
                    i - 1, self.trees[i], self.trees[i - 1]
                )
                if improved:
                    improved_any = True
                    self.split_rotation_history[(self._history_counter, i)] = {
                        "improved": improved,
                        "splits_rotated": splits_rotated,
                        "direction": direction,
                    }
                    self._history_counter += 1
                    clear_split_pair_cache()
        else:
            raise ValueError(f"Unknown direction: {direction}")

        return improved_any

    def _optimize_forward(self) -> bool:
        """Perform a forward optimization pass."""
        return self._run_optimization_pass("forward")

    def _optimize_backward(self) -> bool:
        """Perform a backward optimization pass."""
        return self._run_optimization_pass("backward")

    def optimize(self, n_iterations: int = 3, bidirectional: bool = False) -> None:
        """
        Main entry point for optimization. Repeatedly runs optimization passes.

        Args:
            n_iterations (int): Number of optimization iterations to perform.
            bidirectional (bool): If True, enables a more powerful optimization
                strategy with global change propagation. If False, uses a
                simpler, faster forward-only approach.
        """
        self.logger.info(
            f"Starting tree order optimization with {n_iterations} iterations, bidirectional={bidirectional}"
        )
        self.logger.info(f"Optimizing {len(self.trees)} trees")

        # NOTE: No encoding synchronization needed here anymore.
        # Trees are re-parsed after rooting in the pipeline, ensuring
        # all trees start with consistent encodings from the same parse batch.

        for iteration in range(n_iterations):
            self.logger.info(
                f"Starting optimization iteration {iteration + 1}/{n_iterations}"
            )
            forward_improved = self._optimize_forward()

            backward_improved = False

            if bidirectional:
                backward_improved = self._optimize_backward()
                self.logger.debug(f"Backward pass improved: {backward_improved}")

            if not (forward_improved or backward_improved):
                self.logger.info(
                    f"No improvements found in iteration {iteration + 1}, stopping early"
                )
                break

        # Final alignment to ensure global coherence

        final_pairwise_alignment_pass(self.trees)
        self.logger.info("Tree order optimization completed (final alignment applied)")

    def optimize_with_anchor_ordering(
        self,
        anchor_weight_policy: str = "destination",
        circular: bool = False,
        circular_boundary_policy: str = "between_anchor_blocks",
    ) -> None:
        """
        Optimizes tree ordering using the anchor-based lattice algorithm.

        This method provides a deterministic, non-iterative solution that:
        - Uses the lattice algorithm to identify moving taxa between tree pairs
        - Directly computes optimal leaf orderings based on topological analysis
        - Applies reordering to minimize crossings when displaying trees side-by-side

        Unlike the rotation-based `optimize()` method:
        - No iterations are needed (the solution is computed analytically)
        - No forward/backward passes required
        - More powerful for handling significant topological differences
        - May be slower due to lattice algorithm computation

        The process:
        1. For each consecutive tree pair (trees[i], trees[i+1]):
           - Run lattice algorithm to find moving partitions
           - Compute optimal orderings for both trees
           - Apply reordering directly to the nodes
        2. Apply final pairwise alignment pass for global coherence
        """
        self.logger.info("Starting anchor-based tree order optimization")
        self.logger.info(f"Optimizing {len(self.trees)} trees")

        n = len(self.trees)

        # NOTE: No encoding synchronization needed here anymore.
        # Trees are re-parsed after rooting in the pipeline, ensuring
        # all trees start with consistent encodings from the same parse batch.

        # Apply anchor-based ordering to each consecutive pair
        for i in range(n - 1):
            self.logger.info(f"Processing tree pair ({i}, {i + 1})")

            # Retrieve precomputed solution if available
            precomputed_solution = None
            if self.precomputed_pair_solutions and i < len(
                self.precomputed_pair_solutions
            ):
                precomputed_solution = self.precomputed_pair_solutions[i]

            # Default to destination-anchored ordering so only jumping taxa move
            derive_order_for_pair(
                self.trees[i],
                self.trees[i + 1],
                anchor_weight_policy=anchor_weight_policy,
                circular=circular,
                circular_boundary_policy=circular_boundary_policy,
                precomputed_solution=precomputed_solution,
            )

            # After ordering pair (i, i+1), their common splits are now aligned.
            # Propagate the orientation from tree i backwards and from tree i+1 forwards.
            common_splits = get_common_splits(self.trees[i], self.trees[i + 1])

            self._propagate_from_index_backward(i, common_splits)
            self._propagate_from_index_forward(i + 1, common_splits)

        # Final alignment to ensure global coherence
        final_pairwise_alignment_pass(self.trees)

        for idx, tree in enumerate(self.trees):
            self.logger.debug(
                f"Final order for tree {idx}: {tree.to_newick(lengths=False)}"
            )

        self.logger.info(
            "Anchor-based optimization completed (final alignment applied)"
        )

    def _propagate(
        self,
        ref_tree: Node,
        splits_to_propagate: PartitionSet[Partition],
        target_trees: List[Node],
    ) -> None:
        """Generic helper to propagate orientations to a list of target trees."""
        if not splits_to_propagate:
            return

        orientation_map = build_orientation_map(ref_tree, splits_to_propagate)

        for target_tree in target_trees:
            # The map is mutated in-place, so we create a copy for each target
            map_copy = orientation_map.copy()
            reorder_tree_if_full_common(ref_tree, target_tree, map_copy)

    def _propagate_from_index_forward(
        self, ref_index: int, splits_to_propagate: PartitionSet[Partition]
    ) -> None:
        """Propagate child orderings forward starting from `ref_index`."""
        if not splits_to_propagate or ref_index >= len(self.trees) - 1:
            return

        ref_tree: Node = self.trees[ref_index]
        target_trees: List[Node] = self.trees[ref_index + 1 :]

        self.logger.debug(
            "Forward propagating from tree %d to trees %s",
            ref_index,
            list(range(ref_index + 1, len(self.trees))),
        )
        self._propagate(ref_tree, splits_to_propagate, target_trees)

    def _propagate_from_index_backward(
        self, ref_index: int, splits_to_propagate: PartitionSet[Partition]
    ) -> None:
        """Propagate child orderings backward starting from `ref_index`."""
        if not splits_to_propagate or ref_index <= 0:
            return

        ref_tree: Node = self.trees[ref_index]
        target_trees: List[Node] = self.trees[:ref_index]
        # Reverse for logical propagation from near to far
        target_trees.reverse()

        self.logger.debug(
            "Backward propagating from tree %d to trees %s",
            ref_index,
            list(range(ref_index - 1, -1, -1)),
        )
        self._propagate(ref_tree, splits_to_propagate, target_trees)
