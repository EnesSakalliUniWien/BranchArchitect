"""Tree processing pipeline."""

from typing import List, Optional, Dict
import logging
import time
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.movie_pipeline.types import (
    PipelineConfig,
    InterpolationResult,
    create_empty_result,
    create_single_tree_result,
    DistanceMetrics,
    TreeMetadata,
)
from brancharchitect.leaforder.tree_order_optimiser import TreeOrderOptimizer
from brancharchitect.distances.distances import (
    calculate_along_trajectory,
    relative_robinson_foulds_distance,
    weighted_robinson_foulds_distance,
)
from brancharchitect.tree_interpolation.sequential_interpolation import (
    SequentialInterpolationBuilder,
)
from brancharchitect.jumping_taxa.lattice.compute_pivot_solutions_with_deletions import (
    compute_pivot_solutions_with_deletions,
)
from brancharchitect.tree_interpolation.types import TreeInterpolationSequence
from brancharchitect.tree import Node
from .tree_rooting import root_trees


class TreeInterpolationPipeline:
    """
    Coordinates the full workflow for processing and interpolating phylogenetic trees.

    This includes rooting, leaf order optimization, lattice-based interpolation,
    and distance metric calculation.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initializes the tree interpolation pipeline.

        Args:
            config: Pipeline configuration settings.
            logger: Logger instance for pipeline events.
        """
        self.config: PipelineConfig = config or PipelineConfig()
        self.logger = logger or logging.getLogger(self.config.logger_name)

    def process_trees(self, trees: Node | List[Node]) -> InterpolationResult:
        """
        Executes the complete tree interpolation pipeline.

        Args:
            trees: List of phylogenetic trees to process.

        Returns:
            An InterpolationResult object containing all interpolated trees,
            metadata, and analysis results.
        """
        start_time = time.time()

        processed_trees: Node | List[Node] = trees

        if isinstance(processed_trees, Node):
            processed_trees = [processed_trees]
        if not processed_trees:
            return create_empty_result()
        if len(processed_trees) == 1:
            processed_trees = self._apply_rooting_if_enabled(processed_trees)
            return create_single_tree_result(processed_trees)

        processed_trees = self._apply_rooting_if_enabled(processed_trees)
        precomputed_pair_solutions = self._precompute_pair_solutions(processed_trees)

        t_opt_start = time.perf_counter()
        processed_trees = self._optimize_tree_order(
            processed_trees,
            precomputed_pair_pivot_split_sets=self._extract_current_pivot_split_sets(
                precomputed_pair_solutions, processed_trees
            ),
        )
        self.logger.info(
            f"Leaf order optimization took {time.perf_counter() - t_opt_start:.3f}s"
        )

        seq_result = self._interpolate_tree_sequence(
            processed_trees, precomputed_pair_solutions=precomputed_pair_solutions
        )

        self.logger.info("Calculating distance metrics...")
        t_dist_start = time.perf_counter()
        distances = self._calculate_distances(processed_trees)
        self.logger.info(
            f"Distance metrics calculated in {time.perf_counter() - t_dist_start:.3f}s"
        )

        processing_time = time.time() - start_time
        self.logger.info(
            f"Processed {len(processed_trees)} trees in {processing_time:.2f} seconds"
        )

        return InterpolationResult(
            interpolated_trees=seq_result.interpolated_trees,
            tree_metadata=seq_result.tree_metadata,
            tree_pair_solutions=seq_result.tree_pair_solutions,
            rfd_list=distances.rfd_list,
            wrfd_list=distances.wrfd_list,
            processing_time=processing_time,
            pair_interpolation_ranges=seq_result.pair_interpolation_ranges,
        )

    # --- Private helpers ---

    def _interpolate_tree_sequence(
        self,
        trees: List[Node],
        precomputed_pair_solutions: Optional[
            List[Optional[Dict[Partition, List[Partition]]]]
        ] = None,
    ) -> TreeInterpolationSequence:
        """
        Orchestrates the interpolation between all consecutive tree pairs.
        """
        result: TreeInterpolationSequence = SequentialInterpolationBuilder(
            logger=self.logger,
            precomputed_pair_solutions=precomputed_pair_solutions,
        ).build(trees)

        original_tree_global_indices = result.get_original_tree_indices()
        pair_solutions, pair_ranges = result.build_pair_solutions(
            original_tree_global_indices, logger=self.logger
        )
        tree_metadata = self._create_global_tree_metadata(
            result.current_pivot_edge_tracking, original_tree_global_indices
        )

        # Attach derived fields directly to the sequence for downstream consumption
        result.tree_pair_solutions = pair_solutions  # type: ignore[attr-defined]
        result.pair_interpolation_ranges = pair_ranges  # type: ignore[attr-defined]
        result.tree_metadata = tree_metadata  # type: ignore[attr-defined]

        return result

    def _optimize_tree_order(
        self,
        trees: List[Node],
        precomputed_pair_pivot_split_sets: Optional[
            List[Optional[PartitionSet[Partition]]]
        ] = None,
    ) -> List[Node]:
        """
        Optimizes the leaf node order to minimize visual crossings.
        """
        if len(trees) <= 1:
            return trees

        optimizer = TreeOrderOptimizer(
            trees,
            precomputed_active_changing_splits=precomputed_pair_pivot_split_sets,
        )

        if self.config.use_anchor_ordering:
            self.logger.info("Using anchor-based ordering (lattice algorithm)")
            optimizer.optimize_with_anchor_ordering(
                anchor_weight_policy=self.config.anchor_weight_policy,
                circular=self.config.circular,
                circular_boundary_policy=self.config.circular_boundary_policy,
            )
        else:
            self.logger.info("Using rotation-based optimization")
            optimizer.optimize(
                n_iterations=self.config.optimization_iterations,
                bidirectional=self.config.bidirectional_optimization,
            )

        return trees

    def _precompute_pair_solutions(
        self, trees: List[Node]
    ) -> List[Optional[Dict[Partition, List[Partition]]]]:
        """
        Runs the lattice algorithm once for each adjacent pair of trees.
        """
        if len(trees) < 2:
            return []
        sols: List[Optional[Dict[Partition, List[Partition]]]] = []
        for i in range(len(trees) - 1):
            source_tree = trees[i]
            destination_tree = trees[i + 1]
            try:
                self.logger.info(f"Precomputing solution for pair {i}-{i + 1}...")
                solution_dict, _ = compute_pivot_solutions_with_deletions(
                    source_tree.deep_copy(), destination_tree.deep_copy()
                )
                sols.append(solution_dict)
            except Exception as e:
                self.logger.error(
                    f"Failed to compute lattice solution for pair {i}-{i + 1}. Error: {e}",
                    exc_info=True,
                )
                sols.append(None)
        return sols

    def _extract_current_pivot_split_sets(
        self,
        precomputed_pair_solutions: List[Optional[Dict[Partition, List[Partition]]]],
        trees: List[Node],
    ) -> List[Optional[PartitionSet[Partition]]]:
        """
        Extracts current pivot split sets from precomputed lattice solutions.
        """
        if not precomputed_pair_solutions:
            return []

        split_sets: List[Optional[PartitionSet[Partition]]] = []
        for solution in precomputed_pair_solutions:
            if solution is None:
                split_sets.append(None)
            else:
                active_changing_splits_list: List[Partition] = list(solution.keys())
                split_set: PartitionSet[Partition] = PartitionSet(
                    set(active_changing_splits_list)
                )
                split_sets.append(split_set)

        return split_sets

    def _calculate_distances(self, trees: List[Node]) -> DistanceMetrics:
        """
        Calculates Robinson-Foulds distances between consecutive trees.
        """
        if len(trees) < 2:
            return DistanceMetrics(rfd_list=[0.0], wrfd_list=[0.0])

        rfd_list: List[float] = calculate_along_trajectory(
            trees, relative_robinson_foulds_distance
        )
        wrfd_list: List[float] = calculate_along_trajectory(
            trees, weighted_robinson_foulds_distance
        )

        return DistanceMetrics(rfd_list=rfd_list, wrfd_list=wrfd_list)

    def _create_global_tree_metadata(
        self,
        current_pivot_edge_tracking: List[Optional[Partition]],
        original_tree_global_indices: List[int],
    ) -> List[TreeMetadata]:
        """
        Generate global metadata for each tree in the sequence.
        """
        tree_metadata: List[TreeMetadata] = []
        original_index_iter = iter(original_tree_global_indices)
        next_original_idx = next(original_index_iter, None)
        current_pair_index = 0
        source_global_idx = 0
        interpolation_step = 0

        for idx, pivot_edge in enumerate(current_pivot_edge_tracking):
            is_original = pivot_edge is None

            if is_original and next_original_idx == idx:
                # This is an original tree
                source_global_idx = idx
                next_original_idx = next(original_index_iter, None)
                tree_metadata.append(
                    TreeMetadata(
                        tree_pair_key=None,
                        step_in_pair=None,
                        source_tree_global_index=None,
                    )
                )
                interpolation_step = 0
                continue

            # This is an interpolated tree
            interpolation_step += 1
            tree_pair_key = f"pair_{current_pair_index}_{current_pair_index + 1}"

            tree_metadata.append(
                TreeMetadata(
                    tree_pair_key=tree_pair_key,
                    step_in_pair=interpolation_step,
                    source_tree_global_index=source_global_idx,
                )
            )

            # Check if we've completed this pair (next tree is original)
            if (
                idx + 1 < len(current_pivot_edge_tracking)
                and current_pivot_edge_tracking[idx + 1] is None
            ):
                current_pair_index += 1
                interpolation_step = 0

        return tree_metadata

    def _apply_rooting_if_enabled(self, trees: List[Node]) -> List[Node]:
        """
        Apply rooting to trees if enabled in the configuration.
        """
        if not self.config.enable_rooting:
            return trees

        try:
            self.logger.info("Applying midpoint rooting...")
            t_root_start = time.perf_counter()
            rooted_trees = root_trees(trees)
            self.logger.info(
                f"Rooting completed in {time.perf_counter() - t_root_start:.3f}s"
            )
            return rooted_trees
        except Exception as e:
            self.logger.error(f"Rooting failed: {e}")
            return trees
