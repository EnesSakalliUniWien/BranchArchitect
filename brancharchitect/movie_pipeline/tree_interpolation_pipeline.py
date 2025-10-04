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
    TreePairSolution,
)
from brancharchitect.leaforder.tree_order_optimiser import TreeOrderOptimizer
from brancharchitect.distances.distances import (
    calculate_along_trajectory,
    relative_robinson_foulds_distance,
    weighted_robinson_foulds_distance,
)
from brancharchitect.tree_interpolation.sequential_interpolation import (
    build_sequential_lattice_interpolations,
)
from brancharchitect.jumping_taxa.lattice.iterate_lattice_algorithm import (
    iterate_lattice_algorithm,
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

    def process_trees(self, trees: List[Node]) -> InterpolationResult:
        """
        Executes the complete tree interpolation pipeline.

        Args:
            trees: List of phylogenetic trees to process.

        Returns:
            An InterpolationResult object containing all interpolated trees,
            metadata, and analysis results.
        """
        start_time = time.time()

        # --- Taxa consistency check (before any processing) ---
        processed_trees: List[Node] = trees

        # Handle edge cases
        if not processed_trees:
            return create_empty_result()
        if len(processed_trees) == 1:
            processed_trees = self._apply_rooting_if_enabled(processed_trees)
            return create_single_tree_result(processed_trees)

        processed_trees = self._apply_rooting_if_enabled(processed_trees)
        # Precompute lattice solutions once per adjacent pair (topology-only; safe w.r.t. ordering)
        # These are used for interpolation and as initial s-edges for the first optimization iteration
        precomputed_pair_solutions = self._precompute_pair_solutions(processed_trees)

        t_opt_start = time.perf_counter()

        processed_trees = self._optimize_tree_order(
            processed_trees,
            precomputed_pair_active_changing_splits=self._extract_active_changing_split_sets(
                precomputed_pair_solutions, processed_trees
            ),
        )

        self.logger.info(
            f"Leaf order optimization took {time.perf_counter() - t_opt_start:.3f}s"
        )

        (
            interpolated_trees,
            tree_metadata,
            tree_pair_solutions,
            pair_interpolation_ranges,
        ) = self._interpolate_tree_sequence(
            processed_trees, precomputed_pair_solutions=precomputed_pair_solutions
        )

        self.logger.info("Calculating distance metrics...")
        t_dist_start = time.perf_counter()

        distances = self._calculate_distances(processed_trees)
        self.logger.info(
            f"Distance metrics calculated in {time.perf_counter() - t_dist_start:.3f}s"
        )

        # Build final result
        processing_time = time.time() - start_time

        self.logger.info(
            f"Processed {len(processed_trees)} trees in {processing_time:.2f} seconds"
        )

        return InterpolationResult(
            interpolated_trees=interpolated_trees,
            tree_metadata=tree_metadata,
            tree_pair_solutions=tree_pair_solutions,
            rfd_list=distances.rfd_list,
            wrfd_list=distances.wrfd_list,
            original_tree_count=len(processed_trees),
            interpolated_tree_count=len(interpolated_trees),
            processing_time=processing_time,
            pair_interpolation_ranges=pair_interpolation_ranges,
        )

    # --- Private helpers ---

    def _interpolate_tree_sequence(
        self,
        trees: List[Node],
        precomputed_pair_solutions: Optional[
            List[Optional[Dict[Partition, List[List[Partition]]]]]
        ] = None,
    ) -> tuple[
        List[Node],
        List[TreeMetadata],
        Dict[str, TreePairSolution],
        List[List[int]],
    ]:
        """
        Orchestrates the interpolation between all consecutive tree pairs.

        This method uses precomputed solutions to generate a full interpolation
        sequence, organizes the results into keyed solution objects, and creates
        globally-indexed metadata for all trees in the final sequence.

        Args:
            trees: List of processed (rooted and optimized) phylogenetic trees.
            precomputed_pair_solutions: Pre-calculated lattice solutions for each pair.

        Returns:
            A tuple containing the complete list of interpolated trees, a parallel
            list of metadata, and a dictionary of pair-specific solutions.
        """
        # Execute the core lattice-based interpolation algorithm
        # This generates the complete sequence with integrated naming and tracking
        result: TreeInterpolationSequence = build_sequential_lattice_interpolations(
            trees,
            precomputed_pair_solutions=precomputed_pair_solutions,
        )

        # Extract all interpolation data from the structured result
        # This unpacks the comprehensive TreeInterpolationSequence into components
        interpolated_trees = result.interpolated_trees  # Complete tree sequence

        solution_to_target_map_list = (
            result.mapping_one
        )  # Map from solution to atoms in TARGET tree
        solution_to_reference_map_list = (
            result.mapping_two
        )  # Map from solution to atoms in REFERENCE tree

        self.logger.debug(f"Solution to Target Maps: {solution_to_target_map_list}")
        self.logger.debug(
            f"Solution to Reference Maps: {solution_to_reference_map_list}"
        )

        active_changing_split_tracking = (
            result.active_changing_split_tracking
        )  # Active changing split applied per tree

        jumping_subtree_solutions_list = (
            result.jumping_subtree_solutions_list
        )  # Jumping subtree solutions

        # Pre-scan to find original tree positions (avoid duplicate scans)
        original_tree_global_indices: List[int] = [
            idx for idx, val in enumerate(active_changing_split_tracking) if val is None
        ]

        # Transform interpolation data into keyed TreePairSolution objects
        # This organizes pair-specific data for efficient lookup and analysis
        tree_pair_solutions_dict, pair_interpolation_ranges = (
            self._create_keyed_solutions(
                trees,
                solution_to_target_map_list,
                solution_to_reference_map_list,
                jumping_subtree_solutions_list,
                active_changing_split_tracking,
                original_tree_global_indices,
            )
        )

        # Generate comprehensive metadata for global tree indexing and navigation
        # This creates parallel metadata enabling reverse lookup and relationship tracking
        tree_metadata: List[TreeMetadata] = self._create_global_tree_metadata(
            active_changing_split_tracking,
            original_tree_global_indices,
        )

        return (
            interpolated_trees,
            tree_metadata,
            tree_pair_solutions_dict,
            pair_interpolation_ranges,
        )

    def _optimize_tree_order(
        self,
        trees: List[Node],
        precomputed_pair_active_changing_splits: Optional[
            List[Optional[PartitionSet[Partition]]]
        ] = None,
    ) -> List[Node]:
        """
        Optimizes the leaf node order to minimize visual crossings.

        Uses either rotation-based or anchor-based ordering depending on configuration:
        - Rotation-based (default): Fast iterative optimization via split rotations
        - Anchor-based: Deterministic lattice-based ordering for topological differences

        Args:
            trees: List of trees to optimize.
            precomputed_pair_active_changing_splits: Pre-calculated active-changing-splits for the optimizer.
                Only used for rotation-based optimization.

        Returns:
            List of trees with optimized leaf order.
        """
        if len(trees) <= 1:
            return trees

        optimizer = TreeOrderOptimizer(
            trees,
            precomputed_active_changing_splits=precomputed_pair_active_changing_splits,
        )

        if self.config.use_anchor_ordering:
            # Use anchor-based ordering (deterministic, non-iterative)
            self.logger.info("Using anchor-based ordering (lattice algorithm)")
            optimizer.optimize_with_anchor_ordering()
        else:
            # Use rotation-based optimization (iterative, heuristic)
            self.logger.info("Using rotation-based optimization")
            optimizer.optimize(
                n_iterations=self.config.optimization_iterations,
                bidirectional=self.config.bidirectional_optimization,
            )

        return trees

    # --- One-time lattice solutions for all adjacent pairs ---
    def _precompute_pair_solutions(
        self, trees: List[Node]
    ) -> List[Optional[Dict[Partition, List[List[Partition]]]]]:
        """
        Runs the lattice algorithm once for each adjacent pair of trees.

        This pre-calculation is an optimization, as the results are used by both
        the leaf order optimizer and the interpolation process.
        """
        if len(trees) < 2:
            return []
        sols: List[Optional[Dict[Partition, List[List[Partition]]]]] = []
        for i in range(len(trees) - 1):
            # Use deep copies to avoid any side effects (lattice may mutate working copies)
            source_tree = trees[i]
            destination_tree = trees[i + 1]
            try:
                self.logger.info(f"Precomputing solution for pair {i}-{i + 1}...")
                sols.append(
                    iterate_lattice_algorithm(
                        source_tree.deep_copy(), destination_tree.deep_copy()
                    )
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to compute lattice solution for pair {i}-{i + 1}. Error: {e}",
                    exc_info=True,  # Set to False in production if too verbose
                )
                sols.append(None)
        return sols

    def _extract_active_changing_split_sets(
        self,
        precomputed_pair_solutions: List[
            Optional[Dict[Partition, List[List[Partition]]]]
        ],
        trees: List[Node],
    ) -> List[Optional[PartitionSet[Partition]]]:
        """
        Extracts the set of active-changing-split keys from precomputed solutions.

        This is used to provide the leaf order optimizer with initial active-changing-splits.
        """
        # Convert solution dicts to PartitionSet keys for optimizer usage
        # Since all trees should have the same taxa encoding, use the first tree's encoding
        from brancharchitect.elements.partition_set import PartitionSet

        if not trees:
            return []

        # All trees should have the same taxa encoding - use the first one
        common_encoding = trees[0].taxa_encoding

        active_changing_split_sets: List[Optional[PartitionSet[Partition]]] = []
        for sol in precomputed_pair_solutions:
            if sol is None:
                active_changing_split_sets.append(None)
                continue
            ps: PartitionSet[Partition] = PartitionSet(encoding=common_encoding)
            for active_changing_split in sol.keys():
                ps.add(active_changing_split)
            active_changing_split_sets.append(ps)
        return active_changing_split_sets

    def _calculate_distances(self, trees: List[Node]) -> DistanceMetrics:
        """
        Calculates Robinson-Foulds distances between consecutive trees.

        Args:
            trees: List of trees to compute distances for.

        Returns:
            A DistanceMetrics object with lists of RF and wRF distances.
        """
        if len(trees) < 2:
            return DistanceMetrics(rfd_list=[0.0], wrfd_list=[0.0])

        # Calculate distances along trajectory
        rfd_list: List[float] = calculate_along_trajectory(
            trees, relative_robinson_foulds_distance
        )
        wrfd_list: List[float] = calculate_along_trajectory(
            trees, weighted_robinson_foulds_distance
        )

        return DistanceMetrics(rfd_list=rfd_list, wrfd_list=wrfd_list)

    def _create_keyed_solutions(
        self,
        trees: List[Node],  # Original trees
        solution_to_target_map_list: List[Dict[Partition, Partition]],
        solution_to_reference_map_list: List[Dict[Partition, Partition]],
        jumping_subtree_solutions_list: List[Dict[Partition, List[List[Partition]]]],
        active_changing_split_tracking: List[Optional[Partition]],
        original_tree_global_indices: List[int],
    ) -> tuple[Dict[str, TreePairSolution], List[List[int]]]:
        """
        Creates a dictionary of TreePairSolution objects keyed by pair identifiers.

        This transforms raw interpolation data into an organized dictionary
        (e.g., "pair_0_1") for easy lookup of solutions and metadata for each pair.

        Args:
            trees: Original trees being interpolated between.
            solution_to_target_map_list: Target tree solution-to-atom mappings.
            solution_to_reference_map_list: Reference tree solution-to-atom mappings.
            jumping_subtree_solutions_list: Jumping subtree solutions for each pair.
            active_changing_split_tracking: Active changing split applied for each step.
            original_tree_global_indices: Pre-scanned positions of original trees.

        Returns:
            A dictionary of TreePairSolution objects.
        """
        tree_pair_solutions: Dict[str, TreePairSolution] = {}
        pair_interpolation_ranges: List[List[int]] = []

        for i in range(len(trees) - 1):
            # Create pair key
            pair_key = f"pair_{i}_{i + 1}"

            # Get jumping subtree solutions for this pair
            pair_jumping_subtree_solutions = (
                jumping_subtree_solutions_list[i]
                if i < len(jumping_subtree_solutions_list)
                else {}
            )

            # Slice per-pair sequences using None boundaries (exclude the boundary None)
            if i < len(original_tree_global_indices) - 1:
                start = original_tree_global_indices[i] + 1
                end = original_tree_global_indices[i + 1]
                interpolation_start_idx = start  # First interpolated tree for this pair
            else:
                # Fallback if boundaries are unexpected; produce empty sequences
                start = 0
                end = 0
                interpolation_start_idx = 0

            pair_active_changing_split_sequence = active_changing_split_tracking[
                start:end
            ]

            # Compute steps based on split sequence only; subtrees removed
            num_steps = len(pair_active_changing_split_sequence)

            # Get global indices for this pair (needed for emit_event function)
            source_global_idx = original_tree_global_indices[i]
            target_global_idx = original_tree_global_indices[i + 1]

            # Build split_change_events (0-based step indices within this pair)
            split_change_events = []
            if num_steps > 0:
                current_split = None
                start_idx = 0
                # Maintain order-preserving list + O(1) membership set

                def emit_event(end_idx: int):
                    if current_split is None:
                        return
                    split_change_events.append(
                        {
                            "split": current_split,
                            "step_range": (start_idx, end_idx),
                            "source_tree_global_index": source_global_idx,
                            "target_tree_global_index": target_global_idx,
                        }
                    )

                for local_idx in range(num_steps):
                    split = pair_active_changing_split_sequence[local_idx]

                    if local_idx == 0:
                        current_split = split
                        start_idx = 0
                        continue

                    if split != current_split:
                        # Close previous event at local_idx - 1
                        emit_event(local_idx - 1)
                        # Start new event
                        current_split = split
                        start_idx = local_idx
                        # Subtree tracking removed

                # Emit final event (end at last index)
                emit_event(num_steps - 1)

            # No running index needed; boundaries determine slices per pair

            solution: TreePairSolution = {
                "jumping_subtree_solutions": pair_jumping_subtree_solutions,
                "solution_to_target_map": solution_to_target_map_list[i]
                if i < len(solution_to_target_map_list)
                else {},
                "solution_to_reference_map": solution_to_reference_map_list[i]
                if i < len(solution_to_reference_map_list)
                else {},
                "ancestor_of_changing_splits": pair_active_changing_split_sequence,
                "split_change_events": split_change_events,
                "source_tree_global_index": source_global_idx,
                "target_tree_global_index": target_global_idx,
                "interpolation_start_global_index": interpolation_start_idx,
            }

            tree_pair_solutions[pair_key] = solution

            # Add the range for this pair (from source original tree to target original tree)
            # This includes the source tree, all interpolated trees, and the target tree
            pair_start = source_global_idx  # Start at source original tree
            pair_end = target_global_idx  # End at target original tree
            pair_interpolation_ranges.append([pair_start, pair_end])

        return tree_pair_solutions, pair_interpolation_ranges

    def _create_global_tree_metadata(
        self,
        active_changing_split_tracking: List[
            Optional[Partition]
        ],  # This is result.active_changing_split_tracking
        original_tree_global_indices: List[int],
    ) -> List[TreeMetadata]:
        """
        Creates a metadata entry for each tree in the final interpolated sequence.

        This allows for reverse lookup from a global index to its original pair
        and step number.
        """
        metadata: List[TreeMetadata] = []

        # Track step within current pair
        current_pair_original_start_idx = 0
        original_tree_idx_counter = 0
        interpolated_step_in_current_pair = 0

        for global_idx in range(len(active_changing_split_tracking)):
            active_changing_split_info = active_changing_split_tracking[global_idx]

            if active_changing_split_info is None:  # This is an original tree
                metadata.append(
                    TreeMetadata(
                        global_tree_index=global_idx,
                        tree_pair_key=None,
                        step_in_pair=None,
                        reference_pair_tree_index=None,
                        target_pair_tree_index=None,
                        source_tree_global_index=global_idx,  # Original tree references itself
                        target_tree_global_index=None,  # No target for original trees
                    )
                )
                # When encountering an original, advance the pair anchor to this original
                current_pair_original_start_idx = original_tree_idx_counter
                original_tree_idx_counter += 1
                interpolated_step_in_current_pair = 0  # Reset step counter for new pair
            else:  # This is an interpolated tree
                interpolated_step_in_current_pair += 1
                pair_key: str = f"pair_{current_pair_original_start_idx}_{current_pair_original_start_idx + 1}"

                # Get global indices of source and target trees using pre-scanned positions
                source_global_idx = original_tree_global_indices[
                    current_pair_original_start_idx
                ]
                target_global_idx = original_tree_global_indices[
                    current_pair_original_start_idx + 1
                ]

                metadata.append(
                    TreeMetadata(
                        global_tree_index=global_idx,
                        tree_pair_key=pair_key,
                        step_in_pair=interpolated_step_in_current_pair,
                        reference_pair_tree_index=current_pair_original_start_idx,
                        target_pair_tree_index=current_pair_original_start_idx + 1,
                        source_tree_global_index=source_global_idx,
                        target_tree_global_index=target_global_idx,
                    )
                )
        return metadata

    def _apply_rooting_if_enabled(self, trees: List[Node]) -> List[Node]:
        """
        Applies midpoint rooting to trees if enabled in the configuration.

        Args:
            trees: List of trees to potentially root.

        Returns:
            The list of trees, rooted if enabled.
        """
        if self.config.enable_rooting:
            self.logger.info("Applying midpoint rooting...")
            t_root_start = time.perf_counter()
            rooted_trees = root_trees(trees)
            self.logger.info(
                f"Rooting completed in {time.perf_counter() - t_root_start:.3f}s"
            )
            return rooted_trees
        return trees
