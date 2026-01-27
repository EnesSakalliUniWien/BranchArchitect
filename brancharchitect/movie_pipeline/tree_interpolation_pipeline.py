"""Tree processing pipeline."""

from typing import List, Optional, Dict, Tuple, Callable, Any
import logging
import time
from joblib import Parallel, delayed
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
from brancharchitect.jumping_taxa.lattice.solvers.lattice_solver import (
    LatticeSolver,
)
from brancharchitect.tree_interpolation.types import TreeInterpolationSequence
from brancharchitect.tree import Node
from brancharchitect.io import serialize_subtree_tracking
from .tree_rooting import root_trees


def _parallel_solve_pair(
    source: Node, destination: Node
) -> Tuple[Optional[Dict[Partition, List[Partition]]], Optional[str]]:
    """
    Helper function to run lattice computation in a separate process.
    Returns (solution_dict, error_message).
    """
    try:
        solver = LatticeSolver(source, destination)
        # Type hint to help Pylance resolve the tuple unpacking
        result: Tuple[Dict[Partition, List[Partition]], List[Any]] = (
            solver.solve_iteratively()
        )
        solution_dict = result[0]
        return solution_dict, None
    except Exception as e:
        return None, str(e)


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

        # Configure unified debug visualization based on pipeline settings
        from brancharchitect.logger import jt_logger

        jt_logger.disabled = not self.config.enable_debug_visualization

    def process_trees(
        self,
        trees: Node | List[Node],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> InterpolationResult:
        """
        Executes the complete tree interpolation pipeline.

        Args:
            trees: List of phylogenetic trees to process.
            progress_callback: Optional callback for progress updates (0-100, message).

        Returns:
            An InterpolationResult object containing all interpolated trees,
            metadata, and analysis results.
        """

        def report(pct: float, msg: str) -> None:
            if progress_callback:
                progress_callback(pct, msg)

        start_time = time.time()

        processed_trees: Node | List[Node] = trees

        if isinstance(processed_trees, Node):
            processed_trees = [processed_trees]
        if not processed_trees:
            return create_empty_result()
        self._ensure_shared_taxa_encoding(processed_trees)
        self._check_for_identical_trees(processed_trees)
        if len(processed_trees) == 1:
            report(10, "Rooting single tree...")
            processed_trees = self._apply_rooting_if_enabled(processed_trees)
            return create_single_tree_result(processed_trees)

        report(5, "Rooting trees...")
        processed_trees = self._apply_rooting_if_enabled(processed_trees)

        report(10, "Precomputing solutions...")
        precomputed_pair_solutions = self._precompute_pair_solutions(processed_trees)

        report(20, "Optimizing tree order...")
        t_opt_start = time.perf_counter()
        processed_trees = self._optimize_tree_order(
            processed_trees,
            precomputed_pair_pivot_split_sets=self._extract_current_pivot_split_sets(
                precomputed_pair_solutions
            ),
            precomputed_pair_solutions=precomputed_pair_solutions,
        )
        self.logger.info(
            f"Leaf order optimization took {time.perf_counter() - t_opt_start:.3f}s"
        )

        report(30, "Interpolating sequence...")

        # Create a sub-callback for interpolation (30-80%)
        interp_callback: Optional[Callable[[float, str], None]] = None
        if progress_callback:

            def _interp_callback(pct: float, msg: str) -> None:
                mapped = 30 + (pct / 100.0) * 50
                progress_callback(mapped, msg)

            interp_callback = _interp_callback

        seq_result = self._interpolate_tree_sequence(
            processed_trees,
            precomputed_pair_solutions=precomputed_pair_solutions,
            progress_callback=interp_callback,
        )

        self.logger.info("Calculating distance metrics...")
        report(80, "Calculating distance metrics...")
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
            subtree_tracking=serialize_subtree_tracking(
                seq_result.current_subtree_tracking
            ),
        )

    def _ensure_shared_taxa_encoding(self, trees: List[Node]) -> None:
        """Force all trees to share the same taxa encoding to avoid split lookup errors."""
        if not trees:
            return

        base_encoding = trees[0].taxa_encoding
        base_names = set(base_encoding.keys())

        for idx, tree in enumerate(trees[1:], start=1):
            names = set(tree.taxa_encoding.keys())
            if names != base_names:
                missing = base_names - names
                extra = names - base_names
                raise ValueError(
                    "Tree taxa mismatch between inputs: "
                    f"tree0 missing={sorted(missing)} extra={sorted(extra)}"
                )

            if tree.taxa_encoding != base_encoding:
                self.logger.info(
                    f"Aligning taxa encoding for tree {idx} to match first tree"
                )
                tree.initialize_split_indices(base_encoding)

    def _check_for_identical_trees(self, trees: List[Node]) -> None:
        """
        Check for consecutive identical trees and log warnings.

        Two trees are considered identical if they have the same set of splits
        (same topology). This check helps identify potential issues in input data.
        """
        if len(trees) < 2:
            return

        identical_pairs: List[Tuple[int, int]] = []

        for i in range(len(trees) - 1):
            t1_splits = trees[i].to_splits()
            t2_splits = trees[i + 1].to_splits()

            # Compare split sets (topology comparison)
            if t1_splits == t2_splits:
                identical_pairs.append((i, i + 1))
                self.logger.warning(
                    f"Trees {i} and {i + 1} are topologically identical "
                    f"(same splits). No interpolation needed between them."
                )

        if identical_pairs:
            self.logger.info(
                f"Found {len(identical_pairs)} pair(s) of identical consecutive trees: "
                f"{identical_pairs}"
            )

    # --- Private helpers ---

    def _interpolate_tree_sequence(
        self,
        trees: List[Node],
        precomputed_pair_solutions: Optional[
            List[Optional[Dict[Partition, List[Partition]]]]
        ] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> TreeInterpolationSequence:
        """
        Orchestrates the interpolation between all consecutive tree pairs.
        """
        result: TreeInterpolationSequence = SequentialInterpolationBuilder(
            logger=self.logger,
            precomputed_pair_solutions=precomputed_pair_solutions,
        ).build(trees, progress_callback=progress_callback)

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
        precomputed_pair_solutions: Optional[
            List[Optional[Dict[Partition, List[Partition]]]]
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
            precomputed_pair_solutions=precomputed_pair_solutions,
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
        Runs the lattice algorithm for each adjacent pair of trees in parallel.
        Uses joblib for efficient parallelization with lower overhead than ProcessPoolExecutor.
        """
        if len(trees) < 2:
            return []

        n_pairs = len(trees) - 1
        self.logger.info(f"Precomputing solutions for {n_pairs} pairs using joblib...")

        # Use joblib.Parallel with loky backend (default) for efficient parallelization
        # n_jobs=-1 uses all available cores
        results = Parallel(n_jobs=-1)(
            delayed(_parallel_solve_pair)(trees[i], trees[i + 1])
            for i in range(n_pairs)
        )

        # Process results and log any errors
        sols: List[Optional[Dict[Partition, List[Partition]]]] = []
        if results is None:
            return sols

        for i, result_tuple in enumerate(results):
            if result_tuple is None:
                self.logger.error(
                    f"Parallel execution returned None for pair {i}-{i + 1}."
                )
                sols.append(None)
                continue

            # Explicit unpacking to satisfy type checker if it thinks tuple could be None
            solution_dict, error_msg = result_tuple

            if error_msg:
                self.logger.error(
                    f"Failed to compute lattice solution for pair {i}-{i + 1}. Error: {error_msg}"
                )
                sols.append(None)
            else:
                sols.append(solution_dict)

        return sols

    def _extract_current_pivot_split_sets(
        self,
        precomputed_pair_solutions: List[Optional[Dict[Partition, List[Partition]]]],
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
        total = len(current_pivot_edge_tracking)
        tree_metadata: List[TreeMetadata] = [
            TreeMetadata(
                tree_pair_key=None, step_in_pair=None, source_tree_global_index=None
            )
            for _ in range(total)
        ]

        originals = sorted(original_tree_global_indices)
        if not originals:
            return tree_metadata

        # Mark originals explicitly
        for orig_idx in originals:
            tree_metadata[orig_idx] = TreeMetadata(
                tree_pair_key=None,
                step_in_pair=None,
                source_tree_global_index=None,
            )

        # Fill interpolated steps between each consecutive pair of originals
        for pair_idx in range(len(originals) - 1):
            start = originals[pair_idx]
            end = originals[pair_idx + 1]
            tree_pair_key = f"pair_{pair_idx}_{pair_idx + 1}"
            for idx in range(start + 1, end):
                tree_metadata[idx] = TreeMetadata(
                    tree_pair_key=tree_pair_key,
                    step_in_pair=idx - start,
                    source_tree_global_index=start,
                )

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
