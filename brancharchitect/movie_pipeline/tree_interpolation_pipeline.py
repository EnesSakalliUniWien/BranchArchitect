"""Tree processing pipeline."""

from typing import List, Optional, Dict, Tuple, Callable, Any
import logging
import sys
import time
from joblib import Parallel, delayed, parallel_config
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.movie_pipeline.types import (
    PipelineConfig,
    InterpolationResult,
    create_empty_result,
    create_single_tree_result,
)
from brancharchitect.leaforder.tree_order_optimiser import TreeOrderOptimizer
from brancharchitect.leaforder.split_analysis import clear_split_pair_cache
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
from brancharchitect.io import serialize_subtree_highlights
from .temporal_contract import build_temporal_contract
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


def _report_progress(
    callback: Optional[Callable[[float, str], None]], pct: float, msg: str
) -> None:
    """Report progress if callback is provided."""
    if callback:
        callback(pct, msg)


def _create_interpolation_callback(
    callback: Callable[[float, str], None],
) -> Callable[[float, str], None]:
    """Create a sub-callback that maps interpolation progress (0-100) to (30-80)."""

    def _mapped_callback(pct: float, msg: str) -> None:
        mapped = 30 + (pct / 100.0) * 50
        callback(mapped, msg)

    return _mapped_callback


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
        start_time = time.time()
        clear_split_pair_cache()

        processed_trees: Node | List[Node] = trees

        if isinstance(processed_trees, Node):
            processed_trees = [processed_trees]
        if not processed_trees:
            return create_empty_result()
        processed_trees = self._normalize_tree_shape(processed_trees)
        self._ensure_shared_taxa_encoding(processed_trees)
        self._check_for_identical_trees(processed_trees)
        if len(processed_trees) == 1:
            _report_progress(progress_callback, 10, "Rooting single tree...")
            processed_trees = self._apply_rooting_if_enabled(processed_trees)
            processed_trees = self._normalize_tree_shape(processed_trees)
            return create_single_tree_result(processed_trees)

        _report_progress(progress_callback, 5, "Rooting trees...")
        processed_trees = self._apply_rooting_if_enabled(processed_trees)
        processed_trees = self._normalize_tree_shape(processed_trees)
        self._ensure_shared_taxa_encoding(processed_trees)
        clear_split_pair_cache()

        _report_progress(progress_callback, 10, "Precomputing solutions...")
        precomputed_lattice_solutions = self._precompute_lattice_solutions(processed_trees)

        _report_progress(progress_callback, 20, "Optimizing tree order...")
        t_opt_start = time.perf_counter()
        processed_trees = self._optimize_tree_order(
            processed_trees,
            precomputed_pair_pivot_split_sets=self._extract_current_pivot_split_sets(
                precomputed_lattice_solutions
            ),
            precomputed_lattice_solutions=precomputed_lattice_solutions,
        )
        self.logger.info(
            f"Leaf order optimization took {time.perf_counter() - t_opt_start:.3f}s"
        )

        _report_progress(progress_callback, 30, "Interpolating sequence...")

        # Create a sub-callback for interpolation (30-80%)
        interp_callback: Optional[Callable[[float, str], None]] = (
            _create_interpolation_callback(progress_callback)
            if progress_callback
            else None
        )

        seq_result = self._interpolate_tree_sequence(
            processed_trees,
            precomputed_lattice_solutions=precomputed_lattice_solutions,
            progress_callback=interp_callback,
        )

        self.logger.info("Calculating distance metrics...")
        _report_progress(progress_callback, 80, "Calculating distance metrics...")
        t_dist_start = time.perf_counter()
        (
            robinson_foulds_distances,
            weighted_robinson_foulds_distances,
        ) = self._calculate_pair_metric_values(processed_trees)
        self.logger.info(
            f"Distance metrics calculated in {time.perf_counter() - t_dist_start:.3f}s"
        )

        processing_time = time.time() - start_time
        self.logger.info(
            f"Processed {len(processed_trees)} trees in {processing_time:.2f} seconds"
        )

        return InterpolationResult(
            interpolated_trees=seq_result.interpolated_trees,
            **build_temporal_contract(
                seq_result,
                robinson_foulds_distances,
                weighted_robinson_foulds_distances,
            ),
            processing_time=processing_time,
            subtree_highlight_tracking=serialize_subtree_highlights(
                seq_result.current_subtree_highlights
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

    def _normalize_tree_shape(self, trees: List[Node]) -> List[Node]:
        """Remove topology-neutral unary internal nodes before interpolation."""
        for tree in trees:
            tree.collapse_unary_internal_nodes(preserve_lengths=True)
        return trees

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
        precomputed_lattice_solutions: Optional[
            List[Optional[Dict[Partition, List[Partition]]]]
        ] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> TreeInterpolationSequence:
        """
        Orchestrates the interpolation between all consecutive tree pairs.
        """
        result: TreeInterpolationSequence = SequentialInterpolationBuilder(
            logger=self.logger,
            precomputed_lattice_solutions=precomputed_lattice_solutions,
        ).build(trees, progress_callback=progress_callback)

        return result

    def _optimize_tree_order(
        self,
        trees: List[Node],
        precomputed_pair_pivot_split_sets: Optional[
            List[Optional[PartitionSet[Partition]]]
        ] = None,
        precomputed_lattice_solutions: Optional[
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
            precomputed_lattice_solutions=precomputed_lattice_solutions,
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

    def _precompute_lattice_solutions(
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

        # Detect if running in PyInstaller frozen executable
        is_frozen = getattr(sys, "frozen", False)

        if is_frozen or n_pairs == 1:
            # In frozen executables, multiprocessing spawn doesn't work reliably
            # due to how PyInstaller packages the application. Run sequentially
            # to ensure stability. Performance impact is acceptable for typical
            # tree counts in interactive usage.
            if is_frozen:
                self.logger.info(
                    "Frozen executable detected, running lattice solver sequentially"
                )
            else:
                self.logger.info(
                    "Single tree pair detected, running lattice solver sequentially"
                )
            results = []
            for i in range(n_pairs):
                results.append(_parallel_solve_pair(trees[i], trees[i + 1]))
        else:
            # In development, use default loky backend for best performance
            results = Parallel(n_jobs=-1)(
                delayed(_parallel_solve_pair)(trees[i], trees[i + 1])
                for i in range(n_pairs)
            )

        # Process results and log any errors
        sols: List[Optional[Dict[Partition, List[Partition]]]] = []

        for i, result_tuple in enumerate(results):
            solution_dict, error_msg = result_tuple
            if error_msg:
                self.logger.error(
                    f"Failed to compute lattice solution for pair {i}-{i + 1}. "
                    f"Error: {error_msg}"
                )
                sols.append(None)
            else:
                sols.append(solution_dict)

        return sols

    def _extract_current_pivot_split_sets(
        self,
        precomputed_lattice_solutions: List[Optional[Dict[Partition, List[Partition]]]],
    ) -> List[Optional[PartitionSet[Partition]]]:
        """
        Extracts current pivot split sets from precomputed lattice solutions.
        """
        if not precomputed_lattice_solutions:
            return []

        split_sets: List[Optional[PartitionSet[Partition]]] = []
        for solution in precomputed_lattice_solutions:
            if solution is None:
                split_sets.append(None)
            else:
                active_changing_splits_list: List[Partition] = list(solution.keys())
                split_set: PartitionSet[Partition] = PartitionSet(
                    set(active_changing_splits_list)
                )
                split_sets.append(split_set)

        return split_sets

    def _calculate_pair_metric_values(self, trees: List[Node]) -> Tuple[List[float], List[float]]:
        """
        Calculates Robinson-Foulds distances between consecutive trees.
        """
        if len(trees) < 2:
            return [], []

        robinson_foulds_distances: List[float] = calculate_along_trajectory(
            trees, relative_robinson_foulds_distance
        )
        weighted_robinson_foulds_distances: List[float] = calculate_along_trajectory(
            trees, weighted_robinson_foulds_distance
        )

        return robinson_foulds_distances, weighted_robinson_foulds_distances

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
            raise RuntimeError(f"Rooting failed: {e}") from e
