"""Tree processing pipeline."""

from typing import List, Optional, Dict
import logging
import time
import numpy as np
from brancharchitect.elements.partition import Partition
from brancharchitect.movie_pipeline.types import (
    TreeList,
    TreePairSolution,
    TreeMetadata,
    InterpolationSequence,
    PipelineConfig,
    DistanceMetrics,
    create_empty_interpolation_sequence,
    create_single_tree_interpolation_sequence,
)
from brancharchitect.rooting.rooting import midpoint_root
from brancharchitect.leaforder.tree_order_optimiser import TreeOrderOptimizer
from brancharchitect.distances.distances import (
    calculate_along_trajectory,
    relative_robinson_foulds_distance,
    weighted_robinson_foulds_distance,
)
from brancharchitect.tree_interpolation.interpolation import (
    build_sequential_lattice_interpolations,
)
from brancharchitect.tree_interpolation.types import TreeInterpolationSequence
from brancharchitect.tree import Node


class TreeInterpolationPipeline:
    """
    Comprehensive pipeline for processing and interpolating phylogenetic trees.

    This pipeline coordinates the complete workflow for tree interpolation:
    1. Optional midpoint rooting for consistent tree orientation
    2. Tree order optimization for improved visualization quality
    3. Lattice-based interpolation between all adjacent tree pairs
    4. Distance metrics calculation for trajectory analysis
    5. Global indexing and metadata generation for easy navigation

    The pipeline produces a flattened, globally-indexed result structure
    that enables direct access to any tree in the interpolation sequence
    while maintaining clear relationships to source data.

    Example usage:
        pipeline = TreeInterpolationPipeline(
            config=PipelineConfig(enable_rooting=True)
        )
        result = pipeline.process_trees([tree1, tree2, tree3])

        # Direct access to any tree
        tree = result.interpolated_trees[15]
        metadata = result.tree_metadata[15]

        # Lookup tree pair data
        if metadata.tree_pair_key:
            pair_data = result.tree_pair_solutions[metadata.tree_pair_key]
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the tree interpolation pipeline.

        Args:
            config: Pipeline configuration settings. If None, uses default config.
            logger: Logger instance for pipeline events. If None, creates default logger.
        """
        self.config = config or PipelineConfig()
        self.logger = logger or logging.getLogger(self.config.logger_name)

    def process_trees(self, trees: List[Node]) -> InterpolationSequence:
        """
        Execute the complete tree interpolation pipeline.

        Processes a list of phylogenetic trees through the full pipeline:
        rooting, optimization, interpolation, and distance calculation.
        Returns a flattened result structure with global indexing for
        easy access to all interpolated trees and their metadata.

        Args:
            trees: List of phylogenetic trees to process (minimum 1 tree)

        Returns:
            InterpolationSequence with:
            - interpolated_trees: All trees in global sequence
            - tree_metadata: Parallel lookup information for each tree
            - tree_pair_solutions: Keyed solutions for each tree pair
            - Distance metrics and processing metadata

        Raises:
            ValueError: If trees list is invalid or processing fails

        Example:
            result = pipeline.process_trees([tree1, tree2, tree3])
            # Access tree at global index 10
            tree = result.interpolated_trees[10]
            meta = result.tree_metadata[10]
        """
        start_time = time.time()

        # --- Taxa consistency check (before any processing) ---
        taxa_sets: List[set[str]] = [
            set(leaf.name for leaf in tree.get_leaves()) for tree in trees
        ]
        first_taxa = taxa_sets[0]
        for idx, taxa in enumerate(taxa_sets[1:], 1):
            if taxa != first_taxa:
                raise ValueError(
                    f"All trees must have identical taxa sets for interpolation.\n"
                    f"Tree 0 taxa: {sorted(first_taxa)}\n"
                    f"Tree {idx} taxa: {sorted(taxa)}\n"
                    f"Difference: {sorted(first_taxa.symmetric_difference(taxa))}"
                )

        processed_trees = trees

        # Handle edge cases
        if not processed_trees:
            return create_empty_interpolation_sequence()
        if len(processed_trees) == 1:
            return create_single_tree_interpolation_sequence(processed_trees)

        # Main processing pipeline
        # Debug: taxa before optimization
        for idx, tree in enumerate(processed_trees):
            taxa = set(leaf.name for leaf in tree.get_leaves())
            if taxa != first_taxa:
                print(
                    f"[DEBUG] Before optimization: Tree {idx} taxa mismatch: {sorted(taxa)} vs {sorted(first_taxa)}"
                )

        processed_trees = self._optimize_tree_order(processed_trees)

        # Debug: taxa after optimization
        for idx, tree in enumerate(processed_trees):
            taxa = set(leaf.name for leaf in tree.get_leaves())
            if taxa != first_taxa:
                print(
                    f"[DEBUG] After optimization: Tree {idx} taxa mismatch: {sorted(taxa)} vs {sorted(first_taxa)}"
                )

        # Debug: taxa before interpolation
        for idx, tree in enumerate(processed_trees):
            taxa = set(leaf.name for leaf in tree.get_leaves())
            if taxa != first_taxa:
                print(
                    f"[DEBUG] Before interpolation: Tree {idx} taxa mismatch: {sorted(taxa)} vs {sorted(first_taxa)}"
                )

        interpolated_trees, tree_metadata, tree_pair_solutions = (
            self._process_tree_pairs(processed_trees)
        )

        self.logger.info("Calculating distance metrics...")
        distances = self._calculate_distances(processed_trees)

        # Build final result
        processing_time = time.time() - start_time

        self.logger.info(
            f"Processed {len(processed_trees)} trees in {processing_time:.2f} seconds"
        )

        return InterpolationSequence(
            interpolated_trees=interpolated_trees,
            tree_metadata=tree_metadata,
            tree_pair_solutions=tree_pair_solutions,
            rfd_list=distances.rfd_list,
            wrfd_list=distances.wrfd_list,
            distance_matrix=distances.distance_matrix,
            original_tree_count=len(processed_trees),
            interpolated_tree_count=len(interpolated_trees),
            processing_time=processing_time,
        )

    # --- Private helpers ---

    def _process_tree_pairs(
        self, trees: TreeList
    ) -> tuple[List[Node], List[TreeMetadata], Dict[str, TreePairSolution]]:
        """
        Process all consecutive tree pairs through comprehensive lattice-based interpolation.

        This method orchestrates the core interpolation workflow, transforming a sequence
        of phylogenetic trees into a detailed animation-ready sequence with complete
        metadata and solution tracking. It serves as the bridge between raw tree data
        and the structured pipeline output.

        Workflow Coordination:
        1. **Lattice Interpolation**: Calls build_sequential_lattice_interpolations to
           generate the complete interpolation sequence using advanced s-edge processing
        2. **Solution Organization**: Transforms raw interpolation data into keyed
           TreePairSolution objects for efficient lookup and analysis
        3. **Metadata Generation**: Creates comprehensive TreeMetadata for each tree
           with global indexing, source tracking, and relationship information
        4. **Data Flattening**: Converts nested pair-based results into flat, globally-
           indexed structures suitable for direct pipeline consumption

        Data Transformation:
        - Input: List of N processed trees (rooted, optimized)
        - Output: Flattened sequence of M trees where M = N + Σ(5 * s_edges_per_pair)
        - Maintains perfect alignment between trees, metadata, and solution data

        Global Indexing Strategy:
        - Each tree in the final sequence has a unique global index (0 to M-1)
        - Metadata provides reverse lookup from global index to source information
        - TreePairSolution objects are keyed by "pair_i_j" format for easy access

        Args:
            trees: List of processed phylogenetic trees (rooted and optimized).
                  Must have been validated for taxa consistency and prepared
                  for interpolation through prior pipeline stages.

        Returns:
            Tuple containing three synchronized data structures:

            - **interpolated_trees**: Complete flattened sequence of all trees
              (originals + interpolated) in global order for direct access

            - **tree_metadata**: Parallel metadata list providing for each tree:
              * global_tree_index: Position in the complete sequence
              * tree_name: Human-readable identifier (e.g., "T0", "IT1_down_2")
              * source_tree_index: Original tree index (None for interpolated)
              * tree_pair_key: Pair identifier (None for originals)
              * s_edge_tracker: S-edge being processed (None for originals/classical)
              * step_in_pair: Step number within pair interpolation sequence

            - **tree_pair_solutions**: Dictionary with "pair_i_j" keys containing:
              * lattice_edge_solutions: Raw jumping taxa algorithm results
              * tree_indices: Source tree indices for this pair
              * mapping_one/mapping_two: Solution-to-atom mappings
              * s_edge_sequence: S-edges processed during interpolation
              * s_edge_distances: Distance metrics for this pair

        Performance Notes:
            - Memory usage scales with total interpolated tree count
            - Global indexing enables O(1) tree access by index
            - Pair-based lookup enables O(1) solution access by pair identifier

        Example:
            trees = [rooted_tree1, rooted_tree2, rooted_tree3]
            interp_trees, metadata, solutions = pipeline._process_tree_pairs(trees)

            # Direct tree access
            tree_15 = interp_trees[15]
            meta_15 = metadata[15]

            # Pair solution lookup
            if meta_15.tree_pair_key:
                pair_data = solutions[meta_15.tree_pair_key]
                print(f"This tree uses s-edge: {meta_15.s_edge_tracker}")
        """
        # Execute the core lattice-based interpolation algorithm
        # This generates the complete sequence with integrated naming and tracking
        result: TreeInterpolationSequence = build_sequential_lattice_interpolations(
            trees
        )

        # Extract all interpolation data from the structured result
        # This unpacks the comprehensive TreeInterpolationSequence into components
        interpolated_trees = result.interpolated_trees  # Complete tree sequence
        tree_names = result.interpolation_sequence_labels  # Human-readable names
        mapping_one = result.mapping_one  # Target tree mappings per pair
        mapping_two = result.mapping_two  # Reference tree mappings per pair
        s_edge_tracking = result.s_edge_tracking  # S-edge applied per tree
        s_edge_lengths = result.s_edge_lengths  # Steps per pair
        lattice_solutions_list = result.lattice_solutions_list  # Raw algorithm results
        s_edge_distances_list = result.s_edge_distances_list  # Distance metrics

        # Transform interpolation data into keyed TreePairSolution objects
        # This organizes pair-specific data for efficient lookup and analysis
        tree_pair_solutions_dict: Dict[str, TreePairSolution] = (
            self._create_keyed_solutions(
                trees,
                mapping_one,
                mapping_two,
                lattice_solutions_list,
                s_edge_lengths,
                s_edge_tracking,
                s_edge_distances_list,
            )
        )

        # Generate comprehensive metadata for global tree indexing and navigation
        # This creates parallel metadata enabling reverse lookup and relationship tracking
        tree_metadata: List[TreeMetadata] = self._create_global_tree_metadata(
            trees,
            tree_names,
            s_edge_tracking,
            s_edge_lengths,
        )

        return interpolated_trees, tree_metadata, tree_pair_solutions_dict

    def _root_trees(self, trees: TreeList) -> TreeList:
        """
        Apply midpoint rooting to all trees for consistent orientation.

        Midpoint rooting places the root at the midpoint of the longest path
        between any two leaves, providing a consistent tree orientation that
        improves interpolation quality and visualization.

        Args:
            trees: List of trees to root

        Returns:
            List of midpoint-rooted trees (new copies, originals unchanged)
        """
        return [midpoint_root(tree) for tree in trees]

    def _optimize_tree_order(self, trees: TreeList) -> TreeList:
        """
        Optimize the order of leaf nodes for improved visualization quality.

        Uses the TreeOrderOptimizer to rearrange leaf order within each tree
        to minimize visual crossing when trees are displayed side-by-side.
        This improves the clarity of interpolation animations and comparisons.

        Args:
            trees: List of trees to optimize (must have >1 tree for optimization)

        Returns:
            List of trees with optimized leaf order (modifies trees in-place)
        """
        if len(trees) <= 1:
            return trees

        optimizer = TreeOrderOptimizer(trees)
        optimizer.optimize(
            n_iterations=self.config.optimization_iterations,
            bidirectional=self.config.bidirectional_optimization,
        )
        return trees

    def _calculate_distances(self, trees: TreeList) -> DistanceMetrics:
        """
        Calculate Robinson-Foulds and weighted Robinson-Foulds distances.

        Computes distance metrics along the tree trajectory to quantify
        the phylogenetic differences between consecutive trees. These
        metrics provide objective measures of interpolation quality.

        Args:
            trees: List of trees to compute distances for

        Returns:
            DistanceMetrics containing:
            - rfd_list: Robinson-Foulds distances between consecutive trees
            - wrfd_list: Weighted Robinson-Foulds distances between consecutive trees
            - distance_matrix: Pairwise distance matrix (computed if needed)
        """
        if len(trees) < 2:
            return DistanceMetrics(
                rfd_list=[0.0], wrfd_list=[0.0], distance_matrix=np.zeros((1, 1))
            )

        # Calculate distances along trajectory
        rfd_list: List[float] = calculate_along_trajectory(
            trees, relative_robinson_foulds_distance
        )
        wrfd_list: List[float] = calculate_along_trajectory(
            trees, weighted_robinson_foulds_distance
        )

        # Calculate distance matrix (placeholder - can be enhanced later)
        distance_matrix = np.zeros((len(trees), len(trees)))

        return DistanceMetrics(
            rfd_list=rfd_list, wrfd_list=wrfd_list, distance_matrix=distance_matrix
        )

    def _create_keyed_solutions(
        self,
        trees: TreeList,
        mapping_one: List[Dict[Partition, Partition]],
        mapping_two: List[Dict[Partition, Partition]],
        lattice_solutions_list: List[Dict[Partition, List[List[Partition]]]],
        s_edge_lengths: List[int],
        s_edge_tracking: List[Optional[Partition]],
        s_edge_distances_list: List[Dict[Partition, Dict[str, float]]],
    ) -> Dict[str, TreePairSolution]:
        """
        Create a dictionary of TreePairSolution objects keyed by pair identifiers.

        Transforms the raw interpolation data into organized TreePairSolution
        objects, each containing lattice solutions, mappings, and s-edge sequences
        for a specific tree pair. Uses "pair_i_j" keys for easy lookup.

        Args:
            trees: Original trees being interpolated between
            mapping_one: Target tree solution-to-atom mappings for each pair
            mapping_two: Reference tree solution-to-atom mappings for each pair
            lattice_solutions_list: Raw lattice algorithm results for each pair
            s_edge_lengths: Number of interpolation steps per pair
            s_edge_tracking: S-edge applied for each interpolation step
            s_edge_distances_list: Distance metrics for each s-edge in each pair

        Returns:
            Dictionary with keys like "pair_0_1", "pair_1_2" containing
            TreePairSolution objects with complete interpolation data
        """
        tree_pair_solutions: Dict[str, TreePairSolution] = {}
        s_edge_idx = 0

        for i in range(len(trees) - 1):
            # Create pair key
            pair_key = f"pair_{i}_{i + 1}"

            # Get mappings for this tree pair
            pair_mapping_one: Dict[Partition, Partition] = (
                mapping_one[i] if i < len(mapping_one) else {}
            )
            pair_mapping_two: Dict[Partition, Partition] = (
                mapping_two[i] if i < len(mapping_two) else {}
            )

            # Get lattice edge solutions for this pair
            pair_lattice_solutions = (
                lattice_solutions_list[i] if i < len(lattice_solutions_list) else {}
            )

            # Get s-edge distances for this pair
            pair_s_edge_distances = (
                s_edge_distances_list[i] if i < len(s_edge_distances_list) else {}
            )

            # Use s_edge_lengths for correct slicing
            num_steps = s_edge_lengths[i] if i < len(s_edge_lengths) else 0
            pair_s_edge_sequence = s_edge_tracking[s_edge_idx : s_edge_idx + num_steps]
            s_edge_idx += num_steps

            solution = TreePairSolution(
                lattice_edge_solutions=pair_lattice_solutions,
                tree_indices=(i, i + 1),
                mapping_one=pair_mapping_one,
                mapping_two=pair_mapping_two,
                s_edge_sequence=pair_s_edge_sequence,
                s_edge_distances=pair_s_edge_distances,
            )

            tree_pair_solutions[pair_key] = solution

        return tree_pair_solutions

    def _create_global_tree_metadata(
        self,
        trees: List[Node],  # Original trees
        tree_names: List[str],  # This is result.interpolation_sequence_labels
        s_edge_tracking: List[Optional[Partition]],  # This is result.s_edge_tracking
        s_edge_lengths: List[Optional[int]],  # S-edge lengths
    ) -> List[TreeMetadata]:
        metadata: List[TreeMetadata] = []

        # We need to keep track of the original tree index for source_tree_index
        original_tree_idx_counter = 0

        # We also need to keep track of the step within a pair for interpolated trees
        # and the current pair's original tree indices.
        current_pair_original_start_idx = 0
        interpolated_step_in_current_pair = 0

        for global_idx in range(len(tree_names)):
            tree_name = tree_names[global_idx]
            s_edge_info = s_edge_tracking[global_idx]

            if s_edge_info is None:  # This is an original tree
                metadata.append(
                    TreeMetadata(
                        global_tree_index=global_idx,
                        tree_name=tree_name,
                        source_tree_index=original_tree_idx_counter,
                        tree_pair_key=None,
                        s_edge_tracker=None,
                        step_in_pair=None,
                    )
                )
                original_tree_idx_counter += 1
                # When we encounter an original tree, it marks the start of a new potential pair
                current_pair_original_start_idx = original_tree_idx_counter - 1
                interpolated_step_in_current_pair = 0  # Reset step counter for new pair
            else:  # This is an interpolated tree
                interpolated_step_in_current_pair += 1
                pair_key: str = f"pair_{current_pair_original_start_idx}_{current_pair_original_start_idx + 1}"
                s_edge_str = str(s_edge_info.indices)

                metadata.append(
                    TreeMetadata(
                        global_tree_index=global_idx,
                        tree_name=tree_name,
                        source_tree_index=None,
                        tree_pair_key=pair_key,
                        s_edge_tracker=s_edge_str,
                        step_in_pair=interpolated_step_in_current_pair,
                    )
                )
        return metadata
