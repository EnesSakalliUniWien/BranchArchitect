"""Tree processing pipeline."""

from typing import List, Optional, Dict
import logging
import time
from brancharchitect.elements.partition import Partition
from brancharchitect.movie_pipeline.types import (
    PipelineConfig,
    InterpolationSequence,
    create_empty_interpolation_sequence,
    create_single_tree_interpolation_sequence,
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
from brancharchitect.tree_interpolation.types import TreeInterpolationSequence
from brancharchitect.tree import Node
from skbio import TreeNode as SkbioTreeNode
from brancharchitect.parser.newick_parser import parse_newick


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
        self.config: PipelineConfig = config or PipelineConfig()
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
        processed_trees: List[Node] = trees

        # Handle edge cases
        if not processed_trees:
            return create_empty_interpolation_sequence()
        if len(processed_trees) == 1:
            if self.config.enable_rooting:
                processed_trees = self._root_trees(processed_trees)
            return create_single_tree_interpolation_sequence(processed_trees)

        if self.config.enable_rooting:
            self.logger.info("Applying midpoint rooting...")
            processed_trees = self._root_trees(processed_trees)

        processed_trees = self._optimize_tree_order(processed_trees)

        (
            interpolated_trees,
            tree_metadata,
            tree_pair_solutions,
            mapping_one,
            mapping_two,
        ) = self._process_tree_pairs(processed_trees)

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
            mapping_one=mapping_one,
            mapping_two=mapping_two,
            rfd_list=distances.rfd_list,
            wrfd_list=distances.wrfd_list,
            original_tree_count=len(processed_trees),
            interpolated_tree_count=len(interpolated_trees),
            processing_time=processing_time,
        )

    # --- Private helpers ---

    def _process_tree_pairs(
        self, trees: List[Node]
    ) -> tuple[
        List[Node],
        List[TreeMetadata],
        Dict[str, TreePairSolution],
        List[Dict[Partition, Partition]],
        List[Dict[Partition, Partition]],
    ]:
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
              * tree_pair_key: Pair identifier (None for originals)
              * step_in_pair: Step number within pair interpolation sequence

            - **tree_pair_solutions**: Dictionary with "pair_i_j" keys containing:
              * lattice_edge_solutions: Raw jumping taxa algorithm results
              * mapping_one/mapping_two: Solution-to-atom mappings
              * ancestor_of_changing_splits: Ancestor splits associated with each interpolation step

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

            # Pair solution lookup (example)
            if meta_15.tree_pair_key:
                pair_data = solutions[meta_15.tree_pair_key]
        """
        # Execute the core lattice-based interpolation algorithm
        # This generates the complete sequence with integrated naming and tracking
        result: TreeInterpolationSequence = build_sequential_lattice_interpolations(
            trees
        )

        # Extract all interpolation data from the structured result
        # This unpacks the comprehensive TreeInterpolationSequence into components
        interpolated_trees = result.interpolated_trees  # Complete tree sequence
        mapping_one = result.mapping_one  # Target tree mappings per pair
        mapping_two = result.mapping_two  # Reference tree mappings per pair
        active_changing_split_tracking = (
            result.active_changing_split_tracking
        )  # Active changing split applied per tree
        subtree_tracking = result.subtree_tracking  # Subtree information per tree
        # pair_interpolated_tree_counts is available if needed for summaries
        pair_interpolated_tree_counts = result.pair_interpolated_tree_counts
        lattice_solutions_list = result.lattice_solutions_list  # Raw algorithm results

        # Transform interpolation data into keyed TreePairSolution objects
        # This organizes pair-specific data for efficient lookup and analysis
        tree_pair_solutions_dict: Dict[str, TreePairSolution] = (
            self._create_keyed_solutions(
                trees,
                mapping_one,
                mapping_two,
                lattice_solutions_list,
                active_changing_split_tracking,
                subtree_tracking,
            )
        )

        # Generate comprehensive metadata for global tree indexing and navigation
        # This creates parallel metadata enabling reverse lookup and relationship tracking
        tree_metadata: List[TreeMetadata] = self._create_global_tree_metadata(
            trees,
            active_changing_split_tracking,
            subtree_tracking,
        )

        return (
            interpolated_trees,
            tree_metadata,
            tree_pair_solutions_dict,
            mapping_one,
            mapping_two,
        )

    def _root_trees(self, trees: List[Node]) -> List[Node]:
        """
        Apply midpoint rooting to all trees for consistent orientation.

        Midpoint rooting places the root at the midpoint of the longest path
        between any two leaves, providing a consistent tree orientation that
        improves interpolation quality and visualization.

        This implementation uses scikit-bio for the rooting calculation.
        It converts each tree to the Newick format, reads it into a
        scikit-bio TreeNode, performs the rooting, and then converts it
        back to a brancharchitect Node object.

        Args:
            trees: List of trees to root

        Returns:
            List of midpoint-rooted trees (new copies, originals unchanged)
        """
        rooted_trees: List[Node] = []
        for tree in trees:
            # 1. Convert brancharchitect.tree.Node to Newick string
            newick_string = tree.to_newick()

            # 2. Create a skbio.TreeNode from the Newick string
            skbio_tree = SkbioTreeNode.read([newick_string])

            # 3. Root the skbio.TreeNode at the midpoint
            rooted_skbio_tree = skbio_tree.root_at_midpoint()

            # 4. Convert the rooted skbio.TreeNode back to a Newick string
            rooted_newick_string = str(rooted_skbio_tree)

            # 5. Parse the new Newick string back to a brancharchitect.tree.Node
            # The parser returns a list, so we take the first element.
            rooted_tree = parse_newick(
                rooted_newick_string, force_list=True, treat_zero_as_epsilon=True
            )[0]
            rooted_trees.append(rooted_tree)

        return rooted_trees

    def _optimize_tree_order(self, trees: List[Node]) -> List[Node]:
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

    def _calculate_distances(self, trees: List[Node]) -> DistanceMetrics:
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
        mapping_one: List[Dict[Partition, Partition]],
        mapping_two: List[Dict[Partition, Partition]],
        lattice_solutions_list: List[Dict[Partition, List[List[Partition]]]],
        s_edge_tracking: List[Optional[Partition]],
        subtree_tracking: List[Optional[Partition]],
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
            s_edge_tracking: S-edge applied for each interpolation step

        Returns:
            Dictionary with keys like "pair_0_1", "pair_1_2" containing
            TreePairSolution objects with complete interpolation data
        """
        tree_pair_solutions: Dict[str, TreePairSolution] = {}
        # Identify original-tree boundaries in the tracking list (marked by None)
        boundary_indices = [
            idx for idx, val in enumerate(s_edge_tracking) if val is None
        ]

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

            # Slice per-pair sequences using None boundaries (exclude the boundary None)
            if i < len(boundary_indices) - 1:
                start = boundary_indices[i] + 1
                end = boundary_indices[i + 1]
            else:
                # Fallback if boundaries are unexpected; produce empty sequences
                start = 0
                end = 0

            pair_s_edge_sequence = s_edge_tracking[start:end]
            pair_subtree_sequence = subtree_tracking[start:end]
            num_steps = min(len(pair_s_edge_sequence), len(pair_subtree_sequence))

            # Build split_change_events (0-based step indices within this pair)
            split_change_events = []
            if num_steps > 0:
                current_split = None
                start_idx = 0
                seen_subtrees: list[Partition] = []

                def emit_event(end_idx: int):
                    if current_split is None:
                        return
                    split_change_events.append(
                        {
                            "split": current_split,
                            "step_range": (start_idx, end_idx),
                            "subtrees": seen_subtrees.copy(),
                        }
                    )

                for local_idx in range(num_steps):
                    split = pair_s_edge_sequence[local_idx]
                    subtree = pair_subtree_sequence[local_idx]

                    # Normalize subtree list and dedupe while preserving order
                    if subtree is not None and subtree not in seen_subtrees:
                        seen_subtrees.append(subtree)

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
                        seen_subtrees = []
                        if subtree is not None:
                            seen_subtrees.append(subtree)

                # Emit final event (end at last index)
                emit_event(num_steps - 1)

            # No running index needed; boundaries determine slices per pair

            solution = TreePairSolution(
                lattice_edge_solutions=pair_lattice_solutions,
                mapping_one=pair_mapping_one,
                mapping_two=pair_mapping_two,
                ancestor_of_changing_splits=pair_s_edge_sequence,
                subtree_sequence=pair_subtree_sequence,
            )

            # Attach split_change_events (not part of TypedDict fields initially; add via update)
            solution["split_change_events"] = split_change_events  # type: ignore[index]

            tree_pair_solutions[pair_key] = solution

        return tree_pair_solutions

    def _create_global_tree_metadata(
        self,
        trees: List[Node],  # Original trees
        active_changing_split_tracking: List[
            Optional[Partition]
        ],  # This is result.active_changing_split_tracking
        subtree_tracking: List[Optional[Partition]],  # This is result.subtree_tracking
    ) -> List[TreeMetadata]:
        metadata: List[TreeMetadata] = []

        # Track step within current pair
        current_pair_original_start_idx = 0
        interpolated_step_in_current_pair = 0

        for global_idx in range(len(active_changing_split_tracking)):
            s_edge_info = active_changing_split_tracking[global_idx]

            if s_edge_info is None:  # This is an original tree
                metadata.append(
                    TreeMetadata(
                        global_tree_index=global_idx,
                        tree_pair_key=None,
                        step_in_pair=None,
                    )
                )
                # When we encounter an original tree, it marks the start of a new potential pair
                current_pair_original_start_idx += 0  # explicit no-op for clarity
                interpolated_step_in_current_pair = 0  # Reset step counter for new pair
            else:  # This is an interpolated tree
                interpolated_step_in_current_pair += 1
                pair_key: str = f"pair_{current_pair_original_start_idx}_{current_pair_original_start_idx + 1}"

                metadata.append(
                    TreeMetadata(
                        global_tree_index=global_idx,
                        tree_pair_key=pair_key,
                        step_in_pair=interpolated_step_in_current_pair,
                    )
                )
        return metadata
