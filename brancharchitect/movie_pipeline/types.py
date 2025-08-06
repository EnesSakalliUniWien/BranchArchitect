"""Core type definitions for phylogenetic analysis."""

from typing import List, Tuple, Any, Dict, TypeAlias, TypedDict, Optional
from dataclasses import dataclass
from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition
from numpy._typing._array_like import NDArray

# Type definitions
TreeList: TypeAlias = List[Node]


@dataclass
class DistanceMetrics:
    """Distance metrics for phylogenetic tree analysis."""

    rfd_list: List[float]
    """Robinson-Foulds distances between consecutive trees."""

    wrfd_list: List[float]
    """Weighted Robinson-Foulds distances between consecutive trees."""

    distance_matrix: NDArray[Any]
    """Pairwise distance matrix between all trees."""


@dataclass
class PipelineConfig:
    """Configuration for tree interpolation pipeline."""

    enable_rooting: bool = False
    optimization_iterations: int = 10
    bidirectional_optimization: bool = False
    enable_distance_matrix: bool = True
    logger_name: str = __name__


class TreePairSolution(TypedDict):
    """Solution data for a single tree pair."""

    # Core lattice algorithm result (CRITICAL - was missing!)
    lattice_edge_solutions: Dict[Partition, List[List[Partition]]]

    # Tree pair information
    tree_indices: Tuple[int, int]  # (start_tree_idx, end_tree_idx)

    # Mappings for atom translation
    mapping_one: Dict[Partition, Partition]  # Mapping from interpolation
    mapping_two: Dict[Partition, Partition]  # Mapping from interpolation

    # Edge sequence for tracking interpolation steps
    s_edge_sequence: List[Optional[Partition]]

    # S-edge distance information
    s_edge_distances: Dict[Partition, Dict[str, float]]
    """Distance information for each s-edge.

    Maps each s-edge (Partition) to a dictionary containing:
    - "target_distance": Average jump path distance from components to s-edge in target tree
    - "reference_distance": Average jump path distance from components to s-edge in reference tree
    - "total_distance": Sum of target and reference distances
    - "component_count": Number of jumping taxa (components) for this s-edge

    Example:
        {
            Partition([1, 3]): {
                "target_distance": 2.5,      # Avg path length in target tree
                "reference_distance": 1.8,   # Avg path length in reference tree
                "total_distance": 4.3,       # Combined distance
                "component_count": 4          # Number of jumping taxa
            }
        }

    Use: Analyze algorithmic complexity and phylogenetic distance of lattice solutions
    """


class TreeMetadata(TypedDict):
    """
    Global metadata for each tree in the complete interpolation sequence.

    Provides JSON-serializable lookup information for every tree, enabling
    easy navigation between interpolated trees and their source data.
    All indices are global across the entire interpolation sequence.

    Example usage:
        # Find which tree pair generated tree at index 15
        metadata = result.tree_metadata[15]
        if metadata.tree_pair_key:
            pair_solution = result.tree_pair_solutions[metadata.tree_pair_key]

        # Check if this is step 3 of interpolation pair 2_3
        if metadata.step_in_pair == 3 and metadata.tree_pair_key == "pair_2_3":
            # This is the reordering step for trees 2->3
    """

    # Global identification across entire sequence
    global_tree_index: int
    """Index of this tree in the flattened interpolated_trees list (0-based).

    This provides direct access: interpolated_trees[global_tree_index] gets this tree.
    Increments continuously across all tree pairs: 0, 1, 2, ..., N-1
    """

    tree_name: str
    """Human-readable name for this tree.

    Format examples:
    - "T0", "T1", "T2" for original trees
    - "IT0_down_1" for down phase of s-edge 1 in pair 0->1
    - "C0_1" for collapse phase of s-edge 1 in pair 0->1
    - "C0_1_reorder" for reorder phase of s-edge 1 in pair 0->1
    - "IT0_up_1" for up phase of s-edge 1 in pair 0->1
    - "IT0_ref_1" for reference snap of s-edge 1 in pair 0->1
    - "IT0_classical_1_3" for classical fallback step 3 of s-edge 1 in pair 0->1
    """

    # Source and relationship tracking
    source_tree_index: Optional[int]
    """Index of the original source tree (0-based), or None for interpolated trees.

    - For original trees: matches their position in input list (0, 1, 2, ...)
    - For interpolated trees: None (they don't correspond to a single source)

    Example: All interpolation steps between T1 and T2 have source_tree_index=None
    """

    tree_pair_key: Optional[str]
    """Key to lookup the TreePairSolution that generated this tree, or None for original trees.

    Format: "pair_{source_idx}_{target_idx}"
    - "pair_0_1": Interpolation from tree 0 to tree 1
    - "pair_1_2": Interpolation from tree 1 to tree 2
    - None: This is an original tree, not interpolated

    Use: tree_pair_solutions[tree_pair_key] gets the full solution data
    """

    # Interpolation step context
    s_edge_tracker: Optional[str]
    """String representation of the s-edge being processed, or None.

    JSON-serializable identifier for the specific s-edge (lattice edge) that
    was applied to generate this tree. None for original trees or classical fallbacks.

    Format: String representation of Partition indices, e.g., "(1,3,5)"
    Use: Track which phylogenetic split is being modified at each step
    """

    step_in_pair: Optional[int]
    """Step number within the s-edge interpolation (1-5), or None for original trees.

    Each s-edge generates exactly 5 interpolation steps:
    - 1: Down phase (apply reference weights to s-edge subset)
    - 2: Collapse (remove zero-length branches from consensus)
    - 3: Reorder (match reference tree's node ordering)
    - 4: Up phase (pre-snap with reference topology but target weights)
    - 5: Snap (final reference state with graft operation)

    None: This is an original tree, not an interpolation step

    Note: step_in_pair refers to position within ONE s-edge, not the entire pair.
    """


class InterpolationSequence(TypedDict):
    """
    Complete result of the tree interpolation pipeline with flattened, globally-indexed structure.

    This is the main output type from TreeInterpolationPipeline.process_trees(). It provides
    a flat, easily accessible structure for all interpolated trees and their associated data,
    enabling efficient streaming, visualization, and analysis of phylogenetic tree interpolations.

    ## Key Design Principles:

    1. **Global Indexing**: Every tree has a unique global index (0, 1, 2, ..., N-1)
    2. **Parallel Arrays**: tree_metadata[i] always describes interpolated_trees[i]
    3. **JSON Serializable**: All lookup keys and references use strings/integers
    4. **Direct Access**: No complex unwrapping needed to access sequences
    5. **Easy Lookup**: Tree pair data accessible via simple dictionary keys

    ## Structure Overview:

    For input trees [T0, T1, T2], the result contains:
    - Original trees: T0, T1, T2 at their respective positions in the sequence
    - Interpolated sequences: ONLY generated when structural differences (s-edges) exist
    - Global metadata: Lookup info for every tree in the sequence
    - Tree pair solutions: Detailed data for each pair (may contain zero interpolations)

    CRITICAL: **No s-edges = No interpolation trees**
    - If T0→T1 has 2 s-edges: generates 2×5=10 interpolated trees
    - If T1→T2 are identical: generates 0×5=0 interpolated trees
    - Result: [T0, 10_interpolated, T1, T2] (14 total trees, not assumed regular pattern!)

    ## Example Usage:

    ```python
    result = pipeline.process_trees([tree0, tree1, tree2])

    # Direct tree access
    tree = result.interpolated_trees[15]
    metadata = result.tree_metadata[15]

    # Check what this tree represents
    if metadata.tree_pair_key:
        print(f"Tree from {metadata.tree_pair_key}, step {metadata.step_in_pair}")
        pair_data = result.tree_pair_solutions[metadata.tree_pair_key]
    else:
        print(f"Original tree T{metadata.source_tree_index}")

    # Stream through all trees
    for i, (tree, meta) in enumerate(zip(result.interpolated_trees, result.tree_metadata)):
        print(f"Global index {i}: {meta.tree_name}")

    # Analyze specific tree pair
    pair_solution = result.tree_pair_solutions["pair_1_2"]
    lattice_data = pair_solution.lattice_edge_solutions
    mappings = (pair_solution.mapping_one, pair_solution.mapping_two)
    ```

    ## Data Flow and Relationships:

    ```
    Input: [T0, T1, T2]  (where T1 and T2 might be identical)
           ↓
    Pipeline Processing:
    - Rooting & Optimization
    - Lattice-based interpolation ONLY when structural differences exist
    - Global indexing and metadata generation
           ↓
    Output: ProcessingResult with (example where T0→T1 has s-edges, T1→T2 identical):
    - interpolated_trees: [T0, IT0_down_1, C0_1, C0_1_reorder, IT0_up_1, IT0_ref_1,
                               T1, T2]
                               ↑-- Note: NO interpolation between T1→T2 (they're identical)
    - tree_metadata: [meta0, meta1, meta2, meta3, meta4, meta5, meta6, meta7]
    - tree_pair_solutions: {"pair_0_1": solution_with_data, "pair_1_2": solution_with_zero_s_edges}
    ```
    """

    # Core flattened sequences - globally indexed
    interpolated_trees: List[Node]
    """
    Complete sequence of all trees in the interpolation: original + interpolated.

    This is the main tree sequence containing every tree in global order.
    Each tree can be accessed directly using its global_tree_index from metadata.

    Structure (CONDITIONAL - depends on s-edge discovery):
        [T0, IT0_down_1, C0_1, C0_1_reorder, IT0_up_1, IT0_ref_1,  # IF s-edge found in pair 0->1
              IT0_down_2, C0_2, C0_2_reorder, IT0_up_2, IT0_ref_2,  # IF 2nd s-edge found
              ...(only if more s-edges exist)...,
         T1,  # Original tree (always present)
              # NOTE: If T1→T2 are identical, NO interpolation trees here
         T2]  # Final tree (always present)

    Real Example (T0≠T1, T1=T2): [T0, IT0_down_1, C0_1, C0_1_reorder, IT0_up_1, IT0_ref_1, T1, T2]

    Access Patterns:
        - Direct: interpolated_trees[global_index]
        - With metadata: interpolated_trees[metadata.global_tree_index]
        - Streaming: for tree in interpolated_trees

    Note: Length equals len(tree_metadata) - they are parallel arrays
    """

    tree_metadata: List[TreeMetadata]
    """
    Parallel metadata array providing complete lookup information for each tree.

    Essential for navigation and understanding relationships between trees.
    tree_metadata[i] always describes interpolated_trees[i].

    Each TreeMetadata contains:
        - global_tree_index: Index in interpolated_trees (redundant but useful)
        - tree_name: Human-readable identifier ("T0", "IT0_down_1", etc.)
        - source_tree_index: Original tree index (for original trees only)
        - tree_pair_key: Key to tree_pair_solutions (for interpolated trees only)
        - s_edge_tracker: String representation of processed s-edge
        - step_in_pair: Interpolation step number (1-5)

    Navigation Examples:
        - Find tree pair: metadata.tree_pair_key → tree_pair_solutions[key]
        - Check step: metadata.step_in_pair (1=down, 2=collapse, 3=reorder, 4=pre-snap, 5=snap)
        - Identify source: metadata.source_tree_index for original trees
    """

    # Tree pair solutions - keyed for easy lookup
    tree_pair_solutions: Dict[str, TreePairSolution]
    """
    Complete interpolation data for each tree pair, keyed by pair identifier.

    Contains all the detailed algorithmic results from lattice-based interpolation
    between each adjacent pair of original trees. Essential for understanding
    the interpolation process and accessing raw computational results.

    Key Format:
        "pair_{source_idx}_{target_idx}" (e.g., "pair_0_1", "pair_1_2")

    TreePairSolution Contents:
        - lattice_edge_solutions: Raw lattice algorithm results
        - tree_indices: (source_idx, target_idx) tuple
        - mapping_one/mapping_two: Solution-to-atom mappings for both trees
        - s_edge_sequence: Sequence of s-edges applied during interpolation

    Usage Examples:
        # Access specific pair data
        pair_data = tree_pair_solutions["pair_1_2"]
        lattice_solutions = pair_data.lattice_edge_solutions

        # Iterate over all pairs
        for pair_key, solution in tree_pair_solutions.items():
            source_idx, target_idx = solution.tree_indices
            print(f"Pair {source_idx}->{target_idx}: {len(solution.s_edge_sequence)} steps")
    """

    # Distance metrics - trajectory analysis
    rfd_list: List[float]
    """
    Robinson-Foulds distances between consecutive trees in the original sequence.

    Measures topological differences between adjacent original trees (before interpolation).
    Does not include distances for interpolated trees - only the input tree trajectory.

    Length: len(original_trees) - 1
    Access: rfd_list[i] = distance between original tree i and i+1

    Use: Quantify phylogenetic differences in the input tree sequence
    """

    wrfd_list: List[float]
    """
    Weighted Robinson-Foulds distances between consecutive trees in the original sequence.

    Like rfd_list but incorporates branch length differences in addition to topology.
    Provides more sensitive distance measurement for trees with similar topologies.

    Length: len(original_trees) - 1
    Access: wrfd_list[i] = weighted distance between original tree i and i+1

    Use: Assess both topological and branch length changes in the trajectory
    """

    distance_matrix: NDArray[Any]
    """
    Pairwise distance matrix between all original trees.

    Full distance matrix showing relationships between all pairs of input trees,
    not just consecutive pairs. Useful for global trajectory analysis.

    Shape: (original_tree_count, original_tree_count)
    Access: distance_matrix[i, j] = distance between original tree i and j

    Use: Global tree relationship analysis, clustering, trajectory optimization
    """

    # Processing metadata - pipeline information
    original_tree_count: int
    """
    Number of original input trees processed by the pipeline.

    This is the count of trees provided to process_trees(), before any interpolation.
    Used to distinguish original trees from interpolated trees in the result.

    Relationship: interpolated_tree_count = original_tree_count + sum(s_edges_found_per_pair × 5)
    Note: s_edges_found_per_pair can be 0 for identical/similar trees
    """

    interpolated_tree_count: int
    """
    Total number of trees in the complete interpolated sequence.

    This equals len(interpolated_trees) and len(tree_metadata).
    Includes both original trees and all interpolated trees.

    Calculation: original_tree_count + sum(s_edges_found_per_pair × 5)
    Where s_edges_found_per_pair can be 0 for identical/similar trees

    Examples:
    - [T0, T1, T2] where all different: original=3, interpolated=3+(a×5+b×5) where a,b=s_edges found
    - [T0, T1, T2] where T1=T2: original=3, interpolated=3+(1×5+0×5)=8
    - [T0, T1] where T0=T1: original=2, interpolated=2+(0×5)=2 (no interpolation!)
    """

    processing_time: float
    """
    Total time taken to process the trees through the complete pipeline (seconds).

    Measures end-to-end performance from process_trees() entry to result generation.
    Includes rooting, optimization, interpolation, distance calculation, and metadata generation.

    Use: Performance monitoring, pipeline optimization, progress reporting
    """


def create_empty_interpolation_sequence() -> InterpolationSequence:
    """Create an empty InterpolationSequence for edge cases like empty tree lists."""
    import numpy as np

    return InterpolationSequence(
        interpolated_trees=[],
        tree_metadata=[],
        tree_pair_solutions={},
        rfd_list=[0.0],
        wrfd_list=[0.0],
        distance_matrix=np.array([[0.0]]),
        original_tree_count=0,
        interpolated_tree_count=0,
        processing_time=0.0,
    )


def create_single_tree_interpolation_sequence(
    trees: List[Node],
) -> InterpolationSequence:
    """Create an InterpolationSequence for a single tree case."""
    import numpy as np

    # Create metadata for the single tree
    tree_metadata = [
        TreeMetadata(
            global_tree_index=0,
            tree_name="T0",
            source_tree_index=0,
            tree_pair_key=None,
            s_edge_tracker=None,
            step_in_pair=None,
        )
    ]

    return InterpolationSequence(
        interpolated_trees=trees,
        tree_metadata=tree_metadata,
        tree_pair_solutions={},
        rfd_list=[0.0],
        wrfd_list=[0.0],
        distance_matrix=np.zeros((1, 1)),
        original_tree_count=1,
        interpolated_tree_count=1,
        processing_time=0.0,
    )
