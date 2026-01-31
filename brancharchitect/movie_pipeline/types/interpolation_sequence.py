from typing import List, Dict, TypedDict, Optional
from brancharchitect.tree import Node
from brancharchitect.tree_interpolation.types import (
    TreePairSolution,
    TreeMetadata,
)


class InterpolationResult(TypedDict):
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
        pair_data = result.tree_pair_solutions[metadata.tree_pair_key]
        # Access pair_data, step number, etc.
    else:
        # Original tree: entries have tree_pair_key == None
        pass

    # Stream through all trees
    for i, (tree, meta) in enumerate(zip(result.interpolated_trees, result.tree_metadata)):
        # Consume or log as needed
        pass

    # Analyze specific tree pair
    pair_solution = result.tree_pair_solutions["pair_1_2"]
    lattice_data = pair_solution.jumping_subtree_solutions
    mappings = (
        pair_solution.solution_to_destination_map,
        pair_solution.solution_to_source_map,
    )
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
        - With metadata: interpolated_trees[i] alongside tree_metadata[i]
        - Streaming: for tree in interpolated_trees

    Note: Length equals len(tree_metadata) - they are parallel arrays
    """

    tree_metadata: List[TreeMetadata]
    """
    Parallel metadata array providing complete lookup information for each tree.

    Essential for navigation and understanding relationships between trees.
    tree_metadata[i] always describes interpolated_trees[i].

    Each TreeMetadata contains:
        - tree_pair_key: Key to tree_pair_solutions (for interpolated trees only, None for originals)
        - step_in_pair: Interpolation step number (1-5), None for originals

    Navigation Examples:
        - Find tree pair: metadata.tree_pair_key → tree_pair_solutions[key]
        - Check step: metadata.step_in_pair (1=down, 2=collapse, 3=reorder, 4=pre-snap, 5=snap)
        - Identify source/originals via entries where tree_pair_key is None
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
        - jumping_subtree_solutions: Raw lattice algorithm results
        - solution_to_destination_map / solution_to_source_map: Solution-to-atom mappings for both trees

    Usage Examples:
        # Access specific pair data
        pair_data = tree_pair_solutions["pair_1_2"]
        lattice_solutions = pair_data.jumping_subtree_solutions

        # Iterate over all pairs
        for pair_key, solution in tree_pair_solutions.items():
            # Process solution data as needed
            pass
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

    # Processing metadata - pipeline information
    processing_time: float
    """
    Total time taken to process the trees through the complete pipeline (seconds).

    Measures end-to-end performance from process_trees() entry to result generation.
    Includes rooting, optimization, interpolation, distance calculation, and metadata generation.

    Use: Performance monitoring, pipeline optimization, progress reporting
    """

    pair_interpolation_ranges: List[List[int]]
    """
    Global index ranges [start, end] for each pair's interpolated trees.

    For each tree pair i->i+1, gives the [start, end] indices where the interpolated
    trees for that pair appear in the global interpolated_trees sequence.

    Example: [[1, 3], [5, 7]] means:
    - Pair 0->1: interpolated trees at global indices 1, 2, 3
    - Pair 1->2: interpolated trees at global indices 5, 6, 7
    """

    subtree_tracking: List[Optional[List[List[int]]]]
    """
    Serialized subtree partition for each tree in the sequence.

    Parallel to interpolated_trees and tree_metadata. For each tree at index i:
    - None: Original tree (no subtree being moved)
    - List[List[int]]: A list of disjoint taxon groups (lists of indices), where each group
                       represents a distinct subtree moving simultaneously in this step.

    This tracks which subtrees are being relocated during each interpolation step,
    enabling visualization of the specific taxa movement.

    Example: [None, [[0, 1]], [[0, 1], [4]], None, [[2, 3]], None]
    - Index 0: Original tree
    - Index 1: Subtree {0,1} is moving
    - Index 2: Subtrees {0,1} and {4} are moving simultaneously
    - Index 3: Original tree
    - Index 4: Subtree {2,3} is moving
    - Index 5: Original tree
    """


def create_single_tree_result(
    trees: List[Node],
) -> InterpolationResult:
    """Create an InterpolationResult for a single tree case."""

    # Create metadata for the single tree
    tree_metadata = [
        TreeMetadata(
            tree_pair_key=None,
            step_in_pair=None,
            source_tree_global_index=None,
        )
    ]

    return InterpolationResult(
        interpolated_trees=trees,
        tree_metadata=tree_metadata,
        tree_pair_solutions={},
        rfd_list=[0.0],
        wrfd_list=[0.0],
        processing_time=0.0,
        pair_interpolation_ranges=[],
        subtree_tracking=[None],  # Single tree has no subtree movement
    )


def create_empty_result() -> InterpolationResult:
    """Create an empty InterpolationResult for edge cases like empty tree lists."""
    return InterpolationResult(
        interpolated_trees=[],
        tree_metadata=[],
        tree_pair_solutions={},
        rfd_list=[0.0],
        wrfd_list=[0.0],
        processing_time=0.0,
        pair_interpolation_ranges=[],
        subtree_tracking=[],
    )
