# Requirements Document

## Introduction

This feature extends the existing `current_pivot_edge_tracking` pattern to also track subtrees during tree interpolation. Currently, `current_pivot_edge_tracking` tracks which pivot edge each interpolated tree belongs to, but the corresponding `current_subtree_tracking` data (which tracks which subtree is being moved for each tree) is generated in `TreePairInterpolation` but NOT propagated through `SequentialInterpolationBuilder` to `TreeInterpolationSequence` and ultimately to the frontend API.

The goal is to expose both pivot_edge and subtree tracking data to the frontend, enabling detailed visualization of which subtree is moving under which pivot edge for each interpolation step.

Additionally, this feature standardizes terminology by renaming legacy `s_edge` references to `pivot_edge` throughout the codebase for consistency.

## Glossary

- **Pivot_Edge**: The active-changing split (partition) being processed during interpolation; defines the scope of tree morphing
- **Subtree**: The partition representing the taxa that are being moved/relocated during a specific interpolation step
- **current_pivot_edge_tracking**: Existing list tracking which pivot edge each tree in the sequence belongs to (None for original trees)
- **current_subtree_tracking**: List tracking which subtree is being moved for each tree in the sequence (None for original trees)
- **TreePairInterpolation**: Data class holding interpolation results for a single tree pair, includes both tracking lists
- **TreeInterpolationSequence**: Data class holding the complete interpolation sequence across all tree pairs
- **SequentialInterpolationBuilder**: Builder class that processes tree pairs and aggregates results into TreeInterpolationSequence
- **Partition**: A bipartition of taxa representing an edge/split in a phylogenetic tree
- **Frontend_Data_Builder**: Service that transforms backend data into frontend-consumable JSON structures

## Requirements

### Requirement 1: Propagate Subtree Tracking Through Pipeline

**User Story:** As a frontend developer, I want subtree tracking data available in the API response alongside pivot_edge tracking, so that I can visualize which subtree is moving for each interpolation step.

#### Acceptance Criteria

1. WHEN SequentialInterpolationBuilder processes a tree pair, THE builder SHALL extend `current_subtree_tracking` from the TreePairInterpolation result
2. WHEN TreeInterpolationSequence is constructed, THE sequence SHALL include a `current_subtree_tracking` field parallel to `current_pivot_edge_tracking`
3. THE `current_subtree_tracking` list SHALL have the same length as `current_pivot_edge_tracking` and `interpolated_trees`
4. FOR original trees (delimiters), THE corresponding `current_subtree_tracking` entry SHALL be None

### Requirement 2: Expose Subtree Tracking in API Response

**User Story:** As a frontend developer, I want subtree tracking data serialized in the API response, so that I can display which subtree is being moved for each tree frame.

#### Acceptance Criteria

1. WHEN the API response is assembled, THE Response SHALL include a `subtree_tracking` field containing serialized subtree partitions per tree
2. WHEN a subtree Partition is serialized, THE Serialization_Utils SHALL convert it to a sorted list of integer indices
3. THE serialization format SHALL match the existing `split_change_tracking` format for consistency
4. THE `subtree_tracking` list SHALL be indexed by global tree index (same as `interpolated_trees`)

### Requirement 3: Maintain Tracking Consistency

**User Story:** As a backend developer, I want tracking data to remain consistent and aligned, so that the frontend can reliably correlate trees with their tracking metadata.

#### Acceptance Criteria

1. THE length of `current_subtree_tracking` SHALL always equal the length of `current_pivot_edge_tracking`
2. WHEN a tree has a non-None pivot_edge, THE corresponding subtree entry SHALL also be non-None (they are paired)
3. WHEN a tree has a None pivot_edge (original tree), THE corresponding subtree entry SHALL also be None
4. THE tracking lists SHALL maintain 1:1 correspondence with the `interpolated_trees` list

### Requirement 4: Integration with Existing Types

**User Story:** As a backend developer, I want subtree tracking integrated into existing data structures, so that the codebase remains maintainable.

#### Acceptance Criteria

1. THE TreeInterpolationSequence dataclass SHALL be extended with a `current_subtree_tracking` field
2. THE SequentialInterpolationBuilder SHALL aggregate subtree tracking from each TreePairInterpolation
3. THE Frontend_Data_Builder SHALL serialize subtree tracking using existing serialization utilities
4. THE InterpolationResult type SHALL include subtree tracking data for downstream consumption


### Requirement 5: Rename s_edge to pivot_edge

**User Story:** As a developer, I want consistent terminology throughout the codebase, so that the code is easier to understand and maintain.

#### Acceptance Criteria

1. THE variable name `s_edge` SHALL be renamed to `pivot_edge` in all relevant code locations
2. THE docstrings and comments referencing `s_edge` SHALL be updated to use `pivot_edge`
3. THE method `get_original_tree_indices` docstring SHALL reference `pivot_edge` instead of `active_changing_split_tracking`
4. THE method `get_interpolated_tree_indices` docstring SHALL reference `pivot_edge` instead of `active_changing_split_tracking`
