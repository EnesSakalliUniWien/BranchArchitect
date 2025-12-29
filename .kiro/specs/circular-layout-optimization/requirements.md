# Requirements Document

## Introduction

This feature addresses leaf ordering optimization for circular phylogenetic tree layouts. When displaying trees in a circular arrangement (radial layout), the "cut point" - where the circle is conceptually split to form a linear sequence - significantly impacts visual clarity. Poor cut point selection can cause stable taxa groups to be split across the 12 o'clock boundary, making the visualization harder to interpret.

### Problem Statement

Current circular layout support exists but lacks comprehensive specification:

1. **Cut Point Selection**: The algorithm needs to intelligently choose where to "cut" the circular arrangement to minimize visual disruption
2. **Boundary Policies**: Different strategies exist (`between_anchor_blocks`, `largest_mover_at_zero`) but their behavior and trade-offs aren't formally specified
3. **Rotation Caching**: The system caches rotation decisions to prevent "spinning" between frames, but cache invalidation rules need clarification
4. **Distance Metrics**: Circular distance calculations exist but their integration with optimization isn't fully specified

### Goals

1. Formalize circular layout optimization requirements
2. Specify boundary policy behaviors and selection criteria
3. Define cache management for rotation stability
4. Integrate circular distance metrics with optimization feedback

## Glossary

- **Circular Layout**: A radial tree visualization where leaves are arranged around a circle
- **Cut Point**: The position in the circular arrangement where the sequence is conceptually "cut" to form a linear order
- **Boundary Policy**: Strategy for selecting the optimal cut point
- **Rotation**: The circular shift applied to a linear leaf order to position the cut point
- **Anchor Block**: A contiguous group of stable taxa that should not be split by the cut point
- **Mover Block**: Taxa that are moving between trees; may be positioned at the cut point for visual emphasis
- **Circular Distance**: A distance metric that accounts for the wrap-around nature of circular arrangements

## Requirements

### Requirement 1: Cut Point Selection

**User Story:** As a visualization user, I want the circular layout cut point to be chosen intelligently, so that stable taxa groups are not split across the 12 o'clock boundary.

#### Acceptance Criteria

1. WHEN displaying a tree in circular layout, THE system SHALL select a cut point that minimizes disruption to anchor blocks
2. THE cut point SHALL NOT split a contiguous anchor block unless no alternative exists
3. WHEN multiple valid cut points exist, THE system SHALL prefer cutting between distinct anchor blocks
4. IF no anchor blocks exist, THE system SHALL fall back to cutting at a band boundary (between movers and anchors)

### Requirement 2: Boundary Policy - Between Anchor Blocks

**User Story:** As a developer, I want the `between_anchor_blocks` policy to reliably find cuts between stable groups, so that the visualization maintains visual coherence.

#### Acceptance Criteria

1. WHEN `between_anchor_blocks` policy is selected, THE algorithm SHALL search for positions where two adjacent taxa belong to different anchor blocks
2. THE search SHALL iterate through all positions in the circular order
3. IF a valid between-block position is found, THE system SHALL use that position as the cut point
4. IF no between-block position exists, THE system SHALL fall back to cutting at any band boundary
5. IF no band boundary exists, THE system SHALL return position 0 (no rotation)

### Requirement 3: Boundary Policy - Largest Mover at Zero

**User Story:** As a visualization user, I want the option to place the largest moving clade at the top of the circle, so that the most significant change is visually prominent.

#### Acceptance Criteria

1. WHEN `largest_mover_at_zero` policy is selected, THE algorithm SHALL identify the largest mover block by taxa count
2. THE cut point SHALL be positioned so the first taxon of the largest mover appears at index 0
3. IF multiple mover blocks have the same size, THE system SHALL use deterministic tie-breaking (lexicographic by indices)
4. IF no mover blocks exist, THE system SHALL return position 0 (no rotation)

### Requirement 4: Rotation Caching

**User Story:** As a visualization user, I want the circular layout to remain stable between animation frames, so that the circle doesn't appear to "spin" randomly.

#### Acceptance Criteria

1. THE system SHALL cache rotation decisions keyed by the edge partition
2. WHEN the same edge is processed again with identical taxa, THE system SHALL reuse the cached rotation
3. IF the taxa set changes, THE system SHALL recompute the rotation
4. THE cache SHALL store both source and destination rotations independently
5. THE cache SHALL be invalidated when tree topology changes

### Requirement 5: Circular Distance Calculation

**User Story:** As a developer, I want accurate circular distance metrics, so that I can evaluate the quality of circular layout optimizations.

#### Acceptance Criteria

1. THE circular distance function SHALL compute the minimum distance accounting for wrap-around
2. FOR each element, THE distance SHALL be `min(|pos_x - pos_y|, n - |pos_x - pos_y|)` where n is the total count
3. THE result SHALL be normalized to the range [0, 1] by dividing by the maximum possible distance
4. THE function SHALL handle edge cases: empty inputs, duplicate elements, mismatched element sets
5. THE function SHALL use caching for performance optimization

### Requirement 6: Circular Distance Integration

**User Story:** As a developer, I want circular distance metrics integrated with the optimization pipeline, so that I can measure and improve circular layout quality.

#### Acceptance Criteria

1. THE pipeline SHALL support computing circular distances between consecutive trees
2. THE system SHALL provide both total and pairwise distance options
3. WHEN `return_pairwise=True`, THE function SHALL return a list of individual distances
4. WHEN `return_pairwise=False`, THE function SHALL return the average distance
5. THE distance calculation SHALL use the current leaf order of each tree

### Requirement 7: Consistency Between Source and Destination

**User Story:** As a visualization user, I want source and destination trees to use compatible rotations, so that the animation transition is smooth.

#### Acceptance Criteria

1. WHEN computing rotations for a tree pair, THE system SHALL compute both source and destination rotations
2. THE rotations SHALL be computed using the same boundary policy
3. THE rotations SHALL be applied consistently to both trees before reordering
4. IF one tree has a cached rotation, THE system SHALL prefer consistency over optimal cut point

### Requirement 8: Error Handling

**User Story:** As a developer, I want robust error handling for circular layout edge cases, so that the system doesn't fail unexpectedly.

#### Acceptance Criteria

1. WHEN input tuples are empty, THE circular distance function SHALL raise a ValueError
2. WHEN input tuples contain duplicates, THE function SHALL raise a ValueError
3. WHEN input tuples have different element sets, THE function SHALL raise a ValueError with details
4. WHEN a mover block cannot be found in the order, THE system SHALL return position 0 gracefully
5. WHEN the key map is missing entries, THE system SHALL handle gracefully without crashing

