# Requirements Document

## Introduction

This feature addresses leaf ordering consistency during tree interpolation. When interpolating between phylogenetic trees, the animation should show **only the structurally necessary movements** - taxa that don't need to move topologically should remain in their original positions.

### Root Cause Analysis

Testing reveals different behaviors based on the number of movers:

**Heavy interpolation cases (multiple movers) - WORK CORRECTLY:**
- reverse_bootstrap_52 (3 movers): Topology ✓, Order ✓
- heiko_4 (4 movers): Topology ✓, Order ✓

**Light interpolation cases (single mover) - ORDER MISMATCH:**
- bird_trees (1 mover: Ostrich): Topology ✓, Order ✗

**Analysis of bird_trees case:**
- Source order: `Emu, Birds(8), Rheas(2), Ostrich, Moas(8), Kiwis, Cassowary`
- Dest Newick order: `Emu, Ostrich, Moas(8), Birds(8), Rheas(2), Kiwis, Cassowary`
- Only 1 split differs (out of 46) - Ostrich's attachment point
- Both `Birds+Rheas` and `Moas+Crocs` clades exist unchanged in BOTH trees

**Key insight:** When there's only one structural change, the destination's Newick-encoded leaf order can differ from source for non-moving clades. The current algorithm preserves source order for non-movers (correct!), but the test was comparing against destination's arbitrary Newick order.

**Pre-alignment helps** by ensuring the destination tree has the same leaf order as source for non-moving taxa, making the final result match the aligned destination.

**Recommendation:** Pre-align destination to source order before interpolation. This:
1. Has no effect on heavy cases (already work correctly)
2. Fixes the order mismatch in light cases
3. Ensures consistent, predictable behavior

Other test results:
- reverse_bootstrap_52: PASS (both with/without alignment)
- heiko_4: PASS (both with/without alignment)
- bird_trees: PASS only with alignment

The goal is to ensure that:
1. Destination tree is aligned to source order before interpolation
2. Only structurally necessary movements occur
3. Non-moving taxa maintain their source order
4. The final tree is topologically equivalent to the destination

## Glossary

- **Leaf Order**: The left-to-right sequence of taxa (leaves) when traversing a tree
- **Pivot Edge**: The active-changing split (partition) defining the scope of tree morphing
- **Moving Subtree**: The partition of taxa being relocated during a specific interpolation step
- **Anchor Taxa**: Taxa that remain stationary during a subtree movement, used as reference points for insertion
- **Other Movers**: Taxa that will move in subsequent steps but haven't moved yet; excluded from anchor calculations
- **Mover Block**: The contiguous group of taxa being moved, preserving their internal order
- **Insertion Position**: The calculated index where the mover block should be inserted among anchors
- **Jumping Taxa Algorithm**: The algorithm that identifies which subtrees need to move between source and destination trees
- **Structural Change**: A topological difference between trees that requires subtree movement to resolve

## Requirements

### Requirement 1: Pre-Interpolation Alignment

**User Story:** As a visualization user, I want the destination tree aligned to source order before interpolation, so that anchor calculations are consistent and predictable.

#### Acceptance Criteria

1. BEFORE interpolation begins, THE system SHALL reorder the destination tree to match the source tree's leaf order
2. THE alignment SHALL preserve the destination tree's topology (same set of splits)
3. WHEN aligning, THE system SHALL use the source tree's complete leaf order as the target
4. THE aligned destination SHALL be used for all anchor calculations during interpolation

### Requirement 2: Minimal Movement Principle

**User Story:** As a visualization user, I want only the structurally necessary taxa to move during interpolation, so that the animation is clear and easy to follow.

#### Acceptance Criteria

1. WHEN interpolating between trees, THE system SHALL move only taxa that require topological changes
2. WHEN a taxon does not need to move structurally, THE system SHALL keep it in its source position
3. THE final tree SHALL be topologically equivalent to the destination (same set of splits)
4. THE leaf order of non-moving taxa SHALL be preserved from the source tree

### Requirement 3: Topological Equivalence Verification

**User Story:** As a developer, I want to verify that the final tree is topologically correct, so that the interpolation produces valid results.

#### Acceptance Criteria

1. THE final tree's split set SHALL match the destination tree's split set exactly
2. WHEN comparing trees, THE comparison SHALL use split sets (not leaf order)
3. IF the final tree is not topologically equivalent to destination, THE system SHALL report an error
4. THE verification SHALL occur after all interpolation steps are complete

### Requirement 4: Smooth Leaf Order Transitions

**User Story:** As a visualization user, I want leaf ordering to transition smoothly between interpolation steps, so that the animation appears natural and taxa don't jump unexpectedly.

#### Acceptance Criteria

1. WHEN a subtree is moved during interpolation, THE leaf order change SHALL be limited to the moving taxa only
2. WHEN multiple subtrees need to move, THE ordering of non-moving taxa SHALL remain stable until their turn
3. WHEN no structural changes occur between trees, THE leaf order SHALL remain unchanged
4. THE internal order of taxa within a moving subtree SHALL be preserved during movement

### Requirement 5: Anchor-Based Insertion Accuracy

**User Story:** As a developer, I want the anchor-based insertion algorithm to correctly position moving taxa, so that the interpolation produces topologically correct intermediate trees.

#### Acceptance Criteria

1. WHEN calculating insertion position, THE algorithm SHALL count only true anchor taxa (excluding all movers)
2. WHEN `other_movers` is provided, THE algorithm SHALL exclude those taxa from anchor calculations
3. THE mover block's internal order SHALL be preserved from the source tree
4. THE insertion position SHALL be calculated relative to the destination tree's anchor ordering

### Requirement 6: Multi-Mover Coordination

**User Story:** As a developer, I want multiple movers to be coordinated correctly, so that each mover is inserted at the correct position relative to anchors only.

#### Acceptance Criteria

1. WHEN multiple subtrees need to move under the same pivot edge, THE system SHALL process them sequentially
2. WHEN processing a mover, THE system SHALL exclude pending movers from anchor calculations
3. THE order in which movers are processed SHALL be determined by the jumping taxa algorithm
4. AFTER all movers have moved, THE leaf order SHALL match the destination exactly

### Requirement 7: Edge Case Handling

**User Story:** As a developer, I want edge cases to be handled gracefully, so that the interpolation doesn't fail or produce incorrect results.

#### Acceptance Criteria

1. WHEN source and destination have identical leaf orders, THE reordering function SHALL return the source unchanged
2. WHEN the moving subtree's taxa are not in the source order, THE function SHALL log a warning and return source unchanged
3. WHEN the pivot edge is not found in either tree, THE function SHALL log a warning and return source unchanged
4. WHEN leaf sets differ between source and destination under the pivot edge, THE function SHALL raise a ValueError

### Requirement 8: Microstep Order Normalization

**User Story:** As a visualization developer, I want all microsteps within a selection to share consistent leaf ordering, so that the animation doesn't show unnecessary reordering between microsteps.

#### Acceptance Criteria

1. WHEN building microsteps for a selection, THE final snapped tree's order SHALL be applied to earlier steps
2. THE `reordered`, `pre_snap_reordered`, and `snapped_tree` SHALL all have identical leaf orders
3. THE order normalization SHALL occur after all structural changes are complete
4. THE normalized order SHALL be the destination-aligned order from the reordering step

### Requirement 9: Debugging and Observability

**User Story:** As a developer debugging ordering issues, I want clear logging and tracking of ordering decisions, so that I can diagnose unexpected behavior.

#### Acceptance Criteria

1. THE `current_subtree_tracking` list SHALL accurately reflect which subtree is moving for each tree
2. THE debug script SHALL output leaf positions for key taxa at each interpolation step
3. WHEN reordering is skipped, THE reason SHALL be logged at WARNING level
4. WHEN reordering fails, THE error SHALL be logged at ERROR level with details

