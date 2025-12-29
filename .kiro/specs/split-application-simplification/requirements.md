# Requirements Document

## Introduction

This feature addresses the complexity in the tree split application execution during tree interpolation. The path computation (collapse_path and expand_path) is already correctly handled by the planning phase (`PivotSplitRegistry`, `build_edge_plan`). The issue is in the **execution** phase where `apply_split_in_tree` has complex recursive retry logic with implicit conflict resolution.

### Key Insight

The planning phase already computes:
- **collapse_path**: Splits unique to source tree (to be removed)
- **expand_path**: Splits unique to destination tree (to be added)

These paths are **guaranteed to be compatible** by construction - the planning phase uses `find_incompatible_splits` to ensure expand splits don't conflict with remaining tree structure after collapse. Therefore, the execution phase should be simple: collapse first, then expand.

### Current Issues

1. **Unnecessary Retry Logic**: `apply_split_in_tree` recursively retries with automatic collapsing, but if paths are computed correctly, this should never be needed
2. **Defensive Over-Engineering**: The validation and retry logic masks bugs in path computation rather than surfacing them
3. **Multiple Tree Copies**: The microsteps create many intermediate tree copies that could be reduced
4. **Scattered Responsibilities**: Collapse logic exists in multiple places (`_collapse_split_in_tree`, `collapse_zero_length_branches_for_node`, `create_collapsed_consensus_tree`)

### Goal

Simplify execution by trusting the path computation and making the execution straightforward:
1. Execute collapse_path → all collapse splits removed
2. Execute expand_path → all expand splits added
3. If anything fails, it's a bug in path computation (fail fast, don't retry)

## Glossary

- **Split**: A partition of taxa indices representing an internal node in a phylogenetic tree
- **Partition**: A tuple of sorted taxa indices with an associated encoding dictionary
- **Apply Split**: Creating a new internal node in a tree that groups the taxa specified by the split
- **Collapse Split**: Removing an internal node by splicing its children up to its parent
- **Collapse Path**: The sequence of splits to be removed during interpolation (computed by planning phase)
- **Expand Path**: The sequence of splits to be added during interpolation (computed by planning phase)
- **Taxa Encoding**: A dictionary mapping taxon names to integer indices
- **Microstep**: One of the 5 intermediate trees generated during a single subtree movement
- **Tabula Rasa**: Strategy where first subtree collapses ALL source-unique splits to create clean slate

## Requirements

### Requirement 1: Simple Split Application

**User Story:** As a developer, I want a simple split application function that applies a split without retry logic, so that bugs in path computation are surfaced immediately.

#### Acceptance Criteria

1. WHEN applying a split to a tree, THE Apply_Split function SHALL create a new internal node grouping the appropriate children
2. WHEN the split is already present in the tree, THE Apply_Split function SHALL return without modification
3. WHEN the split encoding differs from the tree encoding, THE Apply_Split function SHALL re-encode the split before application
4. IF the split cannot be applied, THEN THE Apply_Split function SHALL raise an error immediately (no retry)
5. THE Apply_Split function SHALL NOT automatically collapse conflicting splits

### Requirement 2: Collapse Path Execution

**User Story:** As a developer, I want collapse path execution to be a single atomic operation, so that the tree state is predictable.

#### Acceptance Criteria

1. WHEN executing a collapse path, THE Collapse_Executor SHALL set all specified splits to zero length
2. WHEN executing a collapse path, THE Collapse_Executor SHALL then collapse all zero-length branches in one pass
3. THE Collapse_Executor SHALL preserve splits that exist in the destination tree
4. THE Collapse_Executor SHALL refresh split indices once after all collapses complete

### Requirement 3: Expand Path Execution

**User Story:** As a developer, I want expand path execution to apply splits in correct order without conflict handling, so that execution is simple and predictable.

#### Acceptance Criteria

1. WHEN executing an expand path, THE Expand_Executor SHALL sort splits by size (largest first)
2. WHEN executing an expand path, THE Expand_Executor SHALL apply each split sequentially
3. IF any split fails to apply, THEN THE Expand_Executor SHALL raise an error with diagnostic info
4. THE Expand_Executor SHALL NOT attempt automatic conflict resolution

### Requirement 4: Unified Path Executor

**User Story:** As a developer, I want a single function that executes both collapse and expand paths, so that the microstep logic is simplified.

#### Acceptance Criteria

1. THE Path_Executor SHALL accept collapse_path, expand_path, and destination_tree as inputs
2. THE Path_Executor SHALL execute collapse first, then expand
3. THE Path_Executor SHALL apply reference weights from destination_tree to expanded nodes
4. THE Path_Executor SHALL return the modified tree with correct topology

### Requirement 5: Fail-Fast Error Handling

**User Story:** As a developer, I want execution failures to surface immediately with clear diagnostics, so that path computation bugs are caught early.

#### Acceptance Criteria

1. WHEN a split application fails, THE error message SHALL include the split indices and taxa names
2. WHEN a split application fails, THE error message SHALL include the tree's current split set
3. THE system SHALL NOT mask errors with retry logic
4. THE system SHALL log the collapse_path and expand_path when errors occur

### Requirement 6: Reduced Tree Copies

**User Story:** As a developer, I want to minimize tree copies during microstep execution, so that performance is improved.

#### Acceptance Criteria

1. THE microstep execution SHALL reuse tree instances where possible
2. WHEN a tree copy is required, THE system SHALL use deep_copy only once per microstep sequence
3. THE system SHALL avoid redundant split index recalculations
4. THE system SHALL invalidate caches only when tree structure changes

### Requirement 7: Fail-Fast Error Propagation

**User Story:** As a developer, I want errors to propagate immediately without masking, so that bugs in path computation are caught early.

#### Acceptance Criteria

1. WHEN a split application fails, THE system SHALL raise an error immediately
2. THE system SHALL NOT mask errors with retry logic
3. THE system SHALL log the collapse_path and expand_path when errors occur
4. WHEN debugging, THE error context SHALL include sufficient information to identify the root cause

### Requirement 8: Round-Trip Verification

**User Story:** As a developer, I want to verify that collapse followed by expand produces correct topology, so that interpolation correctness is assured.

#### Acceptance Criteria

1. AFTER executing collapse then expand, THE final tree SHALL contain all expand_path splits
2. AFTER executing collapse then expand, THE final tree SHALL NOT contain collapse-only splits
3. THE verification SHALL be testable via property-based testing
4. WHEN verification fails, THE system SHALL report which splits are missing or extra
