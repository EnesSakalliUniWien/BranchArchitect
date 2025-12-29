# Implementation Plan: Split Application Simplification

## Overview

This implementation simplifies the tree split application and path execution logic by removing defensive retry logic and trusting the planning phase's path computation. The approach is phased: add new functions first, update callers, then remove old code.

**Architecture Note:** Split application (expanding topology) is handled in `split_application.py`. Collapse operations are in `collapse_paths.py`.

## Tasks

- [x] 1. Create core split application utilities
  - [x] 1.1 Create `SplitApplicationError` exception class
    - Add to `brancharchitect/tree_interpolation/consensus_tree/split_application.py`
    - Include split, tree_splits, and message fields
    - Implement `__str__` with taxa names and diagnostic info
    - _Requirements: 5.1, 5.2_

  - [x] 1.2 Create `apply_split_simple` function
    - Add to `brancharchitect/tree_interpolation/consensus_tree/split_application.py`
    - No retry logic, no validation, fail fast
    - Raise `SplitApplicationError` on failure
    - Encoding is guaranteed consistent by pipeline (no re-encoding needed)
    - _Requirements: 1.1, 1.2, 1.4, 1.5_

  - [x] 1.3 Write property test for split application correctness
    - **Property 1: Split Application Correctness**
    - **Validates: Requirements 1.1, 1.2, 3.2**

  - [x] 1.4 Write property test for split application idempotence
    - **Property 2: Split Application Idempotence**
    - **Validates: Requirements 1.2**

- [x] 2. Create expand path execution and update collapse path
  - [x] 2.1 Create `execute_expand_path` function
    - Add to `brancharchitect/tree_interpolation/consensus_tree/split_application.py`
    - Sort splits by size (largest first)
    - Apply each split using `apply_split_simple`
    - Apply reference weights to new nodes
    - Fail fast on any error
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x] 2.2 Update `execute_collapse_path` in `collapse_paths.py`
    - Ensure it uses existing collapse functions
    - Set all collapse splits to zero length
    - Collapse zero-length branches in one pass
    - Preserve splits that exist in destination_tree
    - Refresh split indices once after completion
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 2.3 Write property test for collapse path correctness
    - **Property 3: Collapse Path Correctness**
    - **Validates: Requirements 2.1, 2.2, 2.3**

  - [x] 2.4 Write property test for no automatic conflict resolution
    - **Property 4: No Automatic Conflict Resolution**
    - **Validates: Requirements 1.4, 1.5, 3.4, 5.3**

  - [x] 2.5 Create `execute_path` unified function
    - Add to `brancharchitect/tree_interpolation/consensus_tree/collapse_paths.py`
    - Execute collapse_path first (using collapse_paths collapse)
    - Execute expand_path second (using split_application)
    - Apply reference weights from destination_tree
    - Return modified tree
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [x] 2.6 Write property test for round-trip topology correctness
    - **Property 5: Round-Trip Topology Correctness**
    - **Validates: Requirements 8.1, 8.2**

- [x] 3. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Update microsteps execution to use new functions
  - [x] 4.1 Update `build_microsteps_for_selection` in `microsteps.py`
    - Replace `create_subtree_grafted_tree` call with `execute_expand_path`
    - Use existing collapse functions from `collapse_paths.py`
    - Reduce tree copies where possible
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [x] 4.2 Update `create_subtree_grafted_tree` to use `apply_split_simple`
    - Replace `apply_split_in_tree` calls with `apply_split_simple`
    - Remove validation parameter (no longer needed)
    - _Requirements: 1.1, 5.3_

  - [x] 4.3 Write property test for weight application correctness
    - **Property 6: Weight Application Correctness**
    - **Validates: Requirements 4.3**

- [x] 5. Update consensus tree functions
  - [x] 5.1 Update `create_consensus_tree` in `consensus/consensus_tree.py`
    - Replace `apply_split_in_tree` with `apply_split_simple`
    - Add try/except for `SplitApplicationError` (skip incompatible splits)
    - _Requirements: 1.1, 5.3_

  - [x] 5.2 Update `create_majority_consensus_tree` functions
    - Replace `apply_split_in_tree` with `apply_split_simple`
    - Add try/except for `SplitApplicationError` (skip incompatible splits)
    - _Requirements: 1.1, 5.3_

- [x] 6. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Deprecate and clean up old code
  - [x] 7.1 Mark `apply_split_in_tree` as deprecated
    - Add deprecation warning
    - Update docstring to point to `apply_split_simple`
    - Keep for backward compatibility initially
    - _Requirements: N/A (cleanup)_

  - [x] 7.2 Remove old grafting.py file
    - Move any remaining needed functions to split_application.py
    - Update all imports to use new module
    - _Requirements: N/A (cleanup)_

  - [x] 7.3 Update imports and exports
    - Add new functions to module `__all__`
    - Update any import statements in other modules
    - _Requirements: N/A (cleanup)_

- [x] 8. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
  - Run full test suite including property tests
  - Verify no regressions in existing functionality

## Notes

- All tasks are required including property-based tests
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- The migration is phased to minimize risk of breaking existing functionality
- **Split application (expand) is in `split_application.py`**
- **Collapse operations are in `collapse_paths.py`**
