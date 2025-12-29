# Implementation Plan: Subtree Tracking Integration

## Overview

This implementation adds `current_subtree_tracking` propagation through the interpolation pipeline, mirroring the existing `current_pivot_edge_tracking` pattern. The changes are surgical and follow established patterns.

## Tasks

- [x] 1. Extend TreeInterpolationSequence dataclass
  - [x] 1.1 Add `current_subtree_tracking` field to TreeInterpolationSequence
    - Add field with same type as `current_pivot_edge_tracking`: `list[Optional[Partition]]`
    - Use `default_factory=_empty_partition_list`
    - _Requirements: 1.2, 4.1_

  - [x] 1.2 Write unit test for TreeInterpolationSequence field existence
    - Verify field exists and initializes to empty list
    - _Requirements: 1.2_

- [x] 2. Update SequentialInterpolationBuilder to aggregate subtree tracking
  - [x] 2.1 Add `current_subtree_tracking` to `_initialize_build_state`
    - Initialize as empty list: `self.current_subtree_tracking: List[Optional[Partition]] = []`
    - _Requirements: 1.1, 4.2_

  - [x] 2.2 Extend subtree tracking in `_process_pair`
    - Add: `self.current_subtree_tracking.extend(interpolation_result.current_subtree_tracking)`
    - _Requirements: 1.1, 4.2_

  - [x] 2.3 Append None to subtree tracking in `_add_delimiter_frame`
    - Add: `self.current_subtree_tracking.append(None)`
    - _Requirements: 1.4_

  - [x] 2.4 Include subtree tracking in `_finalize_sequence`
    - Add `current_subtree_tracking=self.current_subtree_tracking` to TreeInterpolationSequence constructor
    - _Requirements: 1.1, 4.2_

  - [x] 2.5 Write property test for length invariant
    - **Property 1: Length Invariant**
    - **Validates: Requirements 1.3, 3.1, 3.4**

  - [x] 2.6 Write property test for pairing invariant
    - **Property 2: Pairing Invariant**
    - **Validates: Requirements 1.4, 3.2, 3.3**

- [x] 3. Checkpoint - Verify core tracking works
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Extend InterpolationResult and TreeInterpolationPipeline
  - [x] 4.1 Add `subtree_tracking` field to InterpolationResult TypedDict
    - Add field: `subtree_tracking: List[Optional[List[int]]]`
    - Located in `brancharchitect/movie_pipeline/types/interpolation_sequence.py`
    - _Requirements: 4.4_

  - [x] 4.2 Add serialization helper method to TreeInterpolationPipeline
    - Add `_serialize_subtree_tracking(self, tracking: List[Optional[Partition]]) -> List[Optional[List[int]]]`
    - Convert each Partition to sorted list of indices, None stays None
    - _Requirements: 2.2_

  - [x] 4.3 Include subtree_tracking in process_trees return value
    - Call serialization helper and include in InterpolationResult
    - _Requirements: 2.1, 4.4_

  - [x] 4.4 Write property test for serialization determinism
    - **Property 3: Serialization Determinism**
    - **Validates: Requirements 2.2**

- [x] 5. Update Frontend Data Builder
  - [x] 5.1 Add subtree_tracking to MovieData dataclass
    - Add field: `subtree_tracking: List[Optional[List[int]]]`
    - _Requirements: 4.3_

  - [x] 5.2 Pass subtree_tracking in build_movie_data_from_result
    - Extract from InterpolationResult and pass to MovieData constructor
    - _Requirements: 4.3_

  - [x] 5.3 Include subtree_tracking in assemble_frontend_dict
    - Add `"subtree_tracking": movie_data.subtree_tracking` to return dict
    - _Requirements: 2.1, 2.3_

  - [x] 5.4 Write unit test for API response structure
    - Verify subtree_tracking field exists in response
    - Verify format matches split_change_tracking
    - _Requirements: 2.1, 2.3_

- [x] 6. Final checkpoint - Full integration test
  - All 423 tests pass (including 14 subtree tracking tests)

- [x] 7. Write property test for aggregation correctness
  - **Property 4: Aggregation Correctness**
  - **Validates: Requirements 1.1, 4.2**

- [x] 8. Rename s_edge to pivot_edge throughout codebase
  - [x] 8.1 Rename s_edge variables in TreeInterpolationSequence
    - Updated `get_original_tree_indices` and `get_interpolated_tree_indices` methods
    - Updated docstrings to reference pivot_edge instead of s-edge
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [x] 8.2 Rename split_change_tracking to pivot_edge_tracking
    - Updated MovieData dataclass field name
    - Updated frontend_data_builder.py function and variable names
    - Updated assemble_frontend_dict output key
    - Updated all tests to use new naming
    - _Requirements: 5.1, 5.2_

## Notes

- All tasks are required for comprehensive implementation
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
