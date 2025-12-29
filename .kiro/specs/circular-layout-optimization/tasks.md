# Implementation Tasks

## Task 1: Review and document existing circular layout implementation

- [ ] Review `_apply_circular_rotation` in `anchor_order.py`
- [ ] Review `_boundary_between_anchor_blocks` function
- [ ] Review `_boundary_largest_mover_at_zero` function
- [ ] Document current cache behavior in `_rotation_cut_cache`
- [ ] Identify any gaps between implementation and requirements

**Files to review:**
- `brancharchitect/leaforder/anchor_order.py`

## Task 2: Review and document circular distance metrics

- [ ] Review `circular_distance` function
- [ ] Review `circular_distance_tree_pair` function
- [ ] Review `circular_distances_trees` function
- [ ] Verify error handling for edge cases
- [ ] Document caching behavior with `@functools.lru_cache`

**Files to review:**
- `brancharchitect/leaforder/circular_distances.py`

## Task 3: Add unit tests for boundary policies

- [ ] Test `between_anchor_blocks` with multiple anchor blocks
- [ ] Test `between_anchor_blocks` fallback to band boundary
- [ ] Test `between_anchor_blocks` with no valid cut point
- [ ] Test `largest_mover_at_zero` with single mover
- [ ] Test `largest_mover_at_zero` with multiple movers (tie-breaking)
- [ ] Test `largest_mover_at_zero` with no movers

**Files to create/modify:**
- `test/leaforder/test_circular_boundary_policies.py`

## Task 4: Add unit tests for rotation caching

- [ ] Test cache hit scenario (same taxa)
- [ ] Test cache miss scenario (different taxa)
- [ ] Test cache update behavior
- [ ] Test cache stability across multiple calls

**Files to create/modify:**
- `test/leaforder/test_rotation_cache.py`

## Task 5: Add unit tests for circular distance

- [ ] Test basic circular distance calculation
- [ ] Test normalized output range [0, 1]
- [ ] Test wrap-around behavior
- [ ] Test error handling for empty inputs
- [ ] Test error handling for duplicate elements
- [ ] Test error handling for mismatched element sets

**Files to create/modify:**
- `test/leaforder/test_circular_distances.py`

## Task 6: Integration tests for circular layout optimization

- [ ] Test circular layout with single mover case
- [ ] Test circular layout with multiple movers
- [ ] Test circular layout with no movers (identical trees)
- [ ] Test animation stability (no spinning between frames)
- [ ] Compare circular vs linear layout results

**Files to create/modify:**
- `test/leaforder/test_circular_integration.py`

## Task 7: Documentation updates

- [ ] Add docstrings to circular layout functions if missing
- [ ] Document boundary policy selection criteria
- [ ] Document cache invalidation rules
- [ ] Add usage examples to module docstrings

**Files to modify:**
- `brancharchitect/leaforder/anchor_order.py`
- `brancharchitect/leaforder/circular_distances.py`

## Task 8: Performance optimization (if needed)

- [ ] Profile circular distance calculations for large trees
- [ ] Evaluate cache hit rates
- [ ] Consider batch rotation computation for animation sequences
- [ ] Document performance characteristics

**Files to potentially modify:**
- `brancharchitect/leaforder/circular_distances.py`
- `brancharchitect/leaforder/anchor_order.py`

## Summary

This spec formalizes the existing circular layout optimization functionality. The implementation already exists in `anchor_order.py` and `circular_distances.py`. The primary work is:

1. **Documentation**: Ensure the existing code is well-documented
2. **Testing**: Add comprehensive unit and integration tests
3. **Validation**: Verify the implementation meets all requirements
4. **Gap Analysis**: Identify any missing functionality

No major code changes are expected unless gaps are found during review.

