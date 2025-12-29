# Implementation Plan: Partition Operations Optimization

## Overview

This plan implements hash caching for the Partition class to eliminate redundant hash computations. The changes are localized to `brancharchitect/elements/partition.py`.

## Tasks

- [x] 1. Add cached hash slot and update constructors
  - [x] 1.1 Add `_cached_hash` to `__slots__` tuple
    - Add new slot after `_cached_size`
    - _Requirements: 2.1_
  - [x] 1.2 Update `__init__` to compute and cache hash
    - Add `self._cached_hash = hash(bitmask)` after bitmask computation
    - _Requirements: 2.1_
  - [x] 1.3 Update `from_bitmask` to compute and cache hash
    - Add `obj._cached_hash = hash(bitmask)` in the classmethod
    - _Requirements: 2.1_
  - [x] 1.4 Update `__hash__` to return cached value
    - Change from `return hash(self.bitmask)` to `return self._cached_hash`
    - _Requirements: 2.2, 2.4_

- [x] 2. Verify correctness with property tests
  - [x] 2.1 Write property test for hash-equality contract
    - **Property 1: Hash-Equality Contract**
    - **Validates: Requirements 3.1**
  - [x] 2.2 Write property test for bitmask-based equality
    - **Property 2: Bitmask-Based Equality**
    - **Validates: Requirements 1.1, 1.2, 1.3**
  - [x] 2.3 Write property test for hash consistency
    - **Property 3: Hash Consistency**
    - **Validates: Requirements 2.2, 2.3**
  - [x] 2.4 Write property test for set membership
    - **Property 4: Set Membership Correctness**
    - **Validates: Requirements 3.3**
  - [x] 2.5 Write property test for dictionary key correctness
    - **Property 5: Dictionary Key Correctness**
    - **Validates: Requirements 3.3**

- [x] 3. Checkpoint - Run existing tests
  - Ensure all existing Partition tests pass
  - Run `pytest test/ -k partition -v`
  - _Requirements: 3.4_

- [x] 4. Performance validation
  - [x] 4.1 Run profiling to measure improvement
    - Compare `__hash__` cumtime before and after
    - Document performance gain
    - _Requirements: 2.4_

## Notes

- All property tests are required for comprehensive validation
- The implementation is minimal - only 4 lines of code change
- All existing tests must pass to verify backward compatibility
