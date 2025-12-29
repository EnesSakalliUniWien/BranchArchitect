# Implementation Plan: Lattice Bottom Selection Improvement

## Overview

This implementation plan refactors the lattice construction decision logic to use mathematically consistent metrics when comparing nesting solutions with conflict matrix solutions. The changes are focused on the `build_conflict_matrix()` function and supporting components.

## Tasks

- [ ] 1. Create SelectionStrategy enum and SolutionCandidate dataclass
  - Create new file `brancharchitect/jumping_taxa/lattice/types/solution_candidate.py`
  - Implement `SelectionStrategy` enum with PREFER_MINIMAL, PREFER_NESTING, ALL_SOLUTIONS
  - Implement `SolutionCandidate` dataclass with solution, source, taxa_count, depth, rank_key
  - Implement `from_nesting()` and `from_conflict()` class methods
  - Implement `_compute_rank_key()` and `_compute_depth()` static methods
  - _Requirements: 1.1, 1.3, 5.1_

- [ ] 1.1 Write unit tests for SolutionCandidate
  - Test construction from nesting solutions
  - Test construction from conflict solutions
  - Test rank_key computation
  - Test depth computation
  - _Requirements: 1.1, 1.3_

- [ ] 2. Implement UnifiedSolutionRanker
  - Create new file `brancharchitect/jumping_taxa/lattice/solvers/unified_ranker.py`
  - Implement `compare()` method with subset-first ordering
  - Implement `_get_all_indices()` helper
  - Implement `select_best()` method
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [ ]* 2.1 Write property test for ranking correctness
  - **Property 3: Ranking Correctness**
  - **Validates: Requirements 2.1, 2.2, 2.3**

- [ ] 3. Refactor build_conflict_matrix to use unified ranking
  - Modify `brancharchitect/jumping_taxa/lattice/construction/build_pivot_lattices.py`
  - Add `strategy` parameter to `build_conflict_matrix()`
  - Build candidate list from both nesting and conflict solutions
  - Use `UnifiedSolutionRanker.select_best()` for selection
  - Implement `_preview_conflict_solutions()` helper
  - _Requirements: 1.1, 1.2, 5.1, 5.2, 5.4_

- [ ]* 3.1 Write property test for metric consistency
  - **Property 1: Metric Consistency**
  - **Validates: Requirements 1.1, 5.1, 5.2**

- [ ]* 3.2 Write property test for minimality preservation
  - **Property 2: Minimality Preservation**
  - **Validates: Requirements 1.2, 2.4**

- [ ] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Update collect_all_conflicts to preserve all bottoms
  - Modify `brancharchitect/jumping_taxa/lattice/construction/cover_relations.py`
  - Ensure all bottom elements are collected regardless of size
  - Preserve chain relationships (A ⊂ B ⊂ C)
  - _Requirements: 3.1, 3.4_

- [ ]* 5.1 Write property test for frontier completeness
  - **Property 4: Frontier Completeness**
  - **Validates: Requirements 3.1, 3.2, 3.4**

- [ ] 6. Update compute_child_frontiers for completeness
  - Modify `brancharchitect/jumping_taxa/lattice/construction/child_frontiers.py`
  - Make `min_size` parameter configurable (default=1)
  - Ensure self-covering entries are created for uncovered shared children
  - _Requirements: 3.2, 3.3_

- [ ] 7. Verify meet product correctness
  - Review `brancharchitect/jumping_taxa/lattice/solvers/meet_product_solvers.py`
  - Ensure maximal_elements() is applied to all intersection results
  - Verify rectangular and square matrix handling
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ]* 7.1 Write property test for meet product correctness
  - **Property 5: Meet Product Correctness**
  - **Validates: Requirements 4.1, 4.2, 4.3, 4.4**

- [ ] 8. Add configuration support to LatticeSolver
  - Modify `brancharchitect/jumping_taxa/lattice/solvers/pivot_edge_solver.py`
  - Add `selection_strategy` parameter to `LatticeSolver.__init__()`
  - Pass strategy to `build_conflict_matrix()` calls
  - _Requirements: 6.1, 6.2, 6.3_

- [ ]* 8.1 Write property test for tiebreaking determinism
  - **Property 6: Tiebreaking Determinism**
  - **Validates: Requirements 5.4**

- [ ] 9. Update lattice_algorithm API
  - Modify `lattice_algorithm()` function signature to accept strategy parameter
  - Update docstrings with new parameter documentation
  - _Requirements: 6.4_

- [ ] 10. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 11. Integration testing
  - Run existing lattice tests to verify no regressions
  - Test with real phylogenetic tree pairs
  - Verify improved bottom selection on known problematic cases
  - _Requirements: All_

## Notes

- Tasks marked with `*` are optional property-based tests that can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
