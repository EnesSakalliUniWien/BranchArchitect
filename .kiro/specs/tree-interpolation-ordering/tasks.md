# Implementation Tasks

## Task 1: Add pre-alignment to TreeOrderOptimizer

- [x] Add `pre_align: bool = True` parameter to `optimize_with_anchor_ordering` method
- [x] Implement pre-alignment logic before `derive_order_for_pair` call
- [x] Add debug logging for alignment operations
- [x] Update docstring with new parameter documentation

**Files modified:**
- `brancharchitect/leaforder/tree_order_optimiser.py`

## Task 2: Add configuration support

- [x] Add `pre_align: bool = True` field to `PipelineConfig` dataclass
- [x] Add documentation comment explaining the parameter
- [x] Update pipeline to pass `pre_align` to optimizer

**Files modified:**
- `brancharchitect/movie_pipeline/types/pipeline_config.py`
- `brancharchitect/movie_pipeline/tree_interpolation_pipeline.py`

## Task 3: Verify existing tests pass

- [x] Run leaforder test suite
- [x] Run test_heavy_alignment.py with all 3 test cases
- [x] Verify bird_trees case now passes with pre-alignment

**Test results:**
- `test/leaforder/`: 17 passed, 1 xpassed
- `test_heavy_alignment.py`: All 3 cases pass (topology and order)

## Task 4: Documentation

- [x] Create requirements.md with 9 requirements
- [x] Create design.md with architecture and rationale
- [x] Create tasks.md (this file)

**Files created:**
- `.kiro/specs/tree-interpolation-ordering/requirements.md`
- `.kiro/specs/tree-interpolation-ordering/design.md`
- `.kiro/specs/tree-interpolation-ordering/tasks.md`

## Summary

All tasks completed. The pre-alignment feature is now integrated into `optimize_with_anchor_ordering` and enabled by default. This ensures that only structurally necessary taxa move during tree interpolation, fixing the ordering issue observed in single-mover cases like bird_trees.
