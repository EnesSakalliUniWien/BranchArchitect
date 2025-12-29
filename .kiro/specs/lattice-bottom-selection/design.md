# Design Document: Lattice Bottom Selection Improvement

## Overview

This design addresses the mathematical inconsistencies in the current lattice construction algorithm's decision logic for selecting between nesting solutions (derived from bottom elements) and conflict matrix solutions (derived from cover intersections).

The core problem is that the current implementation in `build_conflict_matrix()` uses inconsistent metrics when comparing these two solution types, leading to suboptimal or incorrect jumping taxa identification.

### Mathematical Foundation

The lattice of phylogenetic splits forms a **Boolean lattice** under subset inclusion:
- **Partial Order**: A ≤ B iff A ⊆ B (A is a subset of B)
- **Meet (∧)**: A ∧ B = A ∩ B (intersection)
- **Join (∨)**: A ∨ B = A ∪ B (union)
- **Bottom (⊥)**: The empty set ∅
- **Top (⊤)**: The full taxon set

In this lattice:
- **Nesting solutions** come from bottom elements (minimal elements under a cover)
- **Conflict solutions** come from meet products (intersections of incomparable covers)

The key insight is that **both solution types live in the same lattice**, so they should be compared using the same lattice-theoretic metrics.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Lattice Solver Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Frontier   │───▶│   Conflict   │───▶│   Solution   │       │
│  │  Computation │    │  Collection  │    │   Selection  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Bottom     │    │   Nesting    │    │   Unified    │       │
│  │  Extraction  │    │   + Conflict │    │   Ranking    │       │
│  └──────────────┘    │   Solutions  │    │   Function   │       │
│                      └──────────────┘    └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Changes

1. **Unified Solution Representation**: Both nesting and conflict solutions are represented as `PartitionSet[Partition]` with consistent metadata
2. **Unified Ranking Function**: A single ranking function that uses lattice-theoretic ordering
3. **Configurable Selection Strategy**: Options to prefer nesting, prefer conflict, or return all solutions

## Components and Interfaces

### Component 1: SolutionCandidate

A wrapper class that provides consistent metadata for all solution types.

```python
@dataclass
class SolutionCandidate:
    """
    Unified representation of a solution candidate from any source.

    Attributes:
        solution: The partition set representing the jumping taxa
        source: Origin of the solution ("nesting" or "conflict")
        taxa_count: Total number of taxa in the solution
        depth: Lattice depth (number of subset relationships to bottom)
        rank_key: Precomputed ranking tuple for comparison
    """
    solution: PartitionSet[Partition]
    source: Literal["nesting", "conflict"]
    taxa_count: int
    depth: int
    rank_key: tuple[int, tuple[int, ...], tuple[int, ...]]

    @classmethod
    def from_nesting(cls, solution: PartitionSet[Partition],
                     bottom_matrix_row: list[PartitionSet[Partition]]) -> "SolutionCandidate":
        """Create a candidate from a nesting solution."""
        taxa_count = sum(len(p.taxa) for p in solution)
        depth = cls._compute_depth(solution, bottom_matrix_row)
        rank_key = cls._compute_rank_key(solution)
        return cls(solution, "nesting", taxa_count, depth, rank_key)

    @classmethod
    def from_conflict(cls, solution: PartitionSet[Partition]) -> "SolutionCandidate":
        """Create a candidate from a conflict matrix intersection."""
        taxa_count = sum(len(p.taxa) for p in solution)
        depth = 0  # Conflict solutions don't have depth context
        rank_key = cls._compute_rank_key(solution)
        return cls(solution, "conflict", taxa_count, depth, rank_key)

    @staticmethod
    def _compute_rank_key(solution: PartitionSet[Partition]) -> tuple:
        """Compute deterministic ranking key."""
        num_parts = len(solution)
        sizes = tuple(sorted(len(p.taxa) for p in solution))
        masks = tuple(sorted(p.bitmask for p in solution))
        return (num_parts, sizes, masks)

    @staticmethod
    def _compute_depth(solution: PartitionSet[Partition],
                       context: list[PartitionSet[Partition]]) -> int:
        """Compute lattice depth based on subset relationships."""
        # Count how many elements in context are proper supersets of solution
        depth = 0
        solution_union = set()
        for p in solution:
            solution_union.update(p.indices)

        for ctx_set in context:
            ctx_union = set()
            for p in ctx_set:
                ctx_union.update(p.indices)
            if solution_union < ctx_union:  # Proper subset
                depth += 1
        return depth
```

### Component 2: UnifiedSolutionRanker

A ranking function that implements lattice-theoretic ordering.

```python
class UnifiedSolutionRanker:
    """
    Ranks solutions using lattice-theoretic principles.

    Ranking Order (most preferred first):
    1. Subset relationship: A < B if A ⊂ B (proper subset)
    2. Taxa count: Fewer taxa = more parsimonious
    3. Partition count: Fewer partitions = simpler
    4. Bitmask ordering: Deterministic tiebreaker
    """

    @staticmethod
    def compare(a: SolutionCandidate, b: SolutionCandidate) -> int:
        """
        Compare two solution candidates.

        Returns:
            -1 if a < b (a is preferred)
             0 if a == b (equivalent)
             1 if a > b (b is preferred)
        """
        # Check subset relationship first
        a_indices = UnifiedSolutionRanker._get_all_indices(a.solution)
        b_indices = UnifiedSolutionRanker._get_all_indices(b.solution)

        if a_indices < b_indices:  # a is proper subset of b
            return -1
        if b_indices < a_indices:  # b is proper subset of a
            return 1

        # Incomparable in lattice - use taxa count
        if a.taxa_count < b.taxa_count:
            return -1
        if b.taxa_count < a.taxa_count:
            return 1

        # Equal taxa count - use rank key for determinism
        if a.rank_key < b.rank_key:
            return -1
        if b.rank_key < a.rank_key:
            return 1

        # Truly equal - prefer nesting (simpler logic)
        if a.source == "nesting" and b.source == "conflict":
            return -1
        if b.source == "nesting" and a.source == "conflict":
            return 1

        return 0

    @staticmethod
    def _get_all_indices(solution: PartitionSet[Partition]) -> frozenset[int]:
        """Get all taxon indices from a solution."""
        indices: set[int] = set()
        for partition in solution:
            indices.update(partition.indices)
        return frozenset(indices)

    @staticmethod
    def select_best(candidates: list[SolutionCandidate]) -> SolutionCandidate:
        """Select the best candidate from a list."""
        if not candidates:
            raise ValueError("Cannot select from empty candidate list")

        return min(candidates, key=functools.cmp_to_key(UnifiedSolutionRanker.compare))
```

### Component 3: Improved build_conflict_matrix

The refactored decision logic that uses unified ranking.

```python
def build_conflict_matrix(
    lattice_edge: PivotEdgeSubproblem,
    strategy: SelectionStrategy = SelectionStrategy.PREFER_MINIMAL,
) -> PMatrix:
    """
    Computes conflicting pairs and selects the best solution approach.

    Args:
        lattice_edge: The pivot edge subproblem to analyze
        strategy: Selection strategy (PREFER_MINIMAL, PREFER_NESTING, ALL_SOLUTIONS)

    Returns:
        A matrix for meet product computation, or a 1×1 matrix with the
        selected nesting solution if that's more parsimonious.
    """
    left_covers = lattice_edge.tree1_child_frontiers
    right_covers = lattice_edge.tree2_child_frontiers

    # Collect all conflict types
    conflicting_cover_pairs, nesting_solutions, bottom_matrix = collect_all_conflicts(
        left_covers, right_covers
    )

    # Build candidate list
    candidates: list[SolutionCandidate] = []

    # Add nesting candidates
    for idx, solution in enumerate(nesting_solutions):
        if solution:
            candidate = SolutionCandidate.from_nesting(solution, bottom_matrix[idx])
            candidates.append(candidate)

    # Compute conflict candidates (if we have conflicts)
    if conflicting_cover_pairs:
        # Compute what the conflict matrix would yield
        conflict_solutions = _preview_conflict_solutions(conflicting_cover_pairs)
        for solution in conflict_solutions:
            if solution:
                candidate = SolutionCandidate.from_conflict(solution)
                candidates.append(candidate)

    # Apply selection strategy
    if strategy == SelectionStrategy.ALL_SOLUTIONS:
        # Return conflict matrix to get all solutions
        return conflicting_cover_pairs if conflicting_cover_pairs else [[candidates[0].solution]]

    if strategy == SelectionStrategy.PREFER_NESTING:
        nesting_candidates = [c for c in candidates if c.source == "nesting"]
        if nesting_candidates:
            best = UnifiedSolutionRanker.select_best(nesting_candidates)
            return [[best.solution]]

    # Default: PREFER_MINIMAL
    if candidates:
        best = UnifiedSolutionRanker.select_best(candidates)

        if best.source == "nesting":
            return [[best.solution]]
        else:
            return conflicting_cover_pairs

    return conflicting_cover_pairs


def _preview_conflict_solutions(matrix: PMatrix) -> list[PartitionSet[Partition]]:
    """
    Preview what solutions the conflict matrix would yield without full computation.

    This is used for comparison purposes only - computes intersections
    to estimate solution sizes.
    """
    solutions: list[PartitionSet[Partition]] = []

    for row in matrix:
        if len(row) >= 2:
            intersection = row[0] & row[1]
            if intersection:
                maxima = intersection.maximal_elements()
                solutions.append(maxima)

    return solutions
```

### Component 4: SelectionStrategy Enum

```python
from enum import Enum, auto

class SelectionStrategy(Enum):
    """Strategy for selecting between nesting and conflict solutions."""

    PREFER_MINIMAL = auto()  # Select the smallest solution regardless of source
    PREFER_NESTING = auto()  # Prefer nesting solutions when available
    ALL_SOLUTIONS = auto()   # Return all valid solutions
```

## Data Models

### Existing Data Models (Unchanged)

- `PartitionSet[Partition]`: Collection of partitions with set operations
- `TopToBottom`: Links top-level shared splits to bottom-level splits
- `PMatrix`: Matrix of partition set pairs for conflict analysis
- `PivotEdgeSubproblem`: Contains frontier information for a pivot split

### New Data Models

- `SolutionCandidate`: Unified wrapper for solution comparison
- `SelectionStrategy`: Enum for configuring selection behavior

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Metric Consistency

*For any* comparison between a nesting solution and a conflict solution, the comparison metric SHALL be the total taxa count (sum of |partition.taxa| for all partitions in the solution).

**Validates: Requirements 1.1, 5.1, 5.2**

### Property 2: Minimality Preservation

*For any* selected solution S, there SHALL NOT exist another valid solution S' such that S' ⊂ S (S' is a proper subset of S in terms of taxa indices).

**Validates: Requirements 1.2, 2.4**

### Property 3: Ranking Correctness

*For any* two solutions A and B where A ⊂ B (A is a proper subset of B), the ranking function SHALL return A < B (A is preferred over B).

**Validates: Requirements 2.1, 2.2, 2.3**

### Property 4: Frontier Completeness

*For any* child frontier computation, all bottom elements that cover at least one frontier split SHALL be included in the resulting `bottom_to_frontiers` mapping.

**Validates: Requirements 3.1, 3.2, 3.4**

### Property 5: Meet Product Correctness

*For any* conflict matrix M, the meet product result SHALL contain exactly the maximal elements of the intersection of each row (for rectangular matrices) or diagonal (for square matrices).

**Validates: Requirements 4.1, 4.2, 4.3, 4.4**

### Property 6: Tiebreaking Determinism

*For any* two solutions with equal taxa counts that are incomparable in the subset lattice, the selection SHALL be deterministic based on bitmask ordering, and nesting solutions SHALL be preferred over conflict solutions when all other criteria are equal.

**Validates: Requirements 5.4**

## Error Handling

### Error Cases

1. **Empty candidate list**: If no valid solutions exist, raise `ValueError` with diagnostic information
2. **Inconsistent encoding**: If partition sets have different encodings, raise `ValueError`
3. **Missing frontier entries**: If a bottom element cannot be found in the tree, raise `ValueError`

### Recovery Strategies

1. **Fallback to conflict matrix**: If nesting solution computation fails, fall back to conflict matrix approach
2. **Logging**: All comparison decisions are logged for debugging
3. **Validation**: Input validation at each stage to catch errors early

## Testing Strategy

### Unit Tests

Unit tests will cover:
- `SolutionCandidate` construction from both sources
- `UnifiedSolutionRanker.compare()` with various input combinations
- `SelectionStrategy` behavior for each mode
- Edge cases: empty solutions, single-element solutions, identical solutions

### Property-Based Tests

Property-based tests will use the `hypothesis` library to verify:

1. **Metric Consistency Property**: Generate random nesting and conflict solutions, verify same metric is used
2. **Minimality Property**: Generate solution sets, verify selected solution has no proper subset
3. **Ranking Property**: Generate pairs of solutions with subset relationships, verify ordering
4. **Frontier Completeness Property**: Generate frontier structures, verify all bottoms are included
5. **Meet Product Property**: Generate conflict matrices, verify maximal elements are returned
6. **Determinism Property**: Generate equal-count solutions, verify consistent ordering

### Test Configuration

- Property tests: Minimum 100 iterations per property
- Test framework: pytest with hypothesis
- Each property test tagged with: **Feature: lattice-bottom-selection, Property N: {property_text}**
