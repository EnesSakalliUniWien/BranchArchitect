# Requirements Document

## Introduction

This specification addresses the issue of bottom elements being ignored or not properly selected during lattice construction in the phylogenetic tree reconciliation algorithm. The current implementation sometimes discards valid bottom solutions in favor of conflict matrix solutions, leading to suboptimal or incorrect jumping taxa identification.

The core problem is the decision logic in `build_conflict_matrix()` that chooses between nesting solutions (derived from bottoms) and conflict matrix solutions (derived from cover intersections). The current heuristic uses a simple size comparison that doesn't account for the mathematical properties of the lattice structure.

## Glossary

- **Lattice**: A partially ordered set (poset) where every pair of elements has both a least upper bound (join) and greatest lower bound (meet)
- **Bottom_Element**: In a lattice, the minimal element(s) under a given cover - represents the most specific shared structure
- **Cover**: A set of partitions whose union equals a larger partition set; represents maximal shared clades
- **Meet_Product**: The greatest lower bound operation in the lattice, implemented as set intersection
- **Nesting_Relationship**: When one partition set is a subset of another (A ⊆ B), indicating containment
- **Incomparability**: When two partition sets have proper overlap but neither contains the other
- **Pivot_Split**: A shared split between two trees that serves as the focal point for conflict analysis
- **Jumping_Taxa**: Taxa that must "move" between clades to reconcile two tree topologies
- **Parsimony**: The principle of preferring solutions with the fewest changes (minimal jumping taxa)
- **Conflict_Matrix**: A matrix of conflicting cover pairs used to compute meet products

## Requirements

### Requirement 1: Correct Bottom Selection Criteria

**User Story:** As a phylogenetic researcher, I want the algorithm to correctly identify when bottom elements provide better solutions than conflict matrix intersections, so that I get the most parsimonious reconciliation.

#### Acceptance Criteria

1. WHEN comparing nesting solutions with conflict matrix solutions, THE Lattice_Solver SHALL use a mathematically consistent metric that accounts for both the number of taxa AND the structural relationships
2. WHEN a bottom element is a proper subset of the conflict matrix intersection, THE Lattice_Solver SHALL prefer the bottom element as it represents a more specific solution
3. WHEN multiple nesting solutions exist with equal taxa counts, THE Lattice_Solver SHALL select based on lattice depth (deeper = more specific)
4. IF the conflict matrix intersection is empty but nesting solutions exist, THEN THE Lattice_Solver SHALL use the nesting solution without comparison

### Requirement 2: Lattice-Theoretic Solution Ranking

**User Story:** As a phylogenetic researcher, I want solutions ranked according to lattice-theoretic principles, so that the algorithm produces mathematically sound results.

#### Acceptance Criteria

1. THE Lattice_Solver SHALL rank solutions using the partial order defined by subset inclusion (A ≤ B iff A ⊆ B)
2. WHEN two solutions are incomparable in the lattice, THE Lattice_Solver SHALL use taxa count as a secondary criterion
3. WHEN solutions have equal taxa counts and are incomparable, THE Lattice_Solver SHALL use deterministic tiebreaking based on bitmask ordering
4. THE Lattice_Solver SHALL prefer solutions that are minimal elements in the solution lattice (no smaller valid solution exists)

### Requirement 3: Bottom-to-Frontier Mapping Completeness

**User Story:** As a phylogenetic researcher, I want all valid bottom elements to be considered during conflict resolution, so that no potential solutions are missed.

#### Acceptance Criteria

1. WHEN computing child frontiers, THE Child_Frontier_Computer SHALL include all bottom elements that cover at least one frontier split
2. WHEN a shared direct child is not covered by any unique split, THE Child_Frontier_Computer SHALL create a self-covering entry for it
3. THE Child_Frontier_Computer SHALL NOT discard bottom elements based solely on their size (min_size parameter should be configurable)
4. WHEN bottom elements form a chain (A ⊂ B ⊂ C), THE Child_Frontier_Computer SHALL preserve all elements for later selection

### Requirement 4: Meet Product Correctness

**User Story:** As a phylogenetic researcher, I want the meet product computation to correctly identify all valid solutions, so that the algorithm doesn't miss optimal reconciliations.

#### Acceptance Criteria

1. WHEN computing the meet product of a conflict matrix, THE Meet_Product_Solver SHALL return all maximal elements of the intersection
2. WHEN the conflict matrix is rectangular (k×2), THE Meet_Product_Solver SHALL compute row-wise intersections and return all non-empty results
3. WHEN the conflict matrix is square (n×n), THE Meet_Product_Solver SHALL compute both diagonal and counter-diagonal intersections
4. THE Meet_Product_Solver SHALL apply maximal_elements() to each intersection result to ensure minimality

### Requirement 5: Solution Comparison Consistency

**User Story:** As a phylogenetic researcher, I want the algorithm to use consistent comparison metrics throughout, so that solution selection is predictable and reproducible.

#### Acceptance Criteria

1. THE Lattice_Solver SHALL use the same size metric (taxa count) for both nesting solutions and conflict matrix solutions
2. WHEN comparing solutions from different sources (nesting vs conflict), THE Lattice_Solver SHALL normalize the comparison by computing the actual taxa involved
3. THE Lattice_Solver SHALL log the comparison metrics used when selecting between solution types
4. IF a tie occurs between nesting and conflict solutions, THEN THE Lattice_Solver SHALL prefer nesting solutions (simpler logic, fewer operations)

### Requirement 6: Configurable Selection Strategy

**User Story:** As a phylogenetic researcher, I want to be able to configure the solution selection strategy, so that I can explore different reconciliation approaches.

#### Acceptance Criteria

1. THE Lattice_Solver SHALL support a configuration option to prefer nesting solutions over conflict solutions
2. THE Lattice_Solver SHALL support a configuration option to return all valid solutions instead of just the smallest
3. WHERE the configuration specifies "all solutions", THE Lattice_Solver SHALL return both nesting and conflict solutions when both exist
4. THE Lattice_Solver SHALL document the default selection strategy and its mathematical justification
