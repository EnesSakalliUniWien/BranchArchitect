# Requirements Document

## Introduction

This specification addresses performance bottlenecks in Partition operations identified through profiling. The pipeline makes approximately 12 million `__eq__` calls and 9 million `__hash__` calls on Partition objects, contributing significantly to overall runtime. The goal is to optimize these operations to reduce their overhead.

## Glossary

- **Partition**: An immutable object representing a bipartition of taxa, stored as a bitmask and tuple of indices
- **Bitmask**: An integer where each bit represents presence/absence of a taxon in the partition
- **Indices**: A sorted tuple of taxon indices contained in the partition
- **Hash_Value**: A cached integer used for dictionary lookups and set membership

## Requirements

### Requirement 1: Optimize Partition Equality Comparison

**User Story:** As a developer, I want faster Partition equality checks, so that the 12M `__eq__` calls complete more quickly.

#### Acceptance Criteria

1. WHEN comparing two Partitions for equality, THE Partition SHALL compare bitmasks first as the primary check
2. WHEN bitmasks are equal, THE Partition SHALL return True without comparing indices tuples
3. WHEN bitmasks differ, THE Partition SHALL return False immediately without further comparison
4. IF the other object is not a Partition, THEN THE Partition SHALL return NotImplemented
5. THE Partition equality check SHALL complete in O(1) time using bitmask comparison

### Requirement 2: Optimize Partition Hashing

**User Story:** As a developer, I want faster Partition hashing, so that the 9M `__hash__` calls are more efficient.

#### Acceptance Criteria

1. WHEN a Partition is created, THE Partition SHALL compute and cache its hash value immediately
2. WHEN `__hash__` is called, THE Partition SHALL return the cached hash value without recomputation
3. THE Partition hash SHALL be based on the bitmask for consistency with equality
4. THE Partition `__hash__` method SHALL complete in O(1) time by returning the cached value

### Requirement 3: Maintain Correctness

**User Story:** As a developer, I want the optimized Partition operations to maintain correctness, so that existing functionality is preserved.

#### Acceptance Criteria

1. FOR ALL pairs of Partitions where `p1 == p2`, THE hash values SHALL be equal (`hash(p1) == hash(p2)`)
2. FOR ALL Partitions, THE equality and hash operations SHALL be consistent with current behavior
3. WHEN Partitions are used in sets or as dictionary keys, THE behavior SHALL remain unchanged
4. THE optimized Partition SHALL pass all existing unit tests
