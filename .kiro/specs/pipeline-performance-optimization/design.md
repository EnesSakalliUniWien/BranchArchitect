# Design Document: Partition Operations Optimization

## Overview

This design optimizes the `Partition` class's `__eq__` and `__hash__` methods to reduce overhead from the ~12M equality checks and ~9M hash computations identified in profiling. The key optimization is caching the hash value at construction time, eliminating repeated `hash(self.bitmask)` calls.

## Architecture

The optimization is localized to the `Partition` class in `brancharchitect/elements/partition.py`. No architectural changes are needed - this is a targeted performance improvement to existing methods.

### Current Implementation Analysis

**Current `__hash__`:**
```python
def __hash__(self) -> int:
    return hash(self.bitmask)
```
- Computes `hash(self.bitmask)` on every call
- With 9M calls, this adds up to significant overhead

**Current `__eq__`:**
```python
def __eq__(self, other: Any) -> bool:
    if isinstance(other, Partition):
        return self.bitmask == other.bitmask
    # ... handling for other types
```
- Already uses efficient bitmask comparison for Partition-to-Partition
- The `isinstance` check adds minor overhead but is necessary

### Optimization Strategy

1. **Cache hash at construction**: Add `_cached_hash` slot and compute hash once during `__init__` and `from_bitmask`
2. **Return cached hash**: `__hash__` simply returns the pre-computed value
3. **Maintain correctness**: Hash is based on bitmask, consistent with equality

## Components and Interfaces

### Modified: Partition Class

**New slot:**
```python
__slots__ = (
    "_indices",
    "encoding",
    "bitmask",
    "_cached_reverse_encoding",
    "_cached_size",
    "_cached_hash",  # NEW: cached hash value
)
```

**Modified `__init__`:**
```python
def __init__(self, indices: Tuple[int, ...], encoding: Optional[Dict[str, int]] = None):
    # ... existing code ...
    self.bitmask: int = bitmask
    self._cached_size: int = bitmask.bit_count()
    self._cached_hash: int = hash(bitmask)  # NEW: cache hash at creation
    self._cached_reverse_encoding: Optional[Dict[int, str]] = None
```

**Modified `from_bitmask`:**
```python
@classmethod
def from_bitmask(cls, bitmask: int, encoding: Dict[str, int]) -> Partition:
    obj = cls.__new__(cls)
    obj._indices = None
    obj.encoding = encoding
    obj.bitmask = bitmask
    obj._cached_size = bitmask.bit_count()
    obj._cached_hash = hash(bitmask)  # NEW: cache hash at creation
    obj._cached_reverse_encoding = None
    return obj
```

**Modified `__hash__`:**
```python
def __hash__(self) -> int:
    return self._cached_hash  # Return cached value
```

## Data Models

No changes to data models. The `Partition` class gains one additional integer slot (`_cached_hash`) which has negligible memory impact.

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Hash-Equality Contract

*For any* two Partitions `p1` and `p2`, if `p1 == p2` then `hash(p1) == hash(p2)`.

**Validates: Requirements 3.1**

### Property 2: Bitmask-Based Equality

*For any* two Partitions `p1` and `p2`, `p1 == p2` if and only if `p1.bitmask == p2.bitmask`.

**Validates: Requirements 1.1, 1.2, 1.3**

### Property 3: Hash Consistency

*For any* Partition `p`, calling `hash(p)` multiple times returns the same value.

**Validates: Requirements 2.2, 2.3**

### Property 4: Set Membership Correctness

*For any* Partition `p` and set `s` containing `p`, membership check `p in s` returns True, and for any Partition `q` where `q == p`, `q in s` also returns True.

**Validates: Requirements 3.3**

### Property 5: Dictionary Key Correctness

*For any* Partition `p` used as a dictionary key with value `v`, retrieving with an equal Partition `q` (where `p == q`) returns `v`.

**Validates: Requirements 3.3**

## Error Handling

No new error conditions are introduced. The optimization maintains existing behavior:
- `__eq__` returns `NotImplemented` for incompatible types
- `__hash__` always returns an integer

## Testing Strategy

### Unit Tests
- Verify `_cached_hash` is set after construction via `__init__`
- Verify `_cached_hash` is set after construction via `from_bitmask`
- Verify `__hash__` returns `_cached_hash` value
- Verify existing Partition tests still pass

### Property-Based Tests
- Use Hypothesis to generate random Partitions
- Test hash-equality contract across many random pairs
- Test set and dictionary operations with random Partitions
- Minimum 100 iterations per property test

### Performance Validation
- Run profiling before and after to measure improvement
- Target: Reduce `__hash__` cumtime from current baseline
