# Design Document

## Overview

This design formalizes the circular layout optimization system for phylogenetic tree visualization. The system determines optimal "cut points" for circular arrangements and maintains rotation stability across animation frames.

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Circular Layout Optimization                          │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  blocked_order_and_apply() - Entry Point                           │ │
│  │                                                                     │ │
│  │  if circular:                                                       │ │
│  │    ┌──────────────────────────────────────────────────────────────┐│ │
│  │    │ _apply_circular_rotation()                                   ││ │
│  │    │                                                               ││ │
│  │    │  ┌─────────────────────┐  ┌─────────────────────────────────┐││ │
│  │    │  │ Policy Selection    │  │ Cache Management                │││ │
│  │    │  │                     │  │                                 │││ │
│  │    │  │ between_anchor_     │  │ _rotation_cut_cache             │││ │
│  │    │  │   blocks            │  │   Key: edge.indices             │││ │
│  │    │  │                     │  │   Val: (src_cut, dst_cut, taxa) │││ │
│  │    │  │ largest_mover_      │  │                                 │││ │
│  │    │  │   at_zero           │  │ Cache hit → reuse rotation      │││ │
│  │    │  └─────────────────────┘  │ Cache miss → compute new        │││ │
│  │    │                           └─────────────────────────────────┘││ │
│  │    └──────────────────────────────────────────────────────────────┘│ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  circular_distances.py - Distance Metrics                          │ │
│  │                                                                     │ │
│  │  ┌─────────────────────┐  ┌─────────────────────────────────────┐  │ │
│  │  │ circular_distance() │  │ circular_distances_trees()          │  │ │
│  │  │                     │  │                                     │  │ │
│  │  │ Pairwise distance   │  │ Aggregate distance for tree list   │  │ │
│  │  │ with wrap-around    │  │ Total or pairwise output           │  │ │
│  │  └─────────────────────┘  └─────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Input: sorted_src, sorted_dst (linear orders)
       edge, jumping_partitions, key maps
       circular_boundary_policy
                │
                ▼
┌───────────────────────────────────────┐
│ Policy Selection                       │
│                                        │
│ if policy == "largest_mover_at_zero": │
│   → _boundary_largest_mover_at_zero() │
│ else:                                  │
│   → _boundary_between_anchor_blocks() │
└───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────┐
│ Cache Check                            │
│                                        │
│ cached = _rotation_cut_cache.get(key) │
│ if cached and taxa unchanged:          │
│   → reuse cached cuts                  │
│ else:                                  │
│   → use computed cuts                  │
└───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────┐
│ Cache Update                           │
│                                        │
│ _rotation_cut_cache[key] =            │
│   (src_cut, dst_cut, tuple(sorted_src))│
└───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────┐
│ Apply Rotation                         │
│                                        │
│ rotated_src = _rotate_list(src, cut)  │
│ rotated_dst = _rotate_list(dst, cut)  │
└───────────────────────────────────────┘
                │
                ▼
Output: rotated_src, rotated_dst
```

## Design Details

### Boundary Policy: Between Anchor Blocks

The `_boundary_between_anchor_blocks` function searches for an optimal cut point:

```python
def _boundary_between_anchor_blocks(
    order: List[str], key_map: Dict[str, Tuple[int, int, int]]
) -> int:
    """
    Find a cut point that splits two different anchor blocks.

    Algorithm:
    1. Iterate through all positions (i, i+1) in circular order
    2. Check if both positions are in Band 1 (anchors)
    3. Check if they belong to different blocks (different rank)
    4. If found, return position (i+1) % n
    5. Fallback: find any band boundary
    6. Final fallback: return 0
    """
    n = len(order)
    if n == 0:
        return 0

    # Primary: Cut between different anchor blocks
    for i in range(n):
        a, b = order[i], order[(i + 1) % n]
        if (key_map[a][0] == 1 and      # a is anchor (Band 1)
            key_map[b][0] == 1 and      # b is anchor (Band 1)
            key_map[a][1] != key_map[b][1]):  # different blocks
            return (i + 1) % n

    # Fallback: Cut at any band boundary
    for i in range(n):
        if key_map[order[i]][0] != key_map[order[(i + 1) % n]][0]:
            return (i + 1) % n

    return 0
```

### Boundary Policy: Largest Mover at Zero

The `_boundary_largest_mover_at_zero` function positions the largest mover prominently:

```python
def _boundary_largest_mover_at_zero(
    order: List[str], mover_blocks: List[Partition]
) -> int:
    """
    Find cut point that places the largest mover block at Index 0.

    Algorithm:
    1. Sort mover blocks by size (descending), then by indices (tie-breaker)
    2. Get the largest block's taxa
    3. Find the first occurrence of any taxon from that block
    4. Return that position as the cut point
    """
    if not mover_blocks:
        return 0

    largest = sorted(
        mover_blocks,
        key=lambda p: (-len(p.indices), p.indices)
    )[0]

    block_taxa = set(largest.taxa)
    for i, t in enumerate(order):
        if t in block_taxa:
            return i

    return 0
```

### Rotation Caching Strategy

The cache prevents visual "spinning" during animation:

```python
# Cache structure
_rotation_cut_cache: Dict[
    Tuple[int, ...],                    # edge.indices as key
    Tuple[int, int, Tuple[str, ...]]    # (src_cut, dst_cut, taxa_tuple)
] = {}

# Cache logic in _apply_circular_rotation:
edge_key = tuple(edge.indices)
cached = _rotation_cut_cache.get(edge_key)

if cached and cached[2] == tuple(sorted_src):
    # Taxa unchanged - reuse cached cuts for stability
    c_src, c_dst, _ = cached
    if src_cut == 0 and c_src:
        src_cut = c_src
    if dst_cut == 0 and c_dst:
        dst_cut = c_dst

# Always update cache with current state
_rotation_cut_cache[edge_key] = (src_cut, dst_cut, tuple(sorted_src))
```

### Circular Distance Formula

The normalized circular distance accounts for wrap-around:

```
For each element e in the order:
  diff = |position_in_x(e) - position_in_y(e)|
  circular_diff = min(diff, n - diff)

Total distance = sum(circular_diff for all elements)
Normalized = Total / (n * (n // 2))
```

This produces a value in [0, 1] where:
- 0 = identical orders
- 1 = maximally different (reversed order)

### Configuration

Circular layout is enabled via the `circular` parameter:

```python
def optimize_with_anchor_ordering(
    self,
    anchor_weight_policy: str = "destination",
    circular: bool = False,  # Enable circular optimization
    circular_boundary_policy: str = "between_anchor_blocks",
    pre_align: bool = True,
) -> None:
```

### Affected Components

| Component | Role | Changes |
|-----------|------|---------|
| `anchor_order.py` | Core circular logic | `_apply_circular_rotation`, boundary functions |
| `circular_distances.py` | Distance metrics | `circular_distance`, `circular_distances_trees` |
| `tree_order_optimiser.py` | Integration | Pass `circular` params to `derive_order_for_pair` |

## Test Cases

### Test Case 1: Between Anchor Blocks Policy

**Input:**
- Order: [A, B, C, D, E, F] where {A,B,C} and {D,E,F} are anchor blocks
- Policy: `between_anchor_blocks`

**Expected:**
- Cut point at position 3 (between C and D)
- Rotated order: [D, E, F, A, B, C]

### Test Case 2: Largest Mover at Zero Policy

**Input:**
- Order: [A, B, X, Y, Z, C, D] where {X,Y,Z} is the largest mover
- Policy: `largest_mover_at_zero`

**Expected:**
- Cut point at position 2 (first occurrence of mover)
- Rotated order: [X, Y, Z, C, D, A, B]

### Test Case 3: Cache Stability

**Input:**
- Same edge processed twice with identical taxa

**Expected:**
- Second call returns same rotation as first (cache hit)
- No visual "spinning" between frames

### Test Case 4: Circular Distance

**Input:**
- Order X: (A, B, C, D)
- Order Y: (A, D, C, B)

**Expected:**
- Distance accounts for wrap-around
- B moved 2 positions (or equivalently 2 the other way)
- D moved 2 positions
- Normalized result in [0, 1]

## Alternatives Considered

### Alternative 1: Always Cut at Position 0

Simply don't rotate - use the linear order as-is.

**Rejected because:**
- May split anchor blocks across the boundary
- Doesn't optimize for visual clarity

### Alternative 2: Minimize Total Circular Distance

Compute all possible rotations and pick the one minimizing distance to reference.

**Rejected because:**
- O(n²) complexity for each rotation decision
- May cause instability between frames (different "optimal" each time)

### Alternative 3: User-Specified Cut Point

Let users manually specify where to cut.

**Rejected because:**
- Requires user intervention for each tree
- Doesn't scale for animations with many frames

