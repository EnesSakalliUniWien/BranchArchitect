# Design Document

## Overview

This design addresses the tree interpolation ordering issue where non-moving taxa can appear displaced due to Newick encoding order differences between source and destination trees. The solution introduces a pre-alignment step that ensures destination trees match source order before anchor calculations.

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TreeOrderOptimizer                                │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  optimize_with_anchor_ordering(pre_align=True)              │    │
│  │                                                              │    │
│  │  For each tree pair (i, i+1):                               │    │
│  │    ┌──────────────────────────────────────────────────────┐ │    │
│  │    │ 1. PRE-ALIGNMENT (NEW)                               │ │    │
│  │    │    source_order = trees[i].get_current_order()       │ │    │
│  │    │    trees[i+1].reorder_taxa(source_order)             │ │    │
│  │    └──────────────────────────────────────────────────────┘ │    │
│  │                          ↓                                   │    │
│  │    ┌──────────────────────────────────────────────────────┐ │    │
│  │    │ 2. ANCHOR ORDERING (existing)                        │ │    │
│  │    │    derive_order_for_pair(trees[i], trees[i+1])       │ │    │
│  │    └──────────────────────────────────────────────────────┘ │    │
│  │                          ↓                                   │    │
│  │    ┌──────────────────────────────────────────────────────┐ │    │
│  │    │ 3. PROPAGATION (existing)                            │ │    │
│  │    │    _propagate_from_index_backward/forward            │ │    │
│  │    └──────────────────────────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                          ↓                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  final_pairwise_alignment_pass(trees)                       │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Source Tree (T1)          Destination Tree (T2)
     │                           │
     │                           │
     ▼                           ▼
get_current_order()        reorder_taxa(source_order)
     │                           │
     │                           │
     └─────────┬─────────────────┘
               │
               ▼
     derive_order_for_pair(T1, T2_aligned)
               │
               ▼
     Anchor-based ordering applied
               │
               ▼
     Only structurally necessary taxa move
```

## Design Details

### Pre-Alignment Strategy

The pre-alignment step is inserted at the beginning of each tree pair processing in `optimize_with_anchor_ordering`:

```python
def optimize_with_anchor_ordering(
    self,
    anchor_weight_policy: str = "destination",
    circular: bool = False,
    circular_boundary_policy: str = "between_anchor_blocks",
    pre_align: bool = True,  # NEW PARAMETER
) -> None:
    for i in range(n - 1):
        # NEW: Pre-align destination to source order
        if pre_align:
            source_order = self.trees[i].get_current_order()
            if source_order != self.trees[i + 1].get_current_order():
                self.trees[i + 1].reorder_taxa(source_order)

        # Existing: Apply anchor-based ordering
        derive_order_for_pair(self.trees[i], self.trees[i + 1], ...)
```

### Why Pre-Alignment Works

1. **Topology Preservation**: `reorder_taxa()` only changes leaf order, not tree topology (splits remain identical)

2. **Anchor Consistency**: When destination is aligned to source:
   - Non-moving taxa have identical positions in both trees
   - Anchor calculations use consistent reference points
   - Only structurally necessary movements are computed

3. **No Effect on Heavy Cases**: When multiple taxa move, the anchor algorithm already handles ordering correctly. Pre-alignment is a no-op in these cases.

4. **Fixes Light Cases**: When only one taxon moves, pre-alignment ensures the destination's non-moving taxa match source positions, preventing spurious visual displacement.

### Configuration

The `pre_align` parameter is exposed through `PipelineConfig`:

```python
@dataclass
class PipelineConfig:
    # ... existing fields ...
    pre_align: bool = True  # NEW: Enable pre-alignment by default
```

### Affected Components

| Component | Change | Impact |
|-----------|--------|--------|
| `TreeOrderOptimizer.optimize_with_anchor_ordering` | Add pre-alignment step | Core fix |
| `PipelineConfig` | Add `pre_align` field | Configuration |
| `TreeInterpolationPipeline` | Pass `pre_align` to optimizer | Integration |

## Test Cases

### Test Case 1: Single Mover (bird_trees)

**Input:**
- Source: Ostrich between Birds and Moas clades
- Destination: Ostrich moved to different position

**Expected:**
- Only Ostrich moves visually
- Birds and Moas clades maintain source positions
- Final topology matches destination

### Test Case 2: Multiple Movers (reverse_bootstrap_52)

**Input:**
- 3 movers of sizes 1, 3, and 7 taxa

**Expected:**
- All movers move to correct positions
- Non-movers maintain source positions
- Pre-alignment has no effect (already works)

### Test Case 3: Complex Movers (heiko_4)

**Input:**
- 4+ movers with nested structure

**Expected:**
- All movers coordinated correctly
- Pre-alignment has no effect (already works)

## Alternatives Considered

### Alternative 1: Post-Interpolation Reordering

Reorder the final tree to match expected destination order after interpolation.

**Rejected because:**
- Would cause visual "jump" at the end of animation
- Doesn't address root cause (anchor calculation inconsistency)

### Alternative 2: Modify Anchor Algorithm

Change `derive_order_for_pair` to handle Newick order differences internally.

**Rejected because:**
- More complex implementation
- Would require changes to well-tested anchor logic
- Pre-alignment is simpler and equally effective

### Alternative 3: Always Use Destination Order

Force all trees to use destination's Newick order.

**Rejected because:**
- Would cause non-moving taxa to appear displaced
- Violates "minimal movement" principle
