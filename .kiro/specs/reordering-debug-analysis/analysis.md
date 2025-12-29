# Reordering Debug Analysis: Trees 3-4 (Ostrich Movement)

## Summary

This analysis investigates why Ostrich, GreatRhea, and LesserRhea appear to move unexpectedly during the interpolation between trees 3 and 4 in `small_example copy 3.tree`.

## Key Findings

### 1. Lattice Solver Output

The lattice solver identifies **1 pivot edge** with **2 jumping taxa groups**:

| Group | Taxa | Description |
|-------|------|-------------|
| JT 1 | `{Dinornis, EasternMoa, lbmoa, ECtinamou, Gtinamou, Crypturellus}` | Moa/Tinamou clade |
| JT 2 | `{Ostrich}` | Single taxon |

### 2. Selection Order Discrepancy

**Lattice solver order:**
1. Moa/Tinamou clade (6 taxa)
2. Ostrich (1 taxon)

**Edge plan selection order:**
1. **Ostrich** ← First!
2. Moa/Tinamou clade

**Why the reversal?** The `PivotSplitRegistry` in `builder.py` uses a priority algorithm (`get_next_subtree()`) that may reorder subtrees based on:
- Collapse path availability
- Expand path dependencies
- Tabula rasa strategy (first subtree collapses everything)

### 3. The Reordering Limitation

The `reorder_tree_toward_destination()` function in `reordering.py` computes the correct target order:

```
Source subtree order: ['GreatRhea', 'LesserRhea', 'Ostrich', 'Caiman', ...]
Target subtree order: ['GreatRhea', 'LesserRhea', 'Caiman', ..., 'LBPenguin', 'Ostrich', ...]
```

However, `reorder_taxa()` **cannot achieve this** because:

1. **Tree topology constraint**: `reorder_taxa()` can only reorder children within a node
2. **Clade boundaries**: Ostrich is in a different clade in source vs destination
3. **Structural limitation**: Moving Ostrich from position 2 to position 12 requires topology change, not just reordering

### 4. Solution Mappings

| Tree | Solution | Mapped To |
|------|----------|-----------|
| Source (t1) | `{Ostrich}` | `{GreatRhea, LesserRhea, Ostrich}` |
| Dest (t2) | `{Ostrich}` | Entire pivot edge (19 taxa) |

This shows that in the **source tree**, Ostrich is grouped with GreatRhea/LesserRhea, but in the **destination tree**, Ostrich is part of a much larger clade.

### 5. Collapse/Expand Paths

**For Ostrich subtree:**
- Collapse paths: 2 (including `{GreatRhea, LesserRhea, Ostrich}`)
- Expand paths: 2 (including `{Ostrich}` and the large clade)

**For Moa/Tinamou subtree:**
- Collapse paths: 1
- Expand paths: 2

## Tree Structure Difference

### Source Tree (Tree 3) - Ostrich's Position
```
Level 3: {GreatRhea, LesserRhea, Ostrich}  ← Ostrich is sibling of (GreatRhea, LesserRhea)
  ├── {GreatRhea, LesserRhea}
  │     ├── GreatRhea
  │     └── LesserRhea
  └── Ostrich  ← HERE
```

### Destination Tree (Tree 4) - Ostrich's Position
```
Level 3: {GreatRhea, LesserRhea, Ostrich, Caiman, Alligator, ...13 taxa}
  ├── {GreatRhea, LesserRhea, Caiman, Alligator, ...12 taxa}
  │     ├── {GreatRhea, LesserRhea}
  │     └── {Caiman, Alligator, ...10 taxa}
  └── Ostrich  ← HERE (sibling of a MUCH LARGER clade)
```

**Key difference**: In the source, Ostrich is grouped with just GreatRhea/LesserRhea. In the destination, Ostrich is grouped with a 12-taxon clade that INCLUDES GreatRhea/LesserRhea plus all the birds.

## Root Cause

The issue is that **reordering alone cannot move Ostrich** to its destination position. The actual movement happens through:

1. **Collapse step (C)**: Removes the `{GreatRhea, LesserRhea, Ostrich}` split
2. **Expand step (IT_up)**: Creates the new split that places Ostrich with the larger clade

The `reorder_tree_toward_destination()` function is called **before** the topology changes, so it cannot achieve the target order.

## Relationship: `microsteps.py` ↔ `anchor_order.py`

### `anchor_order.py` (Pre-interpolation)
- Called during **leaf order optimization** phase
- Uses `blocked_order_and_apply()` to position jumping taxa at extremes
- Assigns "bands" to taxa: band 0 (left), band 1 (anchors), band 2 (right)
- Applies to **both source and destination trees** before interpolation

### `microsteps.py` (During interpolation)
- Called for **each selection** under a pivot edge
- Uses `reorder_tree_toward_destination()` to align the **collapsed** tree
- Operates on the **interpolation state** (modified tree)
- Limited by current tree topology

### The Flow

```
1. anchor_order.py: Pre-align source/dest trees (jumping taxa at extremes)
                    ↓
2. For each pivot edge:
   For each selection (subtree):
     a. IT_down: Zero branch lengths
     b. C: Collapse zero-length branches
     c. C_reorder: reorder_tree_toward_destination() ← LIMITED BY TOPOLOGY
     d. IT_up: Graft new splits
     e. IT_ref: Apply reference weights
```

## Why Ostrich is "First"

1. **Tabula rasa strategy**: First subtree gets all collapse work
2. **Ostrich has collapse paths**: `{GreatRhea, LesserRhea, Ostrich}` needs to be collapsed
3. **Priority algorithm**: Subtrees with collapse work may be prioritized

## Recommendations

1. **The reordering step is working correctly** - it just can't achieve impossible reorderings
2. **The topology change steps** (collapse/expand) are what actually move taxa
3. **The visual "jump"** of Ostrich happens when the topology changes, not during reordering

## Files Involved

| File | Role |
|------|------|
| `brancharchitect/tree_interpolation/subtree_paths/execution/reordering.py` | Reordering logic |
| `brancharchitect/tree_interpolation/subtree_paths/execution/microsteps.py` | 5-step interpolation |
| `brancharchitect/leaforder/anchor_order.py` | Pre-interpolation ordering |
| `brancharchitect/tree_interpolation/subtree_paths/planning/builder.py` | Selection ordering |
| `brancharchitect/jumping_taxa/lattice/solvers/lattice_solver.py` | Jumping taxa detection |
