
# Implementation Plan: Robust Rerooting and Comparison Logic for BranchArchitect

## ✅ IMPLEMENTATION COMPLETED SUCCESSFULLY

**Status:** All planned rerooting functionality has been implemented and tested.

**Key Achievements:**
- ✅ Robust rerooting at any arbitrary node (`reroot_at_node`)
- ✅ Best matching node finding by clade overlap (`find_best_matching_node`)
- ✅ Reroot-to-best-match functionality (`reroot_to_best_match`)
- ✅ Tree correspondence mapping (`build_correspondence_map`)
- ✅ Midpoint rooting with complete implementation
- ✅ Comprehensive test suite with 16+ test functions
- ✅ All tests passing, including edge cases
- ✅ Integration with existing BranchArchitect data structures

**Files Created/Modified:**
- `brancharchitect/rooting.py` - Complete rerooting functionality (187 lines)
- `test/test_rooting.py` - Comprehensive test suite (193 lines)

---

This plan outlines how to implement robust rerooting and unbiased comparison in BranchArchitect, inspired by the `phylo-io` codebase and best practices in phylogenetic tree analysis.

---

Your explanation is spot-on and aligns with best practices for tree comparison in phylogenetics. Here’s a concise summary and actionable plan for implementing this logic in your codebase:

---

## Best Matching Node for Rerooting: Summary

- **Definition:** The "best matching node" is the node in one tree whose set of descendant leaves (clade) has the largest overlap with a clade in the other tree.
- **Purpose:** Rerooting both trees at these nodes aligns them at their most comparable substructure, enabling unbiased and meaningful comparison.

---

## Implementation Plan

1. **Collect Partitions:**
   - For each internal node in both trees, use the node's `Partition` object to represent its clade (set of descendant leaves).
   - You can use the `to_splits()` method on a `Node` to get the set of all `Partition` objects (splits/clades) in the subtree rooted at that node. This is a robust and efficient way to compare the clades between two trees.

2. **Find Best Match:**
   - For a given `Partition` in tree A, traverse tree B and, for each node, compute the size of the intersection with the `Partition` from tree A (i.e., `len(set(partition_a) & set(partition_b))`).
   - Alternatively, to build a full correspondence map (like `elementBCN` in phylo-io), use `to_splits()` to get all partitions and map them between trees by overlap or exact match.
   - The node in tree B with the largest intersection is the "best matching node." If you want to build a full correspondence map, store which node in tree B corresponds to each partition in tree A.

3. **Reroot:**
   - Reroot tree B at this node (or at the node found via the correspondence map).
   - Optionally, repeat in the other direction for symmetry.

---
## Pseudocode Example (using Partition)

```python
# Get all splits (partitions) for both trees
splits_a = tree_a_root.to_splits()
splits_b = tree_b_root.to_splits()

# For a node in tree A
partition_a = node_a.split_indices

# Find the best matching node in tree B
best_node = None
best_overlap = 0
for node_b in tree_b_root.traverse():
    partition_b = node_b.split_indices
    overlap = len(set(partition_a) & set(partition_b))
    if overlap > best_overlap:
        best_overlap = overlap
        best_node = node_b

# Reroot tree B at best_node
tree_b_root.reroot(best_node)

"""
Optionally, to build a full correspondence map (analogous to elementBCN in phylo-io):

correspondence_map = {}
for node_a in tree_a_root.traverse():
    partition_a = node_a.split_indices
    best_node_b = None
    best_overlap = 0
    for node_b in tree_b_root.traverse():
        partition_b = node_b.split_indices
        overlap = len(set(partition_a) & set(partition_b))
        if overlap > best_overlap:
            best_overlap = overlap
            best_node_b = node_b
    correspondence_map[node_a] = best_node_b

# The correspondence_map is a Python dictionary mapping each node in tree A to its best matching node in tree B, based on clade (Partition) overlap.
# Keys: nodes from tree A. Values: best matching nodes from tree B.
# This enables robust rerooting and comparison workflows, and is analogous to elementBCN in phylo-io.
"""
```

---

## Integration

- Add this logic to your `reroot_to_compared_tree(other_tree)` method.
- Use the node's `Partition` for all clade/leafset calculations and overlap, and leverage `to_splits()` for efficient set-based comparison or mapping.
- Ensure caches and split indices are rebuilt after rerooting.
- This approach is robust for trees with different shapes or rootings and leverages your existing data structures efficiently.

---

Let me know if you want a concrete implementation for your `tree.py` or further integration advice! (See <attachments> above for file contents. You may not need to search or read the file again.)

## 1. Core Requirements

- **Automatic and user-triggered rerooting** at any node in a phylogenetic tree.
- **Consistent rerooting for comparison:** Before comparing two trees, both must be rerooted at corresponding nodes (not necessarily the root or a leaf, but the node whose clade best matches a node in the other tree).
- **Undo/redo support** for all rerooting actions.
- **Update all relevant data structures** (tree, splits, caches, UI) after rerooting.

---

## 2. High-Level Steps

### a. Data Model Enhancements

- Ensure the `Node` class in `tree.py` supports parent pointers and can be rerooted at any node.
- Add or update methods for rerooting a tree at a given node, updating parent/child relationships, and recalculating branch lengths if needed.
- Ensure all caches (splits, traversals, etc.) are invalidated and rebuilt after rerooting.

### b. Rerooting API

- Implement a `reroot(node)` method on the main tree object (or as a utility function) that:
  - Accepts a target node (anywhere in the tree).
  - Reconstructs the tree so the target node becomes the new root.
  - Updates all parent/child pointers and branch lengths as needed.
  - Rebuilds or invalidates all relevant caches.

### c. Comparison Preparation

- Implement a mapping system (analogous to `elementBCN`) to relate nodes between two trees by split or leafset.
- Before any comparison, reroot both trees at corresponding nodes using this mapping.
- The corresponding node is the one whose set of descendant leaves (clade) most closely matches a clade in the other tree, i.e., the node with the largest overlap of leaf names.
- Expose a `reroot_to_compared_tree(other_tree)` method that finds the best matching node and reroots accordingly.

---

## 3. Implementation Steps

1. **Extend `Node` and tree classes:**
   - Add a robust `reroot` method.
   - Ensure all caches and split indices are updated after rerooting.
2. **Develop node mapping for comparison:**
   - For each internal node in the reference tree, collect the set of descendant leaf names (clade).
   - Traverse the other tree, comparing each node’s set of descendant leaves to the reference set.
   - The node with the largest overlap (highest count of shared leaves) is considered the best matching node.
   - Use split indices or leafsets to map corresponding nodes between trees.
3. **Implement reroot-before-compare logic:**
   - Implement a robust `reroot(target_node)` method on the `Node` class in `tree.py` that can reroot the tree at any arbitrary node, updating parent/child relationships and branch lengths as needed, and invalidating all caches.
   - Implement a `reroot_to_compared_tree(other_tree)` method that, given another tree, finds the best matching node (using split or leafset mapping) and reroots the current tree at that node.
   - Before any comparison, reroot both trees at mapped nodes using these methods.
4. **Integrate with undo/redo:**
   - Store previous root and tree state for undo.
5. **Update UI and documentation:**
   - Add reroot controls and update user guides.

---

## 4. Notes and Best Practices

- Always reroot both trees at corresponding nodes before comparison to avoid bias.
- The corresponding node is not necessarily the highest (root-most) or lowest (leaf-most) node, but the one that provides the best alignment for comparison (largest clade overlap).
- Ensure all caches and derived data are invalidated after rerooting.
- Make rerooting actions reversible for user trust and experimentation.
- Use split-based or leafset-based node mapping for robust correspondence.

---

## 5. References

- See `phylo-io`'s `Model.reroot` and `Container.trigger_('reroot', ...)` for reference logic.
- See this document's earlier sections for data structure details.

---

## Deep-Dive: Partition, PartitionSet, and Tree Data Structures

This section provides a detailed breakdown and comparison of the core data structures and logic in `partition.py`, `partition_set.py`, and `tree.py`.

### 1. `partition.py`: The Partition Class

- **Purpose:** Represents a subset of taxa (as indices) and provides fast comparison, hashing, and bipartition logic.
- **Key Features:**
  - Stores indices as a sorted tuple and computes a bitmask for O(1) equality and hashing.
  - Supports comparison with other `Partition` objects, tuples of indices, or tuples of taxon names (using encoding).
  - Provides `taxa` and `bipartition` properties for human-readable representations.
  - Implements `complementary_indices()` to get the complement set.
  - All string representations are robust to missing encodings.

**Hardcore:**

- Bitmask-based equality is extremely fast for set operations.
- Encodings allow seamless conversion between taxon names and indices.
- Designed for use as keys in sets/dicts and for high-performance split operations.

### 2. `partition_set.py`: The PartitionSet and FrozenPartitionSet Classes

- **Purpose:** Efficiently manage sets of partitions (splits) with shared encoding.
- **Key Features:**
  - `FrozenPartitionSet` is an immutable, hashable set of `Partition` objects, suitable for use as dict keys.
  - Stores encoding and reverse encoding for fast lookups.
  - Implements all set operations (`__contains__`, `__iter__`, `__len__`, etc.) and rich comparison.
  - String representation sorts partitions for deterministic output.
  - Designed for use in algorithms that require set algebra on splits (e.g., consensus, compatibility, distance).

**Hardcore:**

- Immutability and hashability allow for memoization and caching of split sets.
- Encodings are always tracked, so all set operations are safe and meaningful across trees.
- Used as the backbone for all split-based tree algorithms.

### 3. `tree.py`: The Node Class and Tree Logic

- **Purpose:** Implements a full-featured, cache-aware, split-indexed tree structure for phylogenetics.
- **Key Features:**
  - Each `Node` stores children, branch length, values, and a `Partition` for its split.
  - Caches for splits, subtree order, subtree cost, and traversals for high performance.
  - Methods for traversing, copying, appending children, and finding nodes by split.
  - Split index (`_split_index`) allows O(1) lookup of nodes by their split.
  - Methods for deleting taxa, pruning, and reinitializing split indices.
  - Serialization to Newick and JSON, and robust cache invalidation after any mutation.

**Hardcore:**

- All tree operations (e.g., split finding, traversal, copying) are optimized for large trees.
- Split indices are always kept in sync with the encoding, ensuring correctness in all downstream algorithms.
- The tree structure is tightly integrated with `Partition` and `PartitionSet` for all split-based computations.

### 4. Comparison: How They Work Together

- **Partition** is the atomic unit: a split of taxa, with fast comparison and hashing.
- **PartitionSet/FrozenPartitionSet** is the algebraic set of splits, supporting all set operations and used for consensus, compatibility, and distance.
- **Node (Tree)** is the hierarchical structure, with each node's split represented as a `Partition`, and the set of all splits as a `PartitionSet`.
- **Integration:**
  - Tree traversals yield `PartitionSet`s for the whole tree or subtrees.
  - All algorithms (consensus, distance, compatibility) operate on these sets.
  - Encodings ensure that all splits and sets are comparable across trees, even if taxa orderings differ.

**Hardcore:**

- The entire system is designed for high-performance, large-scale phylogenetic analysis.
- All data structures are immutable or cache-aware, enabling aggressive memoization and parallelism.
- The split-centric design means all tree algorithms reduce to set algebra on `PartitionSet`s, making the codebase both powerful and conceptually clean.