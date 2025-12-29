# Design Document

## Overview

This design simplifies the tree split application and path execution logic by trusting the planning phase's path computation and removing defensive retry logic. The key insight is that `collapse_path` and `expand_path` are guaranteed compatible by construction, so execution should be straightforward: collapse first, then expand.

**Module Organization:**
- `split_application.py` - Split application (expanding topology) - NO collapse operations
- `consensus_tree.py` - Collapse operations and consensus tree functions

## Architecture

### Current Architecture (Complex)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    apply_split_in_tree (current)                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  1. Re-encode split if needed                               │    │
│  │  2. Try to apply split                                      │    │
│  │  3. IF validate=True:                                       │    │
│  │     ├─ Check if split in tree                               │    │
│  │     ├─ IF not: find incompatible splits                     │    │
│  │     ├─ Collapse incompatible splits                         │    │
│  │     └─ RECURSIVE CALL to apply_split_in_tree ← PROBLEM      │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### Proposed Architecture (Simple)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Path Execution (simplified)                       │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  execute_path(tree, collapse_path, expand_path, dest_tree)  │    │
│  │  [in consensus_tree.py]                                     │    │
│  │                                                              │    │
│  │  1. execute_collapse_path(tree, collapse_path, dest_tree)   │    │
│  │     ├─ Set all collapse splits to length=0                  │    │
│  │     └─ Collapse zero-length branches (preserve dest splits) │    │
│  │     [uses existing collapse functions in consensus_tree.py] │    │
│  │                                                              │    │
│  │  2. execute_expand_path(tree, expand_path, dest_weights)    │    │
│  │     ├─ Sort by size (largest first)                         │    │
│  │     ├─ Apply each split (fail fast on error)                │    │
│  │     └─ Apply reference weights                              │    │
│  │     [in split_application.py]                               │    │
│  │                                                              │    │
│  │  3. Return modified tree                                    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  apply_split_simple(split, node)  ← NO RETRY, NO COLLAPSE   │    │
│  │  [in split_application.py]                                  │    │
│  │                                                              │    │
│  │  1. Check if already present (idempotent)                   │    │
│  │  2. Find correct parent node                                │    │
│  │  3. Create new internal node                                │    │
│  │  4. IF fails: raise error immediately                       │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### Component 1: Simple Split Application (split_application.py)

```python
def apply_split_simple(split: Partition, node: Node) -> None:
    """
    Apply a split to a tree without retry or validation.

    Args:
        split: The partition to apply
        node: The root node of the tree

    Raises:
        SplitApplicationError: If split cannot be applied (incompatible topology)
    """
```

### Component 2: Collapse Path Executor (consensus_tree.py)

```python
def execute_collapse_path(
    tree: Node,
    collapse_path: List[Partition],
    destination_tree: Optional[Node] = None,
) -> Node:
    """
    Execute collapse path by setting splits to zero length then collapsing.

    Args:
        tree: The tree to modify (will be mutated)
        collapse_path: Splits to collapse
        destination_tree: If provided, preserve splits that exist here

    Returns:
        The modified tree with collapse splits removed
    """
```

### Component 3: Expand Path Executor

```python
def execute_expand_path(
    tree: Node,
    expand_path: List[Partition],
    reference_weights: Dict[Partition, float],
) -> Node:
    """
    Execute expand path by applying splits in size order.

    Args:
        tree: The tree to modify (will be mutated)
        expand_path: Splits to apply
        reference_weights: Weights to apply to new nodes

    Returns:
        The modified tree with expand splits added

    Raises:
        SplitApplicationError: If any split fails to apply
    """
```

### Component 4: Unified Path Executor

```python
def execute_path(
    tree: Node,
    collapse_path: List[Partition],
    expand_path: List[Partition],
    destination_tree: Node,
) -> Node:
    """
    Execute complete collapse→expand path transformation.

    Args:
        tree: The tree to modify (will be mutated)
        collapse_path: Splits to collapse
        expand_path: Splits to expand
        destination_tree: Reference for weights and preservation

    Returns:
        The modified tree with correct topology
    """
```

## Data Models

### SplitApplicationError

```python
@dataclass
class SplitApplicationError(Exception):
    """Error raised when split application fails."""
    split: Partition
    tree_splits: List[Partition]
    message: str

    def __str__(self) -> str:
        taxa_names = [self.split.reverse_encoding[i] for i in self.split.indices]
        return (
            f"{self.message}\n"
            f"Split: {list(self.split.indices)} = ({', '.join(taxa_names)})\n"
            f"Tree has {len(self.tree_splits)} splits"
        )
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Split Application Correctness

*For any* valid tree and *for any* compatible split not already in the tree, applying the split SHALL result in the split being present in the tree's split set.

**Validates: Requirements 1.1, 1.2, 3.2**

### Property 2: Split Application Idempotence

*For any* tree and *for any* split already present in the tree, applying the split SHALL leave the tree unchanged (same split set, same structure).

**Validates: Requirements 1.2**

### Property 3: Collapse Path Correctness

*For any* tree and *for any* collapse_path, after executing the collapse path:

- All splits in collapse_path (that don't exist in destination) SHALL be absent from the tree
- All splits that exist in destination_tree SHALL be preserved

**Validates: Requirements 2.1, 2.2, 2.3**

### Property 4: No Automatic Conflict Resolution

*For any* tree and *for any* incompatible split, attempting to apply the split SHALL raise an error immediately without modifying the tree's topology.

**Validates: Requirements 1.4, 1.5, 3.4, 5.3**

### Property 5: Round-Trip Topology Correctness

*For any* valid (collapse_path, expand_path) pair computed by the planning phase, executing collapse then expand SHALL produce a tree where:

- All splits in expand_path are present
- No splits unique to collapse_path (not in expand_path) are present

**Validates: Requirements 8.1, 8.2**

### Property 6: Weight Application Correctness

*For any* tree and *for any* expand_path with reference weights, after executing the expand path, each newly created node SHALL have the weight from reference_weights.

**Validates: Requirements 4.3**

## Error Handling

### Error Categories

1. **Encoding Errors** (ValueError)
   - Taxa name not found in target encoding
   - Mixed encodings in split set

2. **Application Errors** (SplitApplicationError)
   - Split incompatible with existing topology
   - Split cannot be placed (no valid parent)

3. **Path Errors** (PathExecutionError)
   - Collapse path contains invalid split
   - Expand path fails after collapse

### Error Message Format

```
SplitApplicationError: Cannot apply split - incompatible with existing topology
Split: [0, 1, 5, 6] = (O1, O2, B, B1)
Tree has 12 splits:
  - [0, 1] = (O1, O2)
  - [2, 3, 4] = (A, A1, A2)
  ...
Incompatible splits:
  - [0, 1, 2, 3] = (O1, O2, A, A1) - overlaps without containment
```

## Testing Strategy

### Unit Tests

1. **apply_split_simple**
   - Test successful application
   - Test idempotence (already present)
   - Test encoding re-encoding
   - Test error on incompatible split

2. **execute_collapse_path**
   - Test single split collapse
   - Test multiple split collapse
   - Test preservation of destination splits
   - Test empty collapse path

3. **execute_expand_path**
   - Test single split expand
   - Test multiple split expand (ordering)
   - Test weight application
   - Test error propagation

4. **execute_path**
   - Test complete collapse→expand
   - Test with real tree pairs

### Property-Based Tests

Using `hypothesis` library for Python:

```python
@given(valid_tree_and_compatible_split())
def test_split_application_correctness(tree, split):
    """Property 1: Split application adds split to tree."""
    apply_split_simple(split, tree)
    assert split in tree.to_splits()

@given(valid_tree_with_split())
def test_split_application_idempotence(tree, split):
    """Property 2: Applying existing split is idempotent."""
    original_splits = tree.to_splits()
    apply_split_simple(split, tree)
    assert tree.to_splits() == original_splits

@given(valid_collapse_expand_pair())
def test_round_trip_correctness(tree, collapse_path, expand_path, dest_tree):
    """Property 6: Collapse then expand produces correct topology."""
    result = execute_path(tree, collapse_path, expand_path, dest_tree)
    result_splits = result.to_splits()
    for split in expand_path:
        assert split in result_splits
    for split in collapse_path:
        if split not in expand_path:
            assert split not in result_splits
```

### Test Configuration

- Minimum 100 iterations per property test
- Tag format: **Feature: split-application-simplification, Property {N}: {description}**

## Migration Plan

### Phase 1: Add New Functions

1. Add `apply_split_simple` alongside existing `apply_split_in_tree`
2. Add `execute_collapse_path` and `execute_expand_path`
3. Add `execute_path` unified executor
4. Add comprehensive tests

### Phase 2: Update Callers

1. Update `build_microsteps_for_selection` to use new executor
2. Update `create_subtree_grafted_tree` to use `apply_split_simple`
3. Keep `apply_split_in_tree` for backward compatibility (deprecated)

### Phase 3: Remove Old Code

1. Remove retry logic from `apply_split_in_tree`
2. Remove `_collapse_split_in_tree` (replaced by `execute_collapse_path`)
3. Update all remaining callers

## Alternatives Considered

### Alternative 1: Keep Retry Logic but Limit Depth

Add a max retry depth to prevent infinite loops.

**Rejected because:**
- Still masks bugs in path computation
- Adds complexity without addressing root cause
- Retry should never be needed if paths are correct

### Alternative 2: Validate Paths Before Execution

Add a validation step that checks path compatibility before execution.

**Rejected because:**
- Redundant with planning phase validation
- Adds overhead without benefit
- Planning phase already ensures compatibility

### Alternative 3: Lazy Collapse (On-Demand)

Only collapse splits when they conflict with an expand split.

**Rejected because:**
- More complex execution logic
- Harder to reason about intermediate states
- Current "collapse all first" is simpler and correct
