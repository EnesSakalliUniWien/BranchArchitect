# S-Edge Distance Tests

This directory contains comprehensive tests for the s-edge distance calculation functionality in the BranchArchitect tree interpolation pipeline.

## Test Files

### `test_s_edge_distances.py`
**Core functionality tests** for the `_calculate_s_edge_distances()` function:

- ✅ **Basic distance calculation** with simple trees
- ✅ **Zero distance handling** when component and s-edge are the same node  
- ✅ **Multiple component averaging** across different distances
- ✅ **Weighted vs topological differences** with uneven branch lengths
- ✅ **Missing splits handling** with graceful 0 distance fallback
- ✅ **Empty solution sets** handling
- ✅ **Multiple solution sets** for the same s-edge
- ✅ **None branch length handling** (treated as 0)
- ✅ **Parametrized tests** for different tree topologies and branch lengths

### `test_s_edge_distances_integration.py`
**Pipeline integration tests** to verify end-to-end functionality:

- ✅ **Pipeline includes s-edge distances** in TreePairSolution results
- ✅ **Distance metrics consistency** across different tree pairs
- ✅ **Different topology handling** with meaningful distance patterns
- ✅ **Empty s-edge distances** for similar/identical trees
- ✅ **Performance testing** to ensure reasonable calculation speed
- ✅ **Weighted vs topological insights** comparison
- ✅ **Field validation** and mathematical relationship verification

### `test_s_edge_distances_exact.py`
**Exact value verification** for the specific example from the implementation:

- ✅ **Exact (C, D) split distances**: Target 1.0, Reference 1.0, Total 2.0 (topological)
- ✅ **Exact weighted distances**: Target 0.000, Reference 0.100, Total 0.100
- ✅ **Component count verification**: 2 jumping taxa
- ✅ **All 7 metrics present** with correct types and relationships
- ✅ **Structural vs evolutionary distance** differentiation

## Verified Functionality

### ✅ **Enhanced Distance Calculation**
1. **Topological Distance** (unweighted): Counts edges in jump paths
2. **Weighted Distance**: Sums branch lengths along jump paths  
3. **Dual Tree Analysis**: Calculates distances in both target and reference trees
4. **Comprehensive Metrics**: Returns 7 values per s-edge

### ✅ **Seven Distance Metrics Per S-Edge**
- `target_topological`: Average edge count in target tree
- `target_weighted`: Average branch length sum in target tree
- `reference_topological`: Average edge count in reference tree  
- `reference_weighted`: Average branch length sum in reference tree
- `total_topological`: Sum of target + reference topological
- `total_weighted`: Sum of target + reference weighted
- `component_count`: Number of jumping taxa for this s-edge

### ✅ **Robust Implementation**
- ✅ **Correct attribute usage**: Uses `node.length` instead of `node.branch_length`
- ✅ **Edge case handling**: None branch lengths, missing splits, empty solutions
- ✅ **Mathematical consistency**: Totals equal sums of components
- ✅ **Type safety**: Proper numeric types for all distance values

### ✅ **Pipeline Integration**
- ✅ **Structured return types**: Uses `TreeInterpolationSequence` dataclass
- ✅ **Proper data flow**: S-edge distances flow from interpolation to TreePairSolution
- ✅ **Backward compatibility**: All existing functionality preserved
- ✅ **Performance**: Negligible overhead for distance calculations

## Example Output

For the s-edge `(C, D)` in `pair_0_1`:
```
Topological: target=1.0, reference=1.0, total=2.0
Weighted: target=0.000, reference=0.100, total=0.100  
Components: 2
```

This demonstrates that:
- **Structural complexity** (topological): 1 edge jump in each tree
- **Evolutionary distance** (weighted): Different branch length traversal costs
- **Algorithm insight**: 2 jumping taxa (C and D) need to move during interpolation

## Running Tests

```bash
# Run with pytest (if available)
pytest test/distances/ -v

# Run individual test files directly
python test/distances/test_s_edge_distances.py
python test/distances/test_s_edge_distances_integration.py
python test/distances/test_s_edge_distances_exact.py
```

Note: Tests are written to work both with and without pytest for maximum compatibility.