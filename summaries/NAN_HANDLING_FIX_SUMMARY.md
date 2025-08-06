# NaN Handling Fixes for SpectralClustering Error

## Issue Description
The distance analysis notebook was encountering a **SpectralClustering NaN error** when trying to perform clustering analysis on distance matrices containing NaN values. The error occurred because scikit-learn's SpectralClustering algorithm cannot process matrices with NaN values.

## Root Cause
- Distance matrices computed from phylogenetic trees sometimes contain NaN values
- SpectralClustering requires clean numerical data without missing values
- The original `perform_clustering` function did not handle NaN values

## Solution Implemented

### 1. Enhanced `perform_clustering` Function
**File:** `/Users/berksakalli/Projects/BranchArchitect/notebooks/distance_plot_utils.py`

**Changes:**
- Added NaN detection with warning messages
- Implemented median imputation strategy for NaN values
- Added matrix symmetry enforcement (required for spectral clustering)
- Added protection against division by zero in similarity matrix calculation
- Preserved original matrix by creating copies before modification

**Key features:**
```python
def perform_clustering(distance_matrix, n_clusters=3):
    # Check for NaN values and handle them
    if np.isnan(distance_matrix).any():
        print(f"Warning: Found {np.sum(np.isnan(distance_matrix))} NaN values...")
        
        # Strategy: Replace NaN with median of non-NaN values
        distance_matrix = np.copy(distance_matrix)  # Don't modify original
        median_distance = np.median(distance_matrix[~np.isnan(distance_matrix)])
        distance_matrix[np.isnan(distance_matrix)] = median_distance
    
    # Ensure matrix is symmetric (required for spectral clustering)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    
    # Handle zero standard deviation case
    std_dev = np.std(distance_matrix)
    if std_dev == 0:
        similarity_matrix = np.ones_like(distance_matrix)
    else:
        similarity_matrix = np.exp(-distance_matrix / std_dev)
```

### 2. Added `perform_clustering_robust` Function
**Advanced NaN handling with multiple strategies:**

- **`median`**: Replace NaN with median distance (default)
- **`mean`**: Replace NaN with mean distance  
- **`remove`**: Remove rows/columns with too many NaN values
- **`knn_impute`**: Use KNN imputation for sophisticated missing value handling

**Features:**
- Configurable minimum validity ratio for row/column removal
- Returns both cluster labels and valid indices for tracking
- Comprehensive error handling and fallback mechanisms
- Integration with sklearn's KNNImputer

### 3. Enhanced `perform_umap` Function
**Added consistent NaN handling to UMAP:**
- Same median imputation strategy as clustering
- Matrix symmetry enforcement
- Warning messages for transparency
- Copy-based approach to preserve original data

## Testing Results

### Test Matrix Example:
```
[[0.0, 1.2, nan, 2.5],
 [1.2, 0.0, 1.8, nan],
 [nan, 1.8, 0.0, 1.1],
 [2.5, nan, 1.1, 0.0]]
```

### Results:
✅ **perform_clustering**: Successfully handles NaN values
- Warning: Found 4 NaN values in distance matrix. Applying imputation...
- Replaced NaN values with median distance: 1.1500
- Cluster labels: [0 1 1 1]

✅ **perform_clustering_robust**: Multiple strategies work
- Median strategy: ✓ Success
- Mean strategy: ✓ Success  
- Remove strategy: ✓ Success (when applicable)

✅ **perform_umap**: Consistent NaN handling
- Embedding shape: (4, 2)
- No errors or crashes

## Impact
1. **Resolved SpectralClustering NaN Error**: Distance analysis notebook now runs without errors
2. **Robust Pipeline**: Multiple fallback strategies ensure analysis continues even with problematic data
3. **Preserved Data Integrity**: Original matrices are not modified (copy-based approach)
4. **Transparent Processing**: Warning messages inform users about NaN handling
5. **Consistent Behavior**: Both clustering and UMAP functions handle NaN values uniformly

## Files Modified
- `/Users/berksakalli/Projects/BranchArchitect/notebooks/distance_plot_utils.py`
  - Enhanced `perform_clustering` function
  - Added `perform_clustering_robust` function  
  - Enhanced `perform_umap` function

## Backward Compatibility
✅ All existing code continues to work unchanged
✅ Default behavior handles NaN values automatically
✅ Advanced users can choose specific NaN handling strategies

## Verification
The fix has been tested with:
- Various NaN patterns in distance matrices
- Different matrix sizes and cluster counts
- Edge cases (all zeros, high NaN density)
- Integration with the actual distance analysis pipeline

**Status: ✅ RESOLVED** - The SpectralClustering NaN error has been completely fixed and the distance analysis pipeline is now robust to missing values.
