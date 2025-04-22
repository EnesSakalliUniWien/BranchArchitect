# BranchArchitect

Brancharchitect implements algorithms to deal with tree trajectory.
Its main feature is an algorithm to identify a "jumping taxon", that is given two trees that differ by exactly one taxon, that taxon can efficiently be identified.
Additionally brancharchitect can read newick files, write json files, calculate the consensus tree and visualise trees as svgs.

# Examples

## Parse Newick into Python Tree

```{python}
from brancharchitect.newick_parser import parse_newick

with open('newick_file.nwk') as f:
    tree = parse_newick(f.read())
```

## Serialize Tree into JSON


```{python}
from brancharchitect._io import read_newick, write_json

tree = read_newick('newick_file.nwk')
write_json(tree, 'json_file.json')
```

## Call jumping Taxa on a pair of trees

To find jumping taxa, both trees must have the same taxa and the taxa order must be identical.
To ensure that the taxa order is identical, either read both trees from the same file, provide an explicit taxa order or give the taxa order of the first tree when parsing the second tree.
The first example shows parsing both trees from the same file:

```{python}
from brancharchitect.io import read_newick
from brancharchitect.jumping_taxa import call_jumping_taxa

tree1, tree2 = read_newick('newick_file.nwk')

jumping_taxa = call_jumping_taxa(tree1, tree2)
```

Alternatively provide an explicit order:

```{python}
order = ['A', 'C', 'B', 'E']

tree1 = read_newick('newick_file1.nwk', order)
tree2 = read_newick('newick_file2.nwk', order)

jumping_taxa = call_jumping_taxa(tree1, tree2)
```

Or give the the order of the first tree when parsing the other tree:

```{python}
tree1 = read_newick('newick_file1.nwk')
tree2 = read_newick('newick_file2.nwk', tree1._order)

jumping_taxa = call_jumping_taxa(tree1, tree2)
```

## Generate SVG


```{python}

from brancharchitect.io import read_newick, write_svg

tree = read_newick('newick_file.nwk')
write_svg(tree, 'image.svg')
```

# Lattice Solver Algorithm Documentation

The Lattice Solver is a core component of BranchArchitect's jumping taxa identification system. It implements an algorithm that analyzes phylogenetic trees to find taxa that have "jumped" between positions in two different tree topologies.

## Core Concept: Left-Right Comparison

The fundamental principle of the lattice algorithm is the systematic comparison between "left" and "right" elements, which represent corresponding components from the two input trees. This left-right comparison drives the entire algorithm:

1. Every split (bipartition) that exists in both trees is analyzed by comparing its context in the left tree versus the right tree
2. For each common split, we extract and compare their covers (sets that contain the split) from both trees
3. The algorithm identifies differences between these left and right covers that indicate potential jumping taxa
4. These differences are captured in matrices where the first column represents elements from the left tree and the second column represents elements from the right tree

This systematic left-right comparison is what enables the algorithm to precisely identify taxa that have "jumped" from one position in the first tree to another position in the second tree.

## Algorithm Overview

The lattice algorithm analyzes two phylogenetic trees with identical leaf sets to identify "jumping taxa" - leaves that have moved between different positions in the tree structure. The algorithm works by:

1. Constructing a lattice structure representing the relationships between splits (bipartitions) in the two trees
2. Finding partitions that are independent between the trees
3. Creating and solving matrices that encode these relationships
4. Identifying minimal solutions representing the jumping taxa

## Key Components

### 1. Lattice Construction

The algorithm begins by constructing sub-lattices from the two input trees:

```python
def lattice_algorithm(input_tree1, input_tree2, leaf_order):
    # Initialize solution manager
    lattice_solution = LatticeSolutions()
    
    # Construct sub-lattices from the two trees
    current_s_edges = construct_sub_lattices(input_tree1, input_tree2)
    
    # Process sub-lattices to find solutions
    if current_s_edges:
        process_iteration(current_s_edges, lattice_solution)
    
    # Collect and return final solutions
    solutions_set = []
    # ... process solutions and identify jumping taxa
    return solutions_set
```

### 2. Sub-lattice Analysis

For each sub-lattice (represented by a `LatticeEdge`), the algorithm:

```python
def process_single_lattice_edge(edge, solutions_manager):
    # Analyze the lattice edge
    intersection_map, left_minus_right_map, right_minus_left_map = pairwise_lattice_analysis(edge)
    
    # Find independent partitions
    independent_sides = gather_independent_partitions(
        intersection_map, left_minus_right_map, right_minus_left_map
    )
    
    # Create a matrix from independent partitions
    candidate_matrix = create_matrix(independent_sides)
    if not candidate_matrix:
        return True  # No solutions
        
    # Split and solve the matrix
    matrices = split_matrix(candidate_matrix)
    
    # Process solutions based on matrix type
    # ... (degenerate or non-degenerate cases)
    
    # Add valid solutions to the solution manager
    # ...
```

### 3. Identifying Independent Partitions

The algorithm identifies independent partitions which represent potential jumping taxa by comparing left and right elements:

```python
def gather_independent_partitions(intersection_map, left_minus_right_map, right_minus_left_map):
    independent_sides = []
    
    for common_partition in intersection_map:
        left_entry = left_minus_right_map[common_partition]
        right_entry = right_minus_left_map[common_partition]
        
        independent_left = left_entry["covet_left"]
        independent_right = right_entry["covet_right"]
        
        # Check independence conditions
        conditions = check_independence_conditions(left_entry, right_entry)
        is_independent = is_independent_any(conditions)
        
        # If independent, add to result
        if is_independent:
            independent_sides.append({
                "A": independent_left,  # From left tree
                "B": independent_right,  # From right tree
            })
    
    return independent_sides
```

The independence check explicitly compares the relationship between left and right elements:

```python
def check_independence_conditions(left, right):
    # Extract partition sets
    left_arm = left.get("covet_left", PartitionSet())  # From left tree
    right_arm = right.get("covet_right", PartitionSet())  # From right tree
    
    # Condition 1: Left partition is not fully contained in right
    # and has a non-empty right-side residual
    left_non_subsumption = check_non_subsumption_with_residual(
        primary_set=left_arm,
        comparison_set=right_arm,
        residual=left.get("b-a", PartitionSet()),  # Right minus left
    )
    
    # Condition 2: Right partition is not fully contained in left
    # and has a non-empty left-side residual
    right_non_subsumption = check_non_subsumption_with_residual(
        primary_set=right_arm,
        comparison_set=left_arm,
        residual=right.get("a-b", PartitionSet()),  # Left minus right
    )
    
    # Additional conditions comparing left and right
    left_atomic_inclusion = check_atomic_inclusion(left_arm, right_arm)
    right_atomic_inclusion = check_atomic_inclusion(right_arm, left_arm)
    
    return (
        left_non_subsumption,
        right_non_subsumption,
        left_atomic_inclusion,
        right_atomic_inclusion,
    )
```

### 4. Matrix Operations

The algorithm creates and solves matrices to find jumping taxa:

```python
def create_matrix(independent_directions):
    """Create a matrix from independent partitions"""
    matrix = []
    for row in independent_directions:
        a_key = row["A"]  # PartitionSet from left tree
        b_val = row["B"]  # PartitionSet from right tree
        matrix.append([a_key, b_val])
    return matrix
    
def split_matrix(matrix):
    """Split a matrix into smaller matrices if needed"""
    # Special case: 2x2 matrix with different left column values
    if len(matrix) == 2 and len(matrix[0]) == 2 and frozenset(matrix[0][0]) != frozenset(matrix[1][0]):
        return [matrix]
    
    # Group rows by left column value
    groups = {}
    for row in matrix:
        key = frozenset(row[0])
        if key not in groups:
            groups[key] = []
        groups[key].append(row)
    
    # If only one group, return original matrix
    if len(groups) <= 1:
        return [matrix]
    
    # Create new matrices from groups
    result_matrices = []
    for _, rows in groups.items():
        new_matrix = [r[:] for r in rows]
        result_matrices.append(new_matrix)
        
    return result_matrices
```

### 5. Meet Product Computation

A critical operation is the computation of meet products (intersections):

```python
def generalized_meet_product(matrix):
    """Compute meet product based on matrix size and structure"""
    # Validate matrix
    if not matrix or not matrix[0]:
        return []
        
    rows, cols = len(matrix), len(matrix[0])
    
    # Dispatch based on dimensions
    if rows == 1 and cols == 2:
        # Vector case: compute intersection of two sets
        return vector_meet_product(matrix)
    elif rows == cols:
        # Square matrix case: compute diagonal meets
        return square_meet_product(matrix)
    else:
        raise ValueError(f"Generalized meet product not implemented for {rows}×{cols} matrices")

def square_meet_product(matrix):
    """Compute meet product for a square matrix"""
    n = len(matrix)
    solutions = []
    
    if n == 1:
        # 1×1 matrix
        result = matrix[0][0] & matrix[0][1]
        solutions.append(result)
    elif n == 2:
        # 2×2 matrix
        main_diagonal = matrix[0][0] & matrix[1][1]
        counter_diagonal = matrix[0][1] & matrix[1][0]
        
        if main_diagonal:
            solutions.append(main_diagonal)
        if counter_diagonal:
            solutions.append(counter_diagonal)
            
    return solutions
```

### 6. Solution Processing

The algorithm processes and filters solutions:

```python
def process_iteration(sub_lattices, lattice_solutions):
    """Process sub-lattices to find solutions"""
    # Initialize stack with all edges
    processing_stack = sub_lattices.copy()
    
    while processing_stack:
        # Get next edge to process
        s_edge = processing_stack.pop()
        s_edge.visits += 1
        
        # Process the edge
        done = process_single_lattice_edge(s_edge, lattice_solutions)
        
        if not done:
            # Get solutions for this edge and visit
            solutions = lattice_solutions.get_solutions_for_edge_visit(s_edge.split, s_edge.visits)
            
            # If solutions found, remove them from covers and reprocess
            s_edge.remove_solutions_from_covers(solutions)
            processing_stack.append(s_edge)
```

## Theoretical Background

The lattice algorithm is based on mathematical lattice theory, where:

1. **Lattice**: A partially ordered set in which every pair of elements has both a supremum (join) and an infimum (meet).
2. **Meet**: The greatest lower bound (intersection) of two elements in the lattice.
3. **Cover**: The smallest elements that contain a given element.

In the context of phylogenetic trees:
- Each node in a tree defines a bipartition (split) of the taxa set
- The collection of all splits forms a lattice structure
- Jumping taxa can be identified by analyzing how these lattices differ between trees

## Algorithm Steps

1. **Construction**: Build sub-lattices from common splits in both trees
2. **Analysis**: For each sub-lattice:
   - Compute intersection and difference maps between partitions
   - Identify independent partitions
   - Create matrices representing relationships
3. **Solution**: 
   - Split matrices if needed
   - Compute meet products to find solutions
   - Store minimal solutions
4. **Result**: 
   - Process solutions to identify the indices of jumping taxa

## Performance Characteristics

- Time complexity is primarily determined by the number of splits in the trees
- For trees that differ by exactly one taxon, the algorithm efficiently identifies the jumping taxon
- The algorithm can handle degenerate cases where multiple solutions exist

## Example Usage

```{python}
from brancharchitect.tree import Node
from brancharchitect.jumping_taxa.lattice.lattice_solver import lattice_algorithm

# Assuming input_tree1 and input_tree2 are Node objects representing phylogenetic trees
# with identical leaf sets but different topologies

# Get the leaf order (important for consistent indexing)
leaf_order = input_tree1.get_leaf_names()

# Run the lattice algorithm
jumping_taxa_indices = lattice_algorithm(input_tree1, input_tree2, leaf_order)

# Convert indices to taxon names if needed
jumping_taxa_names = [leaf_order[idx] for idx in jumping_taxa_indices]

print(f"Jumping taxa: {jumping_taxa_names}")
```

