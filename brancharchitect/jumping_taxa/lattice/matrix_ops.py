from brancharchitect.partition_set import PartitionSet, Partition
from brancharchitect.jumping_taxa.debug import jt_logger
from brancharchitect.jumping_taxa.lattice.types import PMatrix
from typing import Optional
from pydantic import validate_call


def vector_meet_product(matrix: PMatrix) -> list[PartitionSet]:
    """
    Compute the meet product for a 1×2 matrix (a row vector with two PartitionSets).
    
    Args:
        matrix: PMatrix expected to be [[A, B]]
        
    Returns:
        A list containing the intersection (meet) of A and B, if non-empty.
    """
    if not matrix or len(matrix) != 1 or len(matrix[0]) != 2:
        raise ValueError("Expected a 1x2 matrix (single row with two PartitionSets).")

    row = matrix[0]
    jt_logger.section("Vector Meet Product")
    jt_logger.info(f"Computing meet for row: {row}")
    
    a: PartitionSet = row[0]
    b: PartitionSet = row[1]
    
    result: PartitionSet = a & b  # meet
    jt_logger.info(f"Meet result: {result}")
    
    return [result] if result else []

def filter_out_singleton_rows(matrix: PMatrix) -> PMatrix:
    """
    Return a matrix with rows removed where the meet (intersection) is a singleton.

    Args:
        matrix (PMatrix): The input matrix to filter.

    Returns:
        PMatrix: A new matrix with only rows whose intersection is not a singleton.
    """
    filtered_matrix: PMatrix = []
    
    for idx, row in enumerate(matrix):
        jt_logger.info(f"Processing row {idx}: {row}")
        # Compute meet (intersection of all PartitionSets in the row)
        meet_result = row[0].copy()
        for cell in row[1:]:
            meet_result &= cell

        jt_logger.info(f"Intersection (meet) result: {meet_result}")

        # Keep the row only if the intersection is not a singleton
        if len(meet_result) != 1:
            filtered_matrix.append(row)
        else:
            jt_logger.info(f"Row {idx} removed (singleton meet).")
    
    return filtered_matrix

def generalized_meet_product(matrix: PMatrix) -> list[PartitionSet]:
    """
    Compute meet product based on matrix size and structure.
    """
    jt_logger.section("Generalized Meet Product")    
    # Validate matrix
    if not matrix or not matrix[0]:
        return []
        
    rows, cols = len(matrix), len(matrix[0])
    
    # Ensure consistent dimensions
    if any(len(row) != cols for row in matrix):
        raise ValueError("All rows must have the same number of columns")
    
    # Dispatch to appropriate handler based on dimensions
    if rows == 1 and cols == 2:
        # Single row, two columns (vector case)
        return vector_meet_product(matrix)
    elif rows == cols:
        # Square matrix case
        return square_meet_product(matrix)
    else:
        # General case not implemented
        raise ValueError(f"Generalized meet product not implemented for {rows}×{cols} matrices")


def square_meet_product(matrix: PMatrix) -> list[PartitionSet]:
    jt_logger.section("Square Meet Product")
    jt_logger.matrix_pretty(matrix)
    n :int = len(matrix)
    solutions: list[PartitionSet] = []

    # Special case: 1×1 matrix
    if n == 1:
        # Use matrix elements directly since they're already PartitionSets
        result: PartitionSet = matrix[0][0] & matrix[0][1]
        solutions.append(result)
        return solutions

    # Direct calculation for 2×2 matrix
    if n == 2:
        # Use matrix elements directly since they're already PartitionSets
        main_diagonal: PartitionSet = matrix[0][0] & matrix[1][1]
        counter_diagonal: PartitionSet = matrix[0][1] & matrix[1][0]

        if main_diagonal:
            solutions.append(main_diagonal)
        if counter_diagonal:
            solutions.append(counter_diagonal)
        return solutions

    raise ValueError("Square meet product not implemented for matrices larger than 2x2.")


def create_matrix(independent_directions: list[dict[str, PartitionSet]]) -> Optional[PMatrix]:
    """
    Create a single matrix from direction analysis results.

    Args:
        direction_by_intersection: list of dictionaries with keys:s
            - "A": frozenset of elements from first set
            - "B": frozenset of elements from second set
            - "direction_a": Tuple indicating direction of first set
            - "direction_b": Tuple indicating direction of second set

    Returns:
        list[list[Set]]: A list containing a single matrix where each row is [A, B]
    """
    if not independent_directions:
        return []

    # Create a single matrix with all rows
    matrix : list = []
    for row in independent_directions:
        a_key : PartitionSet = row["A"]
        b_val : PartitionSet = row["B"]
        matrix.append([a_key, b_val])
    jt_logger.section("Matrix Construction")
    jt_logger.matrix(matrix)
    return matrix  # Return as a list of matrices for compatibility with existing code


def split_matrix(matrix: PMatrix) -> list[PMatrix]:
    """
    Split a matrix into smaller matrices based on unique values in the left column.

    If there are only two rows with different left column values, don't split.

    Args:
        matrix: A list of lists representing the matrix to split

    Returns:
        list: A list of matrices split by unique left column values
    """
    
    # Special case: if it's a 2x2 matrix with different left column values, don't split
    if (
        len(matrix) == 2
        and len(matrix[0]) == 2
        and frozenset(matrix[0][0]) != frozenset(matrix[1][0])
    ):
        jt_logger.info("2x2 matrix with independent rows detected - keeping as is")
        return [matrix]

    # Specify the type annotation for groups:
    groups: dict[frozenset, list[list[PartitionSet]]] = {}

    for row in matrix:
        if not row or len(row) < 2:
            continue

        key = row[0]  # The left column value
        hashable_key = frozenset(key)
        if hashable_key not in groups:
            groups[hashable_key] = []
        groups[hashable_key].append(row)

    if len(groups) <= 1:
        return [matrix]

    result_matrices = []
    for _, rows in groups.items():
        new_matrix = [r[:] for r in rows]
        result_matrices.append(new_matrix)

    jt_logger.section("Matrix Splitting")
    jt_logger.info(f"Split matrix into {len(result_matrices)} separate matrices")

    for i, split in enumerate(result_matrices):
        jt_logger.info(f"Matrix {i+1} (based on left value):")
        jt_logger.matrix(split)

    return result_matrices

# ---------------------------
# Lattice and Dependent Solution Functions
# ---------------------------
@validate_call
def meet(sets: list[PartitionSet]) -> PartitionSet:
    """
    Compute the meet (intersection) of a collection of PartitionSets.
    In a Boolean lattice, the meet is just the intersection.
    If sets is empty, returns an empty PartitionSet.
    """
    it = iter(sets)
    try:
        first = next(it)
    except StopIteration:
        return PartitionSet()  # Return an empty PartitionSet.
    # Create a new PartitionSet based on the first element,
    # preserving the lookup and name for consistency.
    result = first.copy()
    for s in it:
        result = result & s  # Uses PartitionSet.__and__ to intersect
    return result


@validate_call
def compute_row_intersections(matrix: PMatrix) -> list[PartitionSet]:
    """
    Given a matrix represented as a list of rows (each row is an iterable of blocks,
    and each block is a set), compute the intersection (the meet) for each row.

    Returns a list where each entry is the intersection set for that row.
    """
    intersections = []
    for idx, row in enumerate(matrix):
        inter = meet(row)
        intersections.append(inter)
    return intersections

@validate_call
def infer_mapping(matrix1 : PMatrix, matrix2 : PMatrix)->list[tuple[int, int]]: 
    """
    Infer a dependency mapping from the two matrices.
    We assume both matrices have the same number of rows, n.
    We infer the mapping as: pair row i of matrix1 with row (n-1-i) of matrix2.
    For example, if n = 2, mapping = [(0,1), (1,0)].
    """
    n1 : int = len(matrix1)
    n2 : int = len(matrix2)
    if n1 != n2:
        raise ValueError(
            "Matrices must have the same number of rows to infer mapping automatically."
        )
        
    n : int = n1
    mapping : list[tuple[int,int]] = [(i, n - 1 - i) for i in range(n)]
    return mapping

@validate_call
def dependent_solutions(matrix1:  PMatrix, matrix2: PMatrix, mapping: list[tuple[int, int]])->list[PartitionSet]:
    """
    Given two matrices (each a list of rows, where each row is an iterable of blocks)
    and a dependency mapping (a list of tuples (i,j)), form the dependent solutions.

    For each (i,j) in mapping, form the solution as the Cartesian product of the elements
    in the intersection of row i of matrix1 and row j of matrix2.

    Returns a list of solution tuples.
    """
    inters1 : list[PartitionSet] = compute_row_intersections(matrix1)
    inters2 : list[PartitionSet] = compute_row_intersections(matrix2)

    solutions : list[PartitionSet] = []
    for i, j in mapping:
        for a in inters1[i]:
            for b in inters2[j]:
                solutions.append(PartitionSet({a, b},name="dependent", look_up=a.lookup))
    return solutions

@validate_call
def dependent_unique_solutions(matrix1: PMatrix, matrix2: PMatrix) -> list[PartitionSet]:
    """
    Returns only those dependent solutions for which the intersections in the mapped rows
    are singletons. For each mapping (i, j), if the intersection set from matrix1's row i
    and matrix2's row j are both singletons, that unique pair is returned.
    Returns a list of unique solution tuples.
    """
    
    jt_logger.section("Dependent Unique Solutions")
    jt_logger.section("Matrix 1")
    jt_logger.matrix(matrix1)
    jt_logger.section("Matrix 2")
    jt_logger.matrix(matrix2)    
    
    inters1 = compute_row_intersections(matrix1)
    
    jt_logger.section("Intersections for Matrix 1")
    jt_logger.info("Intersections for Matrix 1")    
    jt_logger.info(str(inters1))
    
    inters2 = compute_row_intersections(matrix2)
    
    jt_logger.section("Intersections for Matrix 2")
    jt_logger.info("Intersections for Matrix 2")    
    jt_logger.info(str(inters2))
    
    mapping: list[tuple[int, int]] = infer_mapping(matrix1, matrix2)
    unique_solutions : list[PartitionSet] = []
    for i, j in mapping:
        if len(inters1[i]) == 1 and len(inters2[j]) == 1:            
            a : Partition = next(iter(inters1[i]))
            b : Partition = next(iter(inters2[j]))
            solution : PartitionSet = PartitionSet({a, b}, name="dependent", look_up=a.lookup)
            unique_solutions.append(solution)
    return unique_solutions

@validate_call
def solve_matrix_puzzle(matrix1: list[list[PartitionSet]], matrix2:list[list[PartitionSet]])->list[PartitionSet]:
    unique = dependent_unique_solutions(matrix1, matrix2)
    if unique:
        return unique
    return []


def canonicalize_diagonal_swap(matrix):
    """
    Given a 2-row matrix (each row is a list of cells, each cell a set),
    perform a diagonal swap as follows:

      - For each column j:
          If the bottom cell is a singleton (len(cell)==1) and the top cell is not,
          and if either the left or right adjacent column (if any) has a singleton in the top row,
          then swap the two cells in column j so that the singleton moves to the top row.

      - Only swap one column per such occurrence.

    Returns a tuple (new_matrix, swapped_any) where:
      new_matrix: the canonicalized 2-row matrix.
      swapped_any: True if any column was swapped, else False.
    """
    num_cols = len(matrix[0])
    # Make a copy of the original matrix.
    new_matrix = [list(matrix[0]), list(matrix[1])]
    swapped_any = False

    for j in range(num_cols):
        top = new_matrix[0][j]
        bottom = new_matrix[1][j]
        top_is_singleton = len(top) == 1
        bottom_is_singleton = len(bottom) == 1

        # We consider swapping only if the bottom cell is singleton and top is not.
        if bottom_is_singleton and not top_is_singleton:
            # Check for adjacent column(s) that already have a top-row singleton.
            left_top = j > 0 and len(new_matrix[0][j - 1]) == 1
            right_top = j < num_cols - 1 and len(new_matrix[0][j + 1]) == 1
            if left_top or right_top:
                # Swap the column so that the singleton moves to the top row.
                new_matrix[0][j], new_matrix[1][j] = bottom, top
                swapped_any = True
    return new_matrix, swapped_any