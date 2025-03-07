from typing import List, Set, Dict
from brancharchitect.split import PartitionSet
import itertools
from brancharchitect.jumping_taxa.debug import jt_logger, format_set


def generalized_meet_product(matrix):
    """Compute generalized meet product as defined in documentation."""
    n = len(matrix)
    if n == 0:
        return set()

    m = len(matrix[0])
    # Check that all rows have length m
    if any(len(row) != m for row in matrix):
        raise ValueError("All rows must have the same number of columns.")

    if (n == 1) and (m == 2):
        return matrix[0][0].intersection(matrix[0][1])

    # If matrix is square, use the square_meet_product
    if n == m:
        return square_meet_product(matrix)

    # Rectangular case:
    result = set()
    r = min(n, m)

    # If there are more columns than rows (n < m): choose combinations of r columns
    if n < m:
        # For each combination of r column indices (sorted order)
        for cols in itertools.combinations(range(m), r):
            # Build the square submatrix using all rows and only the chosen columns
            submatrix = [[row[c] for c in cols] for row in matrix]
            term = square_meet_product(submatrix)
            result = result.symmetric_difference(term)
    else:  # More rows than columns (m < n): choose combinations of r rows
        for rows in itertools.combinations(range(n), r):
            submatrix = [matrix[r_idx] for r_idx in rows]
            term = square_meet_product(submatrix)
            result = result.symmetric_difference(term)
    return result


def square_meet_product(matrix: List[List[PartitionSet]]) -> PartitionSet:
    """
    Calculate the determinant of a matrix of sets.

    For a 2×2 matrix, the formula is simply:
    det(A) = (A[0][0] ∩ A[1][1]) △ (A[0][1] ∩ A[1][0])

    Args:
        matrix: A square matrix where each element is a set

    Returns:
        A set representing the determinant
    """
    n = len(matrix)

    # Special case: 1×1 matrix
    if n == 1:
        return matrix[0][0].intersection(matrix[0][1])

    # Direct calculation for 2×2 matrix
    if n == 2:
        main_diagonal = matrix[0][0] & (matrix[1][1])
        counter_diagonal = matrix[0][1] & matrix[1][0]
        jt_logger.logger.info(f"{main_diagonal}" + "  △  " + f"{counter_diagonal}")
        return main_diagonal.symmetric_difference(counter_diagonal)

    # For larger matrices, use the standard permutation method
    # but without tracking signs
    result = set()
    for perm in itertools.permutations(range(n)):
        # Calculate intersection along this permutation
        term = None
        for i in range(n):
            entry = matrix[i][perm[i]]
            if term is None:
                term = entry.copy()
            else:
                term = term.intersection(entry)

        # XOR with the result
        result = result.symmetric_difference(term)

    return result


def create_matrix(direction_by_intersection: List[Dict]) -> List[List[Set]]:
    """
    Create a single matrix from direction analysis results.

    Args:
        direction_by_intersection: List of dictionaries with keys:
            - "A": frozenset of elements from first set
            - "B": frozenset of elements from second set
            - "direction_a": Tuple indicating direction of first set
            - "direction_b": Tuple indicating direction of second set

    Returns:
        List[List[Set]]: A list containing a single matrix where each row is [A, B]
    """
    if not direction_by_intersection:
        return []

    # Create a single matrix with all rows
    matrix = []
    for row in direction_by_intersection:
        a_key = frozenset(row["A"])
        b_val = frozenset(row["B"])
        matrix.append([a_key, b_val])
    return matrix  # Return as a list of matrices for compatibility with existing code


def split_matrix(matrix):
    """
    Split a matrix into smaller matrices based on unique values in the left column.

    If there are only two rows with different left values, don't split.

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

    # Group rows by left column value (using frozenset for hashable keys)
    groups = {}
    for row in matrix:
        if not row or len(row) < 2:
            continue

        # Use the left column value (row[0]) as the key
        key = frozenset(row[0])
        if key not in groups:
            groups[key] = []
        groups[key].append(row)

    # If only one group, return the original matrix
    if len(groups) <= 1:
        return [matrix]

    # Create separate matrices for each group
    result_matrices = []
    for left_value, rows in groups.items():
        # Create a new matrix with these rows
        new_matrix = [row[:] for row in rows]
        result_matrices.append(new_matrix)

    # Log the split operation
    jt_logger.section("Matrix Splitting")
    jt_logger.info(f"Split matrix into {len(result_matrices)} separate matrices")

    for i, split in enumerate(result_matrices):
        jt_logger.info(f"Matrix {i+1} (based on left value):")
        jt_logger.matrix(split)
    return result_matrices


def matrix_has_singleton_entries(matrix):
    """
    Check if a matrix has singleton entries.

    Args:
        matrix: A matrix where each element is a set

    Returns:
        bool: True if any element is a singleton, False otherwise
    """
    return any(len(entry) == 1 for row in matrix for entry in row)


def generalized_off_diagonal_meet_product(matrix):
    """
    Compute generalized meet product using off-diagonal meets instead of square meet product.

    For 2x2 matrices, this is simply the off-diagonal meet.
    For larger matrices, we compute off-diagonal meets of all 2x2 submatrices and combine them.

    Args:
        matrix: A matrix where each element is a set

    Returns:
        set: The result of combining all off-diagonal meets
    """
    n = len(matrix)
    if n == 0:
        return set()

    # Check for empty rows
    if any(len(row) == 0 for row in matrix):
        return set()

    m = len(matrix[0])
    # Check that all rows have length m
    if any(len(row) != m for row in matrix):
        raise ValueError("All rows must have the same number of columns.")

    # Special case: 1×2 matrix
    if (n == 1) and (m == 2):
        return matrix[0][0].intersection(matrix[0][1])

    # Direct calculation for 2×2 matrix
    if n == 2 and m == 2:
        jt_logger.section("Off-Diagonal Meet for 2x2 Matrix")
        off_diag = matrix[0][1].intersection(matrix[1][0])
        jt_logger.info(f"Off-diagonal meet: {format_set(off_diag)}")
        return off_diag

    # For larger matrices, select all possible 2×2 submatrices and compute their off-diagonal meets
    jt_logger.section("Generalized Off-Diagonal Meet Computation")
    jt_logger.info(f"Processing {n}x{m} matrix")

    result = set()

    # Select all pairs of rows and columns
    for row_indices in itertools.combinations(range(n), 2):
        for col_indices in itertools.combinations(range(m), 2):
            # Extract the 2x2 submatrix
            submatrix = [
                [
                    matrix[row_indices[0]][col_indices[0]],
                    matrix[row_indices[0]][col_indices[1]],
                ],
                [
                    matrix[row_indices[1]][col_indices[0]],
                    matrix[row_indices[1]][col_indices[1]],
                ],
            ]

            # Compute off-diagonal meet
            off_diag = submatrix[0][1].intersection(submatrix[1][0])

            # Add to result using symmetric difference
            if off_diag:
                result = result.symmetric_difference(off_diag)
                jt_logger.info(f"Submatrix off-diagonal meet: {format_set(off_diag)}")

    jt_logger.info(f"Final result: {format_set(result)}")
    return result


def should_canonicalize_lattice_matrix(matrix):
    """
    Checks if a 2x2 matrix of sets needs to be canonicalized according to the rule.

    Args:
        matrix: A 2x2 matrix where each element is a set

    Returns:
        bool: True if the matrix needs to be canonicalized, False otherwise
    """
    # Check matrix dimensions
    if len(matrix) != 2 or len(matrix[0]) != 2:
        return False
    # Compute unsplit elements (elements common to both cells in a row)
    unsplit_elements = [row[0] & row[1] for row in matrix]
    # Check if each row has exactly one unsplit element
    if all(len(ue) == 1 for ue in unsplit_elements):
        top_unsplit = next(iter(unsplit_elements[0]))
        bottom_unsplit = next(iter(unsplit_elements[1]))
        # If unsplit elements differ, the matrix needs canonicalization
        return top_unsplit != bottom_unsplit
    # No canonicalization needed if not all rows have single unsplit elements
    return False


def canonicalize_lattice_matrix(matrix):
    """
    Transforms a 2x2 matrix of sets according to the rule.

    Args:
        matrix: A 2x2 matrix where each element is a set

    Returns:
        bool: True if the matrix was transformed, False otherwise
    """
    # Log initial matrix state with proper table formatting
    jt_logger.section("Matrix Transform")

    if len(matrix) != 2 or len(matrix[0]) != 2:
        return False

    table_data = []
    for i, row in enumerate(matrix):
        formatted_row = [format_set(cell) for cell in row]
        table_data.append(formatted_row)

    # Compute unsplit elements
    unsplit_elements = [row[0] & row[1] for row in matrix]

    if all(len(ue) == 1 for ue in unsplit_elements):
        top_unsplit = next(iter(unsplit_elements[0]))
        bottom_unsplit = next(iter(unsplit_elements[1]))

        if top_unsplit != bottom_unsplit:
            # Swap and log transformation
            matrix[0][1], matrix[1][1] = matrix[1][1], matrix[0][1]
            jt_logger.info("Matrix was transformed: Column 2 cells were swapped")
            jt_logger.matrix(matrix)
            return True
        else:
            jt_logger.info("No transformation needed: Unsplit elements are identical")
            return False
    else:
        jt_logger.info(
            "No transformation needed: Not all rows have single unsplit elements"
        )

    return matrix


def canonicalize_diagonal_swap_2(matrix):
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
    return new_matrix  # , swapped_any


# ---------------------------
# Lattice and Dependent Solution Functions
# ---------------------------


def meet(sets):
    """
    Compute the meet (intersection) of a collection of sets.
    In a Boolean lattice, the meet is just the intersection.
    If sets is empty, returns the empty set.
    """
    it = iter(sets)
    try:
        result = next(it).copy()
    except StopIteration:
        return set()  # No sets provided.
    for s in it:
        result &= s
    return result


def compute_row_intersections(matrix):
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


def infer_mapping(matrix1, matrix2):
    """
    Infer a dependency mapping from the two matrices.
    We assume both matrices have the same number of rows, n.
    We infer the mapping as: pair row i of matrix1 with row (n-1-i) of matrix2.
    For example, if n = 2, mapping = [(0,1), (1,0)].
    """
    n1 = len(matrix1)
    n2 = len(matrix2)
    if n1 != n2:
        raise ValueError(
            "Matrices must have the same number of rows to infer mapping automatically."
        )
    n = n1
    mapping = [(i, n - 1 - i) for i in range(n)]
    return mapping


def dependent_solutions(matrix1, matrix2, mapping):
    """
    Given two matrices (each a list of rows, where each row is an iterable of blocks)
    and a dependency mapping (a list of tuples (i,j)), form the dependent solutions.

    For each (i,j) in mapping, form the solution as the Cartesian product of the elements
    in the intersection of row i of matrix1 and row j of matrix2.

    Returns a list of solution tuples.
    """
    inters1 = compute_row_intersections(matrix1)
    inters2 = compute_row_intersections(matrix2)

    solutions = []
    for i, j in mapping:
        for a in inters1[i]:
            for b in inters2[j]:
                solutions.append((a, b))
    return solutions

def dependent_unique_solutions(matrix1, matrix2):
    """
    Returns only those dependent solutions for which the intersections in the mapped rows
    are singletons. For each mapping (i, j), if the intersection set from matrix1's row i
    and matrix2's row j are both singletons, that unique pair is returned.
    Returns a list of unique solution tuples.
    """
    inters1 = compute_row_intersections(matrix1)
    inters2 = compute_row_intersections(matrix2)
    mapping = infer_mapping(matrix1, matrix2)
    unique_solutions = []
    for i, j in mapping:
        if len(inters1[i]) == 1 and len(inters2[j]) == 1:
            a = next(iter(inters1[i]))
            b = next(iter(inters2[j]))
            unique_solutions.append((a, b))
    return unique_solutions

def solve_matrix_puzzle(matrix1, matrix2):
    """
    Compute the final dependent solution(s) as follows:
      - First, call dependent_unique_solutions(matrix1, matrix2).
        If one or more unique solutions (i.e. singleton intersections) exist, return them.
      - Otherwise, call dependent_solutions(matrix1, matrix2) to get all candidate solution pairs,
        then compute the union (set) of all first coordinates and the union (set) of all second coordinates.
        Return the pair of unions as a single solution.
      - If several unique solutions exist, return them all.
    Returns a list of solution tuples.
    """
    unique = dependent_unique_solutions(matrix1, matrix2)
    if unique:
        return unique
    else:
        all_sols = solve_matrix_puzzle(matrix1, matrix2)
        first_union = set(sol[0] for sol in all_sols)
        second_union = set(sol[1] for sol in all_sols)
        return [(first_union, second_union)]