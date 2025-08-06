from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.elements.partition import Partition
from brancharchitect.jumping_taxa.debug import jt_logger
from brancharchitect.jumping_taxa.lattice.types import PMatrix
from typing import List

# Use FrozenPartitionSet for hashable, immutable keys
from brancharchitect.elements.frozen_partition_set import FrozenPartitionSet
from functools import reduce

# Ensure Partition is imported for type hints


def _validate_matrix(matrix: PMatrix) -> tuple[int, int]:
    if not matrix or not matrix[0]:
        raise ValueError("Matrix must not be empty and must have at least one column.")
    rows, cols = len(matrix), len(matrix[0])
    if any(len(row) != cols for row in matrix):
        raise ValueError("All rows must have the same number of columns.")
    return rows, cols


def _vector_meet_product(matrix: PMatrix) -> list[PartitionSet[Partition]]:
    jt_logger.section("Vector Meet Product")
    rows, cols = _validate_matrix(matrix)
    if rows != 1 or cols != 2:
        raise ValueError("Expected a 1x2 matrix for vector meet product.")
    a, b = matrix[0]
    result = a & b
    jt_logger.info(f"Meet result: {result}")
    return [result] if result else []


def generalized_meet_product(matrix: PMatrix) -> list[PartitionSet[Partition]]:
    jt_logger.section("Generalized Meet Product")
    rows, cols = _validate_matrix(matrix)
    if rows == 1 and cols == 2:
        return _vector_meet_product(matrix)
    elif rows == cols:
        return _square_meet_product(matrix)
    else:
        raise ValueError(
            f"Generalized meet product not implemented for {rows}x{cols} matrices."
        )


def _square_meet_product(matrix: PMatrix) -> list[PartitionSet[Partition]]:
    jt_logger.section("Square Meet Product")
    jt_logger.matrix_pretty(matrix)
    rows, cols = _validate_matrix(matrix)
    if rows != cols:
        raise ValueError("Matrix must be square for square meet product.")
    if rows == 1:
        result = matrix[0][0]
        return [result] if result else []
    if rows == 2:
        main_diag = matrix[0][0] & matrix[1][1]
        counter_diag = matrix[0][1] & matrix[1][0]
        results: list[PartitionSet[Partition]] = []
        if main_diag:
            results.append(main_diag)
        if counter_diag:
            results.append(counter_diag)
        return results
    raise ValueError(
        "Square meet product not implemented for matrices larger than 2x2."
    )


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

    groups: dict[
        FrozenPartitionSet[Partition], list[list[PartitionSet[Partition]]]
    ] = {}

    for row in matrix:
        if not row or len(row) < 2:
            continue

        key: PartitionSet[Partition] = row[0]  # The left column value
        # Convert to FrozenPartitionSet for grouping
        frozen_key = FrozenPartitionSet(
            set(key), encoding=getattr(key, "encoding", None)
        )
        if frozen_key not in groups:
            groups[frozen_key] = []
        groups[frozen_key].append(row)

    if len(groups) <= 1:
        return [matrix]

    result_matrices: List[PMatrix] = []
    for _, rows in groups.items():
        new_matrix: PMatrix = [r[:] for r in rows]
        result_matrices.append(new_matrix)

    jt_logger.section("Matrix Splitting")
    jt_logger.info(f"Split matrix into {len(result_matrices)} separate matrices")

    for i, split in enumerate(result_matrices):
        jt_logger.info(f"Matrix {i + 1} (based on left value):")
        jt_logger.matrix(matrix=split)
    return result_matrices


# ---------------------------
# Lattice and Dependent Solution Functions
# ---------------------------


def meet(sets: list[PartitionSet[Partition]]) -> PartitionSet[Partition]:
    """
    Compute the meet (intersection) of a collection of PartitionSets.
    In a Boolean lattice, the meet is just the intersection.
    If sets is empty, returns an empty PartitionSet.
    """
    if not sets:
        return PartitionSet()
    return reduce(lambda a, b: a & b, sets)


def compute_row_intersections(matrix: PMatrix) -> list[PartitionSet[Partition]]:
    """
    Given a matrix represented as a list of rows (each row is an iterable of blocks,
    and each block is a set), compute the intersection (the meet) for each row.

    Returns a list where each entry is the intersection set for that row.
    """
    return [meet(row) for row in matrix]


def infer_mapping(matrix1: PMatrix, matrix2: PMatrix) -> list[tuple[int, int]]:
    """
    Infer a dependency mapping from the two matrices.
    If matrices have the same number of rows, use the standard reverse mapping.
    If they differ, create a mapping that pairs as many rows as possible.
    """
    n1: int = len(matrix1)
    n2: int = len(matrix2)
    
    if n1 == n2:
        # Standard case: same number of rows
        n: int = n1
        mapping: list[tuple[int, int]] = [(i, n - 1 - i) for i in range(n)]
        return mapping
    else:
        # Fallback case: different number of rows
        # Map as many as possible, using the smaller matrix size
        min_n = min(n1, n2)
        mapping: list[tuple[int, int]] = []
        
        for i in range(min_n):
            # For matrix1, use index i
            # For matrix2, use reverse index but within its bounds
            j = min(n2 - 1 - i, n2 - 1)
            j = max(0, j)  # Ensure non-negative
            mapping.append((i, j))
        
        return mapping


def dependent_solutions(
    matrix1: PMatrix, matrix2: PMatrix, mapping: list[tuple[int, int]]
) -> list[PartitionSet[Partition]]:
    """
    Given two matrices (each a list of rows, where each row is an iterable of blocks)
    and a dependency mapping (a list of tuples (i,j)), form the dependent solutions.

    For each (i,j) in mapping, form the solution as the Cartesian product of the elements
    in the intersection of row i of matrix1 and row j of matrix2.

    Returns a list of solution tuples.
    """
    inters1: list[PartitionSet[Partition]] = compute_row_intersections(matrix1)
    inters2: list[PartitionSet[Partition]] = compute_row_intersections(matrix2)

    solutions: list[PartitionSet[Partition]] = []
    for i, j in mapping:
        for a in inters1[i]:
            for b in inters2[j]:
                solutions.append(
                    PartitionSet({a, b}, encoding=a.encoding, name="dependent")
                )
    return solutions


def dependent_unique_solutions(
    matrix1: PMatrix, matrix2: PMatrix
) -> list[PartitionSet[Partition]]:
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
    unique_solutions: list[PartitionSet[Partition]] = []
    for i, j in mapping:
        if len(inters1[i]) == 1 and len(inters2[j]) == 1:
            a: Partition = next(iter(inters1[i]))
            b: Partition = next(iter(inters2[j]))
            solution: PartitionSet[Partition] = PartitionSet(
                {a, b}, name="dependent", encoding=a.encoding
            )
            unique_solutions.append(solution)
    return unique_solutions


def solve_matrix_puzzle(
    matrix1: list[list[PartitionSet[Partition]]],
    matrix2: list[list[PartitionSet[Partition]]],
) -> list[PartitionSet[Partition]]:
    unique: list[PartitionSet[Partition]] = dependent_unique_solutions(matrix1, matrix2)
    if unique:
        return unique
    return []
