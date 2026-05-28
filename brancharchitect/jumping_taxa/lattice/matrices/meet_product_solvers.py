from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.elements.partition import Partition
from brancharchitect.jumping_taxa.lattice.matrices.types import PMatrix
from brancharchitect.jumping_taxa.lattice.matrices.matrix_shape_classifier import (
    MatrixClassifier,
    MatrixCategory,
)
from typing import List, Callable, Optional
from itertools import product
import operator

from brancharchitect.logger import jt_logger
from brancharchitect.logger.formatting import format_partition_set

# Type definition for the meet operation
MeetFunction = Callable[
    [PartitionSet[Partition], PartitionSet[Partition]], PartitionSet[Partition]
]


def _vector_meet_product(
    matrix: PMatrix, meet_fn: Optional[MeetFunction] = None
) -> list[PartitionSet[Partition]]:
    """Compute meet product for 1×2 vector matrix."""
    _rows, cols = MatrixClassifier.validate_matrix(matrix)
    if _rows != 1 or cols != 2:
        raise ValueError("Expected a 1x2 matrix for vector meet product.")

    # Include the input matrix in logs for readability
    if not jt_logger.disabled:
        jt_logger.matrix(matrix, title="Vector Meet Product: Input Matrix")

    # Use operator.and_ as default if no custom meet function provided
    op = meet_fn or operator.and_

    a, b = matrix[0]
    result: PartitionSet[Partition] = op(a, b)
    if result:
        if not jt_logger.disabled:
            jt_logger.info(f"[diag] meet/vector raw = {format_partition_set(result)}")
        maxima = result.maximal_elements()
        if not jt_logger.disabled:
            jt_logger.info(
                f"[diag] meet/vector maxima = {format_partition_set(maxima)}"
            )
        return [maxima]
    return []


def _rectangular_row_wise_meet_product(
    matrix: PMatrix, meet_fn: Optional[MeetFunction] = None
) -> list[PartitionSet[Partition]]:
    """
    Handle rectangular matrices (rows x 2) by computing each row's intersection
    and returning ALL non-empty results as separate solutions.
    """
    _rows, cols = MatrixClassifier.validate_matrix(matrix)

    if cols != 2:
        raise ValueError(f"Rectangular row-wise method requires 2 columns, got {cols}")

    # Include the input matrix in logs for readability
    if not jt_logger.disabled:
        jt_logger.matrix(
            matrix, title="Rectangular Row-Wise Meet Product: Input Matrix"
        )

    row_results: list[PartitionSet[Partition]] = []
    # Use operator.and_ as default if no custom meet function provided
    op = meet_fn or operator.and_

    for row in matrix:
        left, right = row[0], row[1]
        result: PartitionSet[Partition] = op(left, right)

        if result:
            if not jt_logger.disabled:
                jt_logger.info(f"[diag] meet/row raw = {format_partition_set(result)}")
            maxima = result.maximal_elements()
            if not jt_logger.disabled:
                jt_logger.info(
                    f"[diag] meet/row maxima = {format_partition_set(maxima)}"
                )
            row_results.append(maxima)

    return row_results


def generalized_meet_product(
    matrix: PMatrix, meet_fn: Optional[MeetFunction] = None
) -> list[PartitionSet[Partition]]:
    """
    Compute generalized meet product using shape-specific strategy.

    This function dispatches to the appropriate solving strategy based on
    matrix shape classification:
    - VECTOR (1×2): Simple intersection
    - SQUARE (n×n): Diagonal intersection
    - RECTANGULAR (k×2): Row-wise intersection

    Args:
        matrix: Input partition matrix
        meet_fn: Optional function to use for 'meet' (intersection) operation.
                 Defaults to standard set intersection (&) if None.
                 Can be used to inject `geometric_intersection`.

    Returns:
        List of partition set solutions

    Raises:
        ValueError: If matrix shape is unsupported
    """
    rows, cols = MatrixClassifier.validate_matrix(matrix)

    # Classify matrix shape and dispatch to appropriate strategy
    category = MatrixClassifier.classify_matrix(matrix)
    # Log strategy selection for readability
    try:
        if not jt_logger.disabled:
            jt_logger.info(
                f"Matrix shape {rows}x{cols} classified as {category.name}. "
                f"Using {category.name.lower().replace('_', ' ')} strategy."
            )
        jt_logger.log_strategy_selection(rows, cols, category.name)
    except Exception:
        pass

    if category == MatrixCategory.VECTOR:
        return _vector_meet_product(matrix, meet_fn)
    elif category == MatrixCategory.SQUARE:
        return _square_meet_product(matrix, meet_fn)
    elif category == MatrixCategory.RECTANGULAR:
        return _rectangular_row_wise_meet_product(matrix, meet_fn)
    else:
        raise ValueError(
            f"Generalized meet product not implemented for {rows}×{cols} matrices."
        )


def _square_meet_product(
    matrix: PMatrix, meet_fn: Optional[MeetFunction] = None
) -> list[PartitionSet[Partition]]:
    """Compute meet product for square matrix via diagonal intersections."""
    rows, cols = MatrixClassifier.validate_matrix(matrix)
    if rows != cols:
        raise ValueError("Matrix must be square for square meet product.")

    if rows == 1:
        # Include the input matrix in logs for readability
        if not jt_logger.disabled:
            jt_logger.matrix(matrix, title="Square Meet Product (1×1): Input Matrix")
        result = matrix[0][0]
        return [result] if result else []

    results: list[PartitionSet[Partition]] = []

    if rows == 2:
        # Use operator.and_ as default if no custom meet function provided
        op = meet_fn or operator.and_

        if not jt_logger.disabled:
            jt_logger.info(
                "Checking main diagonal (top-left & bottom-right) and "
                "counter-diagonal (top-right & bottom-left) for valid intersections."
            )

        # Main diagonal: [0,0] ∩ [1,1]
        a00, a11 = matrix[0][0], matrix[1][1]
        main_diag = op(a00, a11)

        # Counter diagonal: [0,1] ∩ [1,0]
        a01, a10 = matrix[0][1], matrix[1][0]
        counter_diag = op(a01, a10)

        if not jt_logger.disabled:
            jt_logger.section("Square Meet Product Diagonal Results")
            # Include the input matrix in this section for context
            jt_logger.matrix(matrix, title="Input Matrix")

        if main_diag:
            if not jt_logger.disabled:
                jt_logger.info(
                    f"[diag] meet/diag raw = {format_partition_set(main_diag)}"
                )
            maxima = main_diag.maximal_elements()
            if not jt_logger.disabled:
                jt_logger.info(
                    f"[diag] meet/diag maxima = {format_partition_set(maxima)}"
                )
            results.append(maxima)
        elif not jt_logger.disabled:
            jt_logger.info("Main diagonal intersection is empty.")

        if counter_diag:
            if not jt_logger.disabled:
                jt_logger.info(
                    f"[diag] meet/cdiag raw = {format_partition_set(counter_diag)}"
                )
            maxima = counter_diag.maximal_elements()
            if not jt_logger.disabled:
                jt_logger.info(
                    f"[diag] meet/cdiag maxima = {format_partition_set(maxima)}"
                )
            results.append(maxima)
        elif not jt_logger.disabled:
            jt_logger.info("Counter-diagonal intersection is empty.")

        return results

    # Generalized square meet product for n×n (n > 2): evaluate all row-wise
    # combinations of column selections and keep non-empty intersections.
    op = meet_fn or operator.and_
    if not jt_logger.disabled:
        jt_logger.info(
            f"Computing generalized square meet product for {rows}×{cols} matrix."
        )
        jt_logger.matrix(matrix, title="Square Meet Product (n×n): Input Matrix")

    results = []
    seen: set[tuple[int, ...]] = set()

    for col_indices in product(range(cols), repeat=rows):
        result = matrix[0][col_indices[0]]
        for r in range(1, rows):
            result = op(result, matrix[r][col_indices[r]])
            if not result:
                break

        if not result:
            continue

        maxima = result.maximal_elements()
        if not maxima:
            continue

        key = tuple(sorted(p.bitmask for p in maxima))
        if key in seen:
            continue
        seen.add(key)
        results.append(maxima)

    return results


def _connected_row_components(matrix: PMatrix) -> list[PMatrix]:
    """
    Split rows into connected components by shared frontier atoms in any cell.

    Rows are independent only when there is no path between them through shared
    candidate partitions. Whole cover-set equality is too weak: different cover
    sets can still be coupled when they contain the same frontier partition.
    """
    if not matrix:
        return []

    key_to_row_indices: dict[int, list[int]] = {}
    for row_index, row in enumerate(matrix):
        for cell in row:
            for partition in cell:
                key_to_row_indices.setdefault(partition.bitmask, []).append(row_index)

    adjacency: list[set[int]] = [set() for _ in matrix]
    for row_indices in key_to_row_indices.values():
        for row_index in row_indices:
            adjacency[row_index].update(row_indices)

    components: list[PMatrix] = []
    visited: set[int] = set()

    for start_index in range(len(matrix)):
        if start_index in visited:
            continue

        stack = [start_index]
        component_indices: list[int] = []
        visited.add(start_index)

        while stack:
            row_index = stack.pop()
            component_indices.append(row_index)

            for next_index in sorted(adjacency[row_index], reverse=True):
                if next_index in visited:
                    continue
                visited.add(next_index)
                stack.append(next_index)

        component_indices.sort()
        components.append([matrix[index][:] for index in component_indices])

    return components


def split_matrix(matrix: PMatrix) -> list[PMatrix]:
    """
    Split a matrix into the MINIMAL number of smaller matrices.

    SPLITTING STRATEGY:
        Treat matrix rows as a bipartite dependency graph and split only
        disconnected components. Rows are connected when any matrix cell shares
        a frontier partition with any cell in another row. This keeps coupled
        conflict structures together and lets independent components be
        recombined exhaustively.

    ALGORITHM:
        1. Build connected components over all rows
        2. Return one matrix per connected component

    Args:
        matrix: A list of lists representing the matrix to split.

    Returns:
        A list of matrices, split to minimize the number of matrices.
        Returns the original matrix in a list if no effective split is found.
    """
    components = _connected_row_components(matrix)

    if len(components) <= 1:
        return [matrix]

    if not jt_logger.disabled:
        jt_logger.info(f"Splitting into {len(components)} connected components")

    return components


# ---------------------------
# Lattice and Dependent Solution Functions
# ---------------------------


def union_split_matrix_results(
    matrices: List[PMatrix], meet_fn: Optional[MeetFunction] = None
) -> list[PartitionSet[Partition]]:
    """
    Apply generalized meet product to each matrix and recombine alternatives.

    This approach:
    1. Applies generalized_meet_product to each split matrix independently
    2. Computes the Cartesian product of independent component candidates
    3. Unions one candidate from each component without minimizing away witnesses

    Args:
        matrices: List of matrices from matrix splitting
        meet_fn: Optional function to use for 'meet' (intersection) operation.

    Returns:
        List of PartitionSet solutions where each solution contains one
        witness set from each independent component.
    """
    if not matrices:
        return []

    if len(matrices) == 1:
        return generalized_meet_product(matrices[0], meet_fn)

    return _cartesian_matrix_results(matrices, meet_fn)


def _cartesian_matrix_results(
    matrices: List[PMatrix], meet_fn: Optional[MeetFunction] = None
) -> list[PartitionSet[Partition]]:
    """Combine independent submatrix candidates without dropping alternatives."""
    all_results: list[list[PartitionSet[Partition]]] = [
        generalized_meet_product(matrix, meet_fn) for matrix in matrices
    ]

    if any(not results for results in all_results):
        return []

    final_solutions: list[PartitionSet[Partition]] = []
    seen: set[tuple[int, ...]] = set()

    for combination in product(*all_results):
        all_partitions: set[Partition] = set()
        for result in combination:
            all_partitions.update(result)

        if not all_partitions:
            continue

        encoding = next(iter(all_partitions)).encoding
        combined: PartitionSet[Partition] = PartitionSet(
            all_partitions, encoding=encoding, name="cartesian_combined"
        )
        key = tuple(sorted(partition.bitmask for partition in combined))
        if key in seen:
            continue
        seen.add(key)
        final_solutions.append(combined)

    return final_solutions
