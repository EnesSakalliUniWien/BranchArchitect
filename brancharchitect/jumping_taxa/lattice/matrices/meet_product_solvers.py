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

# Use FrozenPartitionSet for hashable, immutable keys
from brancharchitect.elements.frozen_partition_set import FrozenPartitionSet
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

        # Collect non-empty results
        results: list[PartitionSet[Partition]] = []

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

    results: list[PartitionSet[Partition]] = []
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


# ---------------------------
# Solution Metric Functions
# ---------------------------


def solution_size(sol: PartitionSet[Partition]) -> int:
    """
    Calculate the total number of taxa across all partitions in a solution.

    MATHEMATICAL DEFINITION:
        For a solution S = {P₁, P₂, ..., Pₙ} where each Pᵢ is a partition:
        size(S) = Σᵢ |Pᵢ.taxa|

    PHYLOGENETIC INTERPRETATION:
        The solution size represents the total number of individual taxa
        that must be treated as "jumping taxa" to resolve conflicts.
        Smaller solutions are preferred as they minimize phylogenetic
        disruption and represent more parsimonious explanations.

    OPTIMIZATION GOAL:
        When multiple nesting solutions exist, we select the one with
        minimal total size to reduce the number of taxa involved in
        the reticulation event.

    EXAMPLE:
        Solution 1: {(D1, D2), (E1, E2)}     → size = 2 + 2 = 4
        Solution 2: {(F1)}                   → size = 1       ✅ PREFERRED
        Solution 3: {(C1, C2)}               → size = 2

    Args:
        sol: A PartitionSet containing the partitions in the solution

    Returns:
        The total number of taxa across all partitions in the solution
    """
    return sum(len(partition.taxa) for partition in sol)


def matrix_row_size(row: list[PartitionSet[Partition]]) -> int:
    """
    Calculate the combined size of both columns in a matrix row.

    MATHEMATICAL DEFINITION:
        For a matrix row [C₁, C₂] where each Cᵢ is a PartitionSet:
        row_size([C₁, C₂]) = size(C₁) + size(C₂)
                            = Σ|P.taxa| for P ∈ C₁ + Σ|P.taxa| for P ∈ C₂

    PHYLOGENETIC INTERPRETATION:
        The row size represents the total number of taxa involved in
        a conflict pair. Smaller rows indicate more localized conflicts
        that affect fewer taxa, making them simpler to resolve.

    OPTIMIZATION GOAL:
        When building conflict matrices, prioritize rows with smaller
        combined sizes to tackle simpler conflicts first, following
        a "divide and conquer" strategy for phylogenetic reconciliation.

    EXAMPLE:
        Row 1: [{(A, B)}, {(C, D, E)}]      → size = 2 + 3 = 5
        Row 2: [{(X)}, {(Y)}]               → size = 1 + 1 = 2  ✅ PREFERRED
        Row 3: [{(F, G, H)}, {(I, J)}]      → size = 3 + 2 = 5

    Args:
        row: A matrix row containing two PartitionSets (left and right covers)

    Returns:
        The sum of sizes of both PartitionSets in the row
    """
    return sum(solution_size(cell) for cell in row)


def _group_by_column(
    matrix: PMatrix, col_index: int
) -> dict[FrozenPartitionSet[Partition], list[list[PartitionSet[Partition]]]]:
    """Groups matrix rows based on the partition set in a specific column."""
    groups: dict[
        FrozenPartitionSet[Partition], list[list[PartitionSet[Partition]]]
    ] = {}
    for row in matrix:
        if not row or len(row) <= col_index:
            continue  # Skip rows that are too short

        key_ps = row[col_index]
        frozen_key = FrozenPartitionSet(
            set(key_ps), encoding=getattr(key_ps, "encoding", None)
        )

        if frozen_key not in groups:
            groups[frozen_key] = []
        groups[frozen_key].append(row)
    return groups


def split_matrix(matrix: PMatrix) -> list[PMatrix]:
    """
    Split a matrix into the MINIMAL number of smaller matrices.

    SPLITTING STRATEGY:
        Groups matrix rows by column values and chooses the grouping that
        produces the FEWEST matrices (minimal splitting for efficiency):
        - Compares left column grouping vs right column grouping
        - Selects the strategy with fewer groups
        - Fewer matrices → less computational overhead, simpler solutions

    ALGORITHM:
        1. Extract degenerate rows (singleton containment) as separate 1×2 matrices
        2. Check for special cases (independent 2×2 matrices - don't split)
        3. Group remaining rows by left column values
        4. Group remaining rows by right column values
        5. Choose grouping with FEWER groups (minimal split matrices)
        6. If equal, prefer left column for consistency

    Args:
        matrix: A list of lists representing the matrix to split.

    Returns:
        A list of matrices, split to minimize the number of matrices.
        Returns the original matrix in a list if no effective split is found.
    """
    # Extract degenerate singleton rows using classifier
    base_rows, degenerate_matrices = MatrixClassifier.extract_degenerate_rows(matrix)

    if not base_rows:
        return degenerate_matrices

    # Special case: independent 2×2 matrix shouldn't be split
    if MatrixClassifier.is_independent_2x2(base_rows):
        return [base_rows] + degenerate_matrices

    # Group by left column (index 0)
    left_groups = _group_by_column(base_rows, 0)

    # Group by right column (index 1)
    right_groups = _group_by_column(base_rows, 1)

    # Choose the grouping that creates FEWER matrices (minimal splitting)
    # Fewer groups = fewer split matrices = more efficient
    if len(left_groups) < len(right_groups):
        chosen_groups = left_groups
        if not jt_logger.disabled:
            jt_logger.info(
                f"Splitting by left column: {len(left_groups)} groups (fewer than right: {len(right_groups)})"
            )
    elif len(right_groups) < len(left_groups):
        chosen_groups = right_groups
        if not jt_logger.disabled:
            jt_logger.info(
                f"Splitting by right column: {len(right_groups)} groups (fewer than left: {len(left_groups)})"
            )
    else:
        # Equal number of groups - prefer left column for consistency
        chosen_groups = left_groups
        if not jt_logger.disabled:
            jt_logger.info(f"Equal groups ({len(left_groups)}), using left column")

    # If no effective grouping is found (only one group or each row is a group), don't split
    if len(chosen_groups) <= 1 or len(chosen_groups) == len(base_rows):
        return [base_rows] + degenerate_matrices

    result_matrices: List[PMatrix] = []
    for _frozen_key, rows in chosen_groups.items():
        new_matrix: PMatrix = [r[:] for r in rows]
        result_matrices.append(new_matrix)

    return result_matrices + degenerate_matrices


# ---------------------------
# Lattice and Dependent Solution Functions
# ---------------------------


def union_split_matrix_results(
    matrices: List[PMatrix], meet_fn: Optional[MeetFunction] = None
) -> list[PartitionSet[Partition]]:
    """
    Apply generalized meet product to each matrix and create paired solutions using reverse mapping.

    This approach:
    1. Applies generalized_meet_product to each split matrix independently
    2. Uses reverse mapping logic to pair results from different matrices
    3. Creates solutions that maintain dependency relationships

    Args:
        matrices: List of matrices from matrix splitting
        meet_fn: Optional function to use for 'meet' (intersection) operation.

    Returns:
        List of PartitionSet solutions where each solution contains
        paired partitions following reverse mapping logic
    """
    if not matrices:
        return []

    if len(matrices) == 1:
        return generalized_meet_product(matrices[0], meet_fn)

    if len(matrices) == 2:
        return _pair_two_matrix_results(matrices[0], matrices[1], meet_fn)

    # For more than 2 matrices, use the original union approach
    return _union_multiple_matrices(matrices, meet_fn)


def _pair_two_matrix_results(
    matrix1: PMatrix, matrix2: PMatrix, meet_fn: Optional[MeetFunction] = None
) -> list[PartitionSet[Partition]]:
    """
    Handle the specific case of two matrices using a Structure-Preserving Pairing Strategy.

    This strategy pairs solutions from disjoint sub-problems in a complementary manner using
    reverse index mapping (Index i <-> Index n-1-i).

    **Rationale:**
    In the context of the meet product, solutions are typically ordered from "Most Conservative" (Main)
    to "Least Conservative" (Counter/Deletion). By pairing the most conservative option of one side
    with the least conservative option of the other, we enforce a "Trade-off" heuristic.
    This prevents "Aggressive Deletion" scenarios (Counter-Counter) where structure is destroyed
    on both sides, guiding the solver towards solutions that preserve at least one side's structure.
    """
    # Get results from each matrix
    result1 = generalized_meet_product(matrix1, meet_fn)
    result2 = generalized_meet_product(matrix2, meet_fn)

    # Apply reverse mapping logic: result[i] pairs with result[n-1-i]
    n1, n2 = len(result1), len(result2)

    if n1 == n2 and n1 > 0:
        final_solutions: list[PartitionSet[Partition]] = []

        for i in range(n1):
            j = n1 - 1 - i  # Reverse index

            # Create union of the paired results
            set1: set[Partition] = set(result1[i]) if result1[i] else set()
            set2: set[Partition] = set(result2[j]) if result2[j] else set()
            all_partitions: set[Partition] = set1.union(set2)

            if all_partitions:
                encoding = next(iter(all_partitions)).encoding if all_partitions else {}
                paired_solution: PartitionSet[Partition] = PartitionSet(
                    all_partitions, encoding=encoding, name=f"paired_{i}"
                )
                final_solutions.append(paired_solution)

        return final_solutions

    else:
        return _union_results([result1, result2])


def _union_results(
    all_results: list[list[PartitionSet[Partition]]],
) -> list[PartitionSet[Partition]]:
    """
    Creates minimum union solutions from a list of results.

    For each position, collects partitions from all matrices and computes
    the minimum cardinality cover - the smallest set of partitions that
    covers all indices from that position.

    This ensures solutions are minimal (no redundant partitions).
    """
    # Find the maximum number of results across all matrices
    max_results = max(len(results) for results in all_results) if all_results else 0

    final_solutions: list[PartitionSet[Partition]] = []
    for pos in range(max_results):
        # Collect all partitions from this position across matrices
        all_partitions: set[Partition] = set()

        for _matrix_idx, results in enumerate(all_results):
            if pos < len(results) and results[pos]:
                partitions_in_result: set[Partition] = set(results[pos])
                all_partitions.update(partitions_in_result)

        if all_partitions:
            # Create union of all partitions at this position
            encoding = next(iter(all_partitions)).encoding if all_partitions else {}
            union_result: PartitionSet[Partition] = PartitionSet(
                all_partitions, encoding=encoding, name=f"union_{pos}"
            )

            # Apply minimum_cover to get the smallest solution
            # This removes redundant partitions and ensures minimal cardinality
            minimized_result: PartitionSet[Partition] = union_result.minimum_cover()
            final_solutions.append(minimized_result)

    return final_solutions


def _union_multiple_matrices(
    matrices: List[PMatrix], meet_fn: Optional[MeetFunction] = None
) -> list[PartitionSet[Partition]]:
    """
    Union approach for multiple matrices (3 or more).

    This is a fallback strategy when more than 2 matrices are produced
    by split_matrix. It computes solutions for each matrix independently
    and unions them position-wise.
    """
    # Get results from each matrix
    all_results: list[list[PartitionSet[Partition]]] = []
    for matrix in matrices:
        result = generalized_meet_product(matrix, meet_fn)
        all_results.append(result)

    return _union_results(all_results)
