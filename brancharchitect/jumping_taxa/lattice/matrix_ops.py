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
    jt_logger.subsection("Vector Meet Product Computation")
    rows, cols = _validate_matrix(matrix)
    if rows != 1 or cols != 2:
        raise ValueError("Expected a 1x2 matrix for vector meet product.")

    jt_logger.matrix(matrix, title="Input Vector")

    a, b = matrix[0]
    jt_logger.info("Mathematical Step-by-Step Intersection:")
    jt_logger.info(f"  Set A: {a}")
    jt_logger.info(f"  Set B: {b}")

    # Show detailed intersection calculation
    if a and b:
        a_elements = set(a)
        b_elements = set(b)
        jt_logger.info(f"  A elements: {sorted(list(a_elements))}")
        jt_logger.info(f"  B elements: {sorted(list(b_elements))}")

        intersection_elements = a_elements.intersection(b_elements)
        jt_logger.info(f"  A ∩ B elements: {sorted(list(intersection_elements))}")

        result = a & b
        jt_logger.info(f"  Result partition set: {result if result else '∅'}")
    else:
        result = a & b
        jt_logger.info(
            f"  One or both sets empty → Result: {result if result else '∅'}"
        )

    jt_logger.info(f"→ Final intersection result: {result if result else '∅'}")
    return [result] if result else []


def generalized_meet_product(matrix: PMatrix) -> list[PartitionSet[Partition]]:
    jt_logger.section("Generalized Meet Product")
    rows, cols = _validate_matrix(matrix)

    jt_logger.info(f"Processing {rows}x{cols} matrix")
    jt_logger.matrix(matrix, title="Input Matrix")

    # Decision flow logging
    if rows == 1 and cols == 2:
        jt_logger.info("Using vector meet product for 1x2 matrix")
        return _vector_meet_product(matrix)
    elif rows == cols:
        jt_logger.info(f"Using square meet product for {rows}x{cols} matrix")
        return _square_meet_product(matrix)
    else:
        jt_logger.error(f"Unsupported matrix dimensions: {rows}x{cols}")
        raise ValueError(
            f"Generalized meet product not implemented for {rows}x{cols} matrices."
        )


def _square_meet_product(matrix: PMatrix) -> list[PartitionSet[Partition]]:
    jt_logger.subsection("Square Meet Product Computation")
    jt_logger.matrix(matrix, title="Input Square Matrix")
    rows, cols = _validate_matrix(matrix)
    if rows != cols:
        raise ValueError("Matrix must be square for square meet product.")

    if rows == 1:
        result = matrix[0][0]
        jt_logger.info(f"1x1 matrix result: {result}")
        return [result] if result else []

    if rows == 2:
        jt_logger.info(
            "Computing 2x2 diagonal intersections with detailed mathematics:"
        )

        # Main diagonal: [0,0] ∩ [1,1]
        jt_logger.info("\n  📐 Main Diagonal Intersection [0,0] ∩ [1,1]:")
        a00, a11 = matrix[0][0], matrix[1][1]
        jt_logger.info(f"    Position [0,0]: {a00}")
        jt_logger.info(f"    Position [1,1]: {a11}")

        if a00 and a11:
            elements_00 = set(a00)
            elements_11 = set(a11)
            jt_logger.info(f"    [0,0] elements: {sorted(list(elements_00))}")
            jt_logger.info(f"    [1,1] elements: {sorted(list(elements_11))}")

            main_intersection = elements_00.intersection(elements_11)
            jt_logger.info(f"    Common elements: {sorted(list(main_intersection))}")

        main_diag = a00 & a11
        jt_logger.info(f"    → Main diagonal result: {main_diag if main_diag else '∅'}")

        # Counter diagonal: [0,1] ∩ [1,0]
        jt_logger.info("\n  📐 Counter Diagonal Intersection [0,1] ∩ [1,0]:")
        a01, a10 = matrix[0][1], matrix[1][0]
        jt_logger.info(f"    Position [0,1]: {a01}")
        jt_logger.info(f"    Position [1,0]: {a10}")

        if a01 and a10:
            elements_01 = set(a01)
            elements_10 = set(a10)
            jt_logger.info(f"    [0,1] elements: {sorted(list(elements_01))}")
            jt_logger.info(f"    [1,0] elements: {sorted(list(elements_10))}")

            counter_intersection = elements_01.intersection(elements_10)
            jt_logger.info(f"    Common elements: {sorted(list(counter_intersection))}")

        counter_diag = a01 & a10
        jt_logger.info(
            f"    → Counter diagonal result: {counter_diag if counter_diag else '∅'}"
        )

        # Collect non-empty results
        results: list[PartitionSet[Partition]] = []
        jt_logger.info("\n  📋 Collecting Non-Empty Results:")

        if main_diag:
            results.append(main_diag)
            jt_logger.info(f"    ✓ Main diagonal solution added: {main_diag}")
        else:
            jt_logger.info("    ✗ Main diagonal empty - not added")

        if counter_diag:
            results.append(counter_diag)
            jt_logger.info(f"    ✓ Counter diagonal solution added: {counter_diag}")
        else:
            jt_logger.info("    ✗ Counter diagonal empty - not added")

        jt_logger.info(
            f"\n→ Found {len(results)} total solution(s) from diagonal intersections"
        )
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

    jt_logger.section("Matrix Splitting Analysis")
    jt_logger.info(
        f"Analyzing {len(matrix)}x{len(matrix[0]) if matrix else 0} matrix for potential splitting"
    )
    jt_logger.matrix(matrix, title="Input Matrix")

    # Special case: if it's a 2x2 matrix with different left column values, don't split
    if (
        len(matrix) == 2
        and len(matrix[0]) == 2
        and frozenset(matrix[0][0]) != frozenset(matrix[1][0])
    ):
        jt_logger.info("2x2 matrix with independent rows - no split needed")
        jt_logger.debug(f"  Row 1 left: {matrix[0][0]}, Row 2 left: {matrix[1][0]}")
        return [matrix]

    jt_logger.subsection("Grouping Rows by Left Column Values")

    groups: dict[
        FrozenPartitionSet[Partition], list[list[PartitionSet[Partition]]]
    ] = {}

    for row_idx, row in enumerate(matrix):
        if not row or len(row) < 2:
            jt_logger.warning(f"  Skipping row {row_idx + 1}: insufficient columns")
            continue

        key: PartitionSet[Partition] = row[0]  # The left column value
        # Convert to FrozenPartitionSet for grouping
        frozen_key = FrozenPartitionSet(
            set(key), encoding=getattr(key, "encoding", None)
        )

        if frozen_key not in groups:
            groups[frozen_key] = []
            jt_logger.info(f"  New group created for left value: {key}")

        groups[frozen_key].append(row)
        jt_logger.debug(f"  Row {row_idx + 1} added to group with key: {key}")

    jt_logger.info(f"Total groups formed: {len(groups)}")

    if len(groups) <= 1:
        jt_logger.info("No split needed - single unique left column value")
        return [matrix]

    jt_logger.subsection("Creating Split Matrices")

    result_matrices: List[PMatrix] = []
    for group_idx, (group_key, rows) in enumerate(groups.items()):
        new_matrix: PMatrix = [r[:] for r in rows]
        result_matrices.append(new_matrix)
        jt_logger.info(
            f"  Group {group_idx + 1}: {len(rows)} rows with left value {group_key}"
        )

    jt_logger.subsection("Split Results")
    jt_logger.info(f"Created {len(result_matrices)} sub-matrices")

    for i, split in enumerate(result_matrices):
        jt_logger.matrix(
            split,
            title=f"Sub-matrix {i + 1} ({len(split)}x{len(split[0]) if split else 0})",
        )

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
    jt_logger.subsection("Row-wise Intersection Calculations")
    results: List[PartitionSet[Partition]] = []

    for row_idx, row in enumerate(matrix):
        jt_logger.info(f"\n  Row {row_idx} intersection:")
        if not row:
            jt_logger.info(f"    Empty row → ∅")
            results.append(PartitionSet())
            continue

        jt_logger.info(f"    Row elements: {[str(elem) for elem in row]}")

        if len(row) == 1:
            result = row[0]
            jt_logger.info(f"    Single element → {result}")
        else:
            # Show step-by-step intersection for multiple elements
            result = row[0]
            jt_logger.info(f"    Starting with: {result}")

            for i, elem in enumerate(row[1:], 1):
                prev_result = result
                result = result & elem
                jt_logger.info(
                    f"    Step {i}: {prev_result} ∩ {elem} = {result if result else '∅'}"
                )

        jt_logger.info(
            f"    → Row {row_idx} final intersection: {result if result else '∅'}"
        )
        results.append(result)

    return results


def infer_mapping(matrix1: PMatrix, matrix2: PMatrix) -> list[tuple[int, int]]:
    """
    Infer a dependency mapping from the two matrices.
    If matrices have the same number of rows, use the standard reverse mapping.
    If they differ, create a mapping that pairs as many rows as possible.
    """
    n1: int = len(matrix1)
    n2: int = len(matrix2)

    jt_logger.subsection("Row Mapping Strategy")
    jt_logger.info("Creating row-to-row correspondence between the two matrices")
    jt_logger.info(f"Matrix 1 has {n1} rows, Matrix 2 has {n2} rows")

    if n1 == n2:
        # Standard case: same number of rows
        n: int = n1
        jt_logger.info("→ Using REVERSE MAPPING (symmetric matrices)")
        jt_logger.info("Explanation: Row i in Matrix 1 maps to row (n-1-i) in Matrix 2")
        jt_logger.info("This creates dependency pairs where:")
        jt_logger.info("  • First row of M1 pairs with LAST row of M2")
        jt_logger.info("  • Second row of M1 pairs with SECOND-TO-LAST row of M2")
        jt_logger.info("  • And so on...")
        jt_logger.info("This reverse pattern helps find complementary solutions.")

        mapping: list[tuple[int, int]] = [(i, n - 1 - i) for i in range(n)]
        jt_logger.info(f"Creating {n} reverse mappings:")
        for i, j in mapping:
            jt_logger.info(f"  M1[{i}] ↔ M2[{j}] (forward row {i} → reverse row {j})")

        return mapping
    else:
        # Fallback case: different number of rows
        # Map as many as possible, using the smaller matrix size
        min_n = min(n1, n2)
        jt_logger.warning("→ Using PARTIAL MAPPING (asymmetric matrices)")
        jt_logger.warning(
            "Explanation: Matrices have different sizes, so we can't use pure reverse mapping"
        )
        jt_logger.warning(
            f"We'll map as many rows as possible using the smaller size ({min_n})"
        )
        jt_logger.warning("For each row i in the smaller range:")
        jt_logger.warning("  • M1 uses forward index: i")
        jt_logger.warning(
            "  • M2 uses reverse index: min(n2-1-i, n2-1), clamped to [0, n2-1]"
        )

        mapping: list[tuple[int, int]] = []

        jt_logger.info(f"Creating {min_n} partial mappings:")
        for i in range(min_n):
            # For matrix1, use index i
            # For matrix2, use reverse index but within its bounds
            j_unclamped = n2 - 1 - i
            j = min(j_unclamped, n2 - 1)
            j = max(0, j)  # Ensure non-negative
            mapping.append((i, j))

            if j_unclamped != j:
                jt_logger.info(
                    f"  M1[{i}] ↔ M2[{j}] (wanted M2[{j_unclamped}], clamped to {j})"
                )
            else:
                jt_logger.info(f"  M1[{i}] ↔ M2[{j}] (reverse mapping)")

        # Show unmapped rows if any
        unmapped_m1 = list(range(min_n, n1))
        unmapped_m2 = list(range(min_n, n2))
        if unmapped_m1:
            jt_logger.warning(f"Unmapped M1 rows: {unmapped_m1}")
        if unmapped_m2:
            jt_logger.warning(f"Unmapped M2 rows: {unmapped_m2}")

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
    contain exactly one element each. For each mapping (i, j), if the intersection set from
    matrix1's row i and matrix2's row j both contain exactly one element, that unique pair
    is returned. Returns a list of unique solution tuples.
    """

    jt_logger.section("Computing Dependent Unique Solutions")

    # Display both matrices side by side conceptually
    jt_logger.subsection("Matrix 1")
    jt_logger.matrix(matrix1)

    jt_logger.subsection("Matrix 2")
    jt_logger.matrix(matrix2)

    # Compute row intersections
    inters1 = compute_row_intersections(matrix1)
    inters2 = compute_row_intersections(matrix2)

    # Display intersections cleanly
    jt_logger.subsection("Row Intersections")
    jt_logger.info("Matrix 1 row intersections:")
    for i, inter in enumerate(inters1):
        jt_logger.info(f"  Row {i}: {inter if inter else '∅'}")

    jt_logger.info("Matrix 2 row intersections:")
    for i, inter in enumerate(inters2):
        jt_logger.info(f"  Row {i}: {inter if inter else '∅'}")

    # Compute mapping and find unique solutions
    mapping: list[tuple[int, int]] = infer_mapping(matrix1, matrix2)
    unique_solutions: list[PartitionSet[Partition]] = []

    jt_logger.subsection("Finding Unique Solutions Using Row Mapping")
    jt_logger.info("Checking each mapped row pair for unique solutions:")

    for i, j in mapping:
        inter1_size = len(inters1[i])
        inter2_size = len(inters2[j])

        jt_logger.info(f"\n  Checking M1[{i}] ↔ M2[{j}]:")
        jt_logger.info(f"    M1[{i}] intersection: {inters1[i]} (size={inter1_size})")
        jt_logger.info(f"    M2[{j}] intersection: {inters2[j]} (size={inter2_size})")

        if inter1_size == 1 and inter2_size == 1:
            a: Partition = next(iter(inters1[i]))
            b: Partition = next(iter(inters2[j]))
            solution: PartitionSet[Partition] = PartitionSet(
                {a, b}, name="dependent", encoding=a.encoding
            )
            unique_solutions.append(solution)
            jt_logger.info(
                f"    → ✓ Both have exactly 1 element! Creating solution: {{{a}, {b}}}"
            )
        else:
            reason = []
            if inter1_size != 1:
                if inter1_size == 0:
                    reason.append(f"M1[{i}] is empty")
                else:
                    reason.append(
                        f"M1[{i}] has {inter1_size} elements (need exactly 1)"
                    )
            if inter2_size != 1:
                if inter2_size == 0:
                    reason.append(f"M2[{j}] is empty")
                else:
                    reason.append(
                        f"M2[{j}] has {inter2_size} elements (need exactly 1)"
                    )
            jt_logger.info(f"    → ✗ Not a unique solution: {' and '.join(reason)}")

    jt_logger.subsection("Unique Solutions Summary")
    if unique_solutions:
        jt_logger.info(f"✓ Found {len(unique_solutions)} unique solution(s):")
        for idx, sol in enumerate(unique_solutions, 1):
            jt_logger.info(f"  {idx}. {sol}")
    else:
        jt_logger.info(
            "✗ No unique solutions found - no mapped row pairs had exactly one element each"
        )

    return unique_solutions


def union_split_matrix_results(
    matrices: List[PMatrix],
) -> list[PartitionSet[Partition]]:
    """
    Apply generalized meet product to each matrix and create paired solutions using reverse mapping.

    This approach:
    1. Applies generalized_meet_product to each split matrix independently
    2. Uses reverse mapping logic to pair results from different matrices
    3. Creates solutions that maintain dependency relationships

    Args:
        matrices: List of matrices from matrix splitting

    Returns:
        List of PartitionSet solutions where each solution contains
        paired partitions following reverse mapping logic
    """
    jt_logger.subsection("Union Split Matrix Results with Pairing")

    if not matrices:
        jt_logger.info("No matrices provided")
        return []

    if len(matrices) == 1:
        jt_logger.info("Single matrix - using direct generalized meet product")
        return generalized_meet_product(matrices[0])

    if len(matrices) == 2:
        jt_logger.info("Two matrices - using reverse mapping pairing logic")
        return _pair_two_matrix_results(matrices[0], matrices[1])

    # For more than 2 matrices, use the original union approach
    jt_logger.info(f"{len(matrices)} matrices - using generalized union approach")
    return _union_multiple_matrices(matrices)


def _pair_two_matrix_results(
    matrix1: PMatrix, matrix2: PMatrix
) -> list[PartitionSet[Partition]]:
    """
    Handle the specific case of two matrices with reverse mapping pairing logic.
    """
    jt_logger.info("Applying generalized meet product to both matrices:")

    # Get results from each matrix
    jt_logger.info("Matrix 1:")
    result1 = generalized_meet_product(matrix1)
    jt_logger.info("Matrix 2:")
    result2 = generalized_meet_product(matrix2)

    jt_logger.subsection("Creating Reverse Mapping Pairs with Detailed Mathematics")
    jt_logger.info(f"Matrix 1 produced {len(result1)} results")
    jt_logger.info(f"Matrix 2 produced {len(result2)} results")

    # Apply reverse mapping logic: result[i] pairs with result[n-1-i]
    n1, n2 = len(result1), len(result2)

    if n1 == n2 and n1 > 0:
        jt_logger.info(f"✓ Using standard reverse mapping for {n1} pairs")
        jt_logger.info("📚 Reverse Mapping Theory:")
        jt_logger.info(
            "  • Position i in Matrix 1 pairs with position (n-1-i) in Matrix 2"
        )
        jt_logger.info("  • This creates complementary dependency relationships")
        jt_logger.info(
            "  • First result pairs with last, second with second-to-last, etc."
        )
        jt_logger.info(
            f"  • For n={n1}: mappings are {[(i, n1 - 1 - i) for i in range(n1)]}"
        )

        final_solutions = []

        for i in range(n1):
            j = n1 - 1 - i  # Reverse index
            jt_logger.info(
                f"\n  🔗 Pairing Step {i + 1}/{n1}: Position {i} ↔ Position {j}"
            )
            jt_logger.info(f"    Matrix 1[{i}]: {result1[i]}")
            jt_logger.info(f"    Matrix 2[{j}]: {result2[j]}")

            # Show detailed union calculation
            jt_logger.info("    📊 Union Calculation:")
            set1 = set(result1[i]) if result1[i] else set()
            set2 = set(result2[j]) if result2[j] else set()

            jt_logger.info(
                f"      Matrix 1[{i}] elements: {sorted(list(set1)) if set1 else '∅'}"
            )
            jt_logger.info(
                f"      Matrix 2[{j}] elements: {sorted(list(set2)) if set2 else '∅'}"
            )

            # Create union of the paired results
            all_partitions = set1.union(set2)
            jt_logger.info(
                f"      Union result: {sorted(list(all_partitions)) if all_partitions else '∅'}"
            )

            # Show symmetric difference for completeness
            sym_diff = set1.symmetric_difference(set2)
            if sym_diff:
                jt_logger.info(
                    f"      Symmetric difference: {sorted(list(sym_diff))} (elements in exactly one set)"
                )
            else:
                jt_logger.info(
                    "      Symmetric difference: ∅ (sets are identical or one is empty)"
                )

            if all_partitions:
                encoding = next(iter(all_partitions)).encoding if all_partitions else {}
                paired_solution = PartitionSet(
                    all_partitions, encoding=encoding, name=f"paired_{i}"
                )
                final_solutions.append(paired_solution)
                jt_logger.info(f"      → ✓ Paired solution created: {paired_solution}")
            else:
                jt_logger.info(f"      → ✗ Empty pairing at position {i} - skipping")

        jt_logger.info(
            f"\n→ ✅ Created {len(final_solutions)} paired solution(s) total"
        )
        return final_solutions

    else:
        jt_logger.warning(
            f"⚠️  Asymmetric results ({n1} vs {n2}) - falling back to union approach"
        )
        jt_logger.info("📝 Asymmetric Case: Cannot use pure reverse mapping")
        jt_logger.info("   Reason: Different number of results from each matrix")
        jt_logger.info("   Fallback: Using generalized union approach instead")
        return _union_multiple_matrices([matrix1, matrix2])


def _union_multiple_matrices(matrices: List[PMatrix]) -> list[PartitionSet[Partition]]:
    """
    Original union approach for multiple matrices (fallback).
    """
    # Get results from each matrix
    all_results = []
    for i, matrix in enumerate(matrices):
        jt_logger.info(f"Processing matrix {i + 1}/{len(matrices)}")
        result = generalized_meet_product(matrix)
        all_results.append(result)
        jt_logger.debug(f"Matrix {i + 1} results: {len(result)} solution(s)")

    # Find the maximum number of results across all matrices
    max_results = max(len(results) for results in all_results) if all_results else 0
    jt_logger.info(f"Creating {max_results} union solutions")

    final_solutions = []
    for pos in range(max_results):
        # Collect all partitions from this position across matrices
        all_partitions = set()

        for matrix_idx, results in enumerate(all_results):
            if pos < len(results) and results[pos]:
                partitions_in_result = set(results[pos])
                all_partitions.update(partitions_in_result)

        if all_partitions:
            # Create union of all partitions at this position
            encoding = next(iter(all_partitions)).encoding if all_partitions else {}
            union_result = PartitionSet(
                all_partitions, encoding=encoding, name=f"union_{pos}"
            )
            final_solutions.append(union_result)

    return final_solutions


def solve_matrix_puzzle(
    matrix1: list[list[PartitionSet[Partition]]],
    matrix2: list[list[PartitionSet[Partition]]],
) -> list[PartitionSet[Partition]]:
    """Legacy function - now uses the new union approach."""
    return union_split_matrix_results([matrix1, matrix2])
