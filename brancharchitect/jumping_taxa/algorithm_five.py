from brancharchitect.tree import Node
from brancharchitect.jumping_taxa.elemental import find_exact_max_intersection
from brancharchitect.jumping_taxa.debug import (
    jt_logger,
    format_set,
)
from brancharchitect.jumping_taxa.matrix_ops import (
    generalized_meet_product,
    create_matrix,
    split_matrix,
    canonicalize_diagonal_swap_2,
    matrix_has_singleton_entries,
    solve_matrix_puzzle,
    
)
from brancharchitect.jumping_taxa.functional_tree import (
    compare_tree_splits,
    HasseEdge,
)


def format_covet_matrix(a_idx, b_idx, a_set, b_set):
    """Format arms as a 2x2 matrix with diagonal distribution and labels"""
    return f"""Covet A{a_idx} × B{b_idx}:<br> ⎡[[{format_set(a_set)}]⎤<br>⎣[{format_set(b_set)}]⎦"""


# ============================================== Case For Edge Types ====================================================== #
def case_partial_none(sedge: HasseEdge):
    r = find_exact_max_intersection(sedge.arms_t_one, sedge.arms_t_two)
    return r


def set_rule_based_algorithm(s_edge: HasseEdge):
    inter = []
    sym = []
    results = []

    jt_logger.log_covet(s_edge.left_cover, s_edge.right_cover)
    process_of_direction_intersection: dict = {}
    process_of_direction_a_without_b: dict = {}
    process_of_direction_b_without_a: dict = {}

    for a_idx, a in enumerate(s_edge.left_cover, 1):
        for b_idx, b in enumerate(s_edge.right_cover, 1):

            i = set(a) & set(b)
            a, b = set(a), set(b)

            a_without_b = set(a) - set(b)
            b_without_a = set(b) - set(a)

            if i:
                inter.append(tuple(i))
            if a_without_b:
                sym.append(a_without_b)
            if b_without_a:
                sym.append(b_without_a)

            process_of_direction_intersection[frozenset(i)] = {
                "covet_left": a,
                "covet_right": b,
                "b-a": b_without_a,
                "a-b": a_without_b,
            }

            process_of_direction_a_without_b[frozenset(a - b)] = {
                "covet_left": a,
                "covet_right": b,
                "index": (a_idx, b_idx),
                "b-a": b_without_a,
                "a-b": a_without_b,
            }

            process_of_direction_b_without_a[frozenset(b - a)] = {
                "covet_left": a,
                "covet_right": b,
                "index": (a_idx, b_idx),
                "b-a": b_without_a,
                "a-b": a_without_b,
            }

            results.append(
                [
                    format_covet_matrix(a_idx, b_idx, a, b),  # Show arms as matrix
                    format_set(i).ljust(40),
                    format_set(a_without_b).ljust(40),
                    format_set(b_without_a).ljust(40),
                ]
            )

    # Log comparison analysis
    jt_logger.comparison_analysis(results, inter, sym)

    direction_by_intersection = []
    for intersection in process_of_direction_intersection:
        if not intersection:
            continue

        # Check if intersection exists in both difference dictionaries
        exists_in_both = (
            intersection in process_of_direction_a_without_b
            and intersection in process_of_direction_b_without_a
        )

        if not exists_in_both:
            continue

        # Get arm information for this intersection
        a_info = process_of_direction_a_without_b[intersection]
        b_info = process_of_direction_b_without_a[intersection]

        if not (a_info and b_info):
            continue

        # Get arms for comparison
        arm_a = a_info["covet_left"]
        arm_b = b_info["covet_right"]

        # Check if neither arm is subset of the other
        is_independent = (
            (
                (not arm_a.issubset(arm_b))
                and process_of_direction_a_without_b[intersection]["b-a"]
            )
            or (
                (not arm_b.issubset(arm_a))
                and process_of_direction_b_without_a[intersection]["a-b"]
            )
            or (arm_a.issubset(arm_b) and len(arm_a) == 1)
            or (arm_b.issubset(arm_a) and len(arm_b) == 1)
        )

        jt_logger.info(
            f"Intersection: {format_set(intersection)}; Arm A: {format_set(arm_a)}; Arm B: {format_set(arm_b)}; Independent: {is_independent}"
        )

        # Check if neither arm is subset of the other
        jt_logger.section(
            f"Independence Check for intersection: {format_set(intersection)}"
        )
        jt_logger.info(f"Arm A: {format_set(arm_a)}")
        jt_logger.info(f"Arm B: {format_set(arm_b)}")

        # Evaluate each condition separately for detailed logging
        condition1 = (not arm_a.issubset(arm_b)) and process_of_direction_a_without_b[
            intersection
        ]["b-a"]
        condition2 = (not arm_b.issubset(arm_a)) and process_of_direction_b_without_a[
            intersection
        ]["a-b"]
        condition3 = arm_a.issubset(arm_b) and len(arm_a) == 1 and len(arm_b) > 1
        condition4 = arm_b.issubset(arm_a) and len(arm_b) == 1 and len(arm_a) > 1

        # Log detailed subset relationships
        jt_logger.info(f"Not A ⊆ B: {arm_a.issubset(arm_b)}")
        jt_logger.info(f"Not B ⊆ A: {arm_b.issubset(arm_a)}")
        jt_logger.info(
            f"A - B: {format_set(process_of_direction_a_without_b[intersection]['a-b'])}"
        )
        jt_logger.info(
            f"B - A: {format_set(process_of_direction_b_without_a[intersection]['b-a'])}"
        )

        # Log which condition was satisfied
        if condition1:
            jt_logger.info("✓ Independent: A is not subset of B and B-A is non-empty")
        elif condition2:
            jt_logger.info("✓ Independent: B is not subset of A and A-B is non-empty")
        elif condition3:
            jt_logger.info("✓ Independent: A is singleton subset of B")
        elif condition4:
            jt_logger.info("✓ Independent: B is singleton subset of A")
        else:
            jt_logger.info("✗ Not independent: None of the independence conditions met")

        is_independent = condition1 or condition2 or condition3 or condition4
        jt_logger.info(f"Final independence determination: {is_independent}")

        if is_independent:

            direction_by_intersection.append(
                {
                    "pair": a_info["index"],  # Changed from "index" to "pair"
                    "A": frozenset(arm_a),
                    "B": frozenset(arm_b),
                    "direction_a": (1, 0),
                    "direction_b": (0, 1),
                    "common": intersection,  # Add intersection as common elements
                }
            )

    # Replace print with proper logging
    jt_logger.log_bidirectional_analysis(direction_by_intersection)

    # Create matrices using new function
    matrix = create_matrix(direction_by_intersection)

    jt_logger.matrix(matrix)

    matrices = split_matrix(matrix)

    solutions = []

    jt_logger.section("Meet Result Computation")

    if len(matrices) == 2:
        dependent_solutions = solve_matrix_puzzle(matrices[0], matrices[1])
        for solution in dependent_solutions:
            jt_logger.info(f"Dependent Solution: {solution}")
            
        return dependent_solutions[0]
    for matrix in matrices:
        meet_results = None
        if (matrix_has_singleton_entries(matrix)) and len(matrix) > 1:
            jt_logger.info("Matrix is canonical")
            matrix = canonicalize_diagonal_swap_2(matrix)
            meet_results = generalized_meet_product(matrix)
        else:
            meet_results = generalized_meet_product(matrix)

        jt_logger.log_meet_result(meet_results)

        # Find minimum size of results
        min_size = min(len(result) for result in meet_results)
        # Get all results of minimum size

        min_results = [result for result in meet_results if len(result) == min_size]

        jt_logger.info(f"Minimum size of results: {min_size}")

        solutions.append(tuple(min_results[0]))

        jt_logger.info(f"Taken Result: {format_set(min_results[0])}")

    return solutions if solutions else tuple()

# ============================================== Algorithm 5 ====================================================== #


def algorithm_5_for_sedge(s_edge: HasseEdge):
    """
    Process a single s-edge based on its classification.
    """
    # Add tree visualization before processing
    jt_logger.print_sedge_comparison(s_edge)

    # Log s-edge classification information
    jt_logger.info("\nS-Edge Classification:")
    jt_logger.info(f"Split: {s_edge.split}")
    jt_logger.info(f"Tree 1 Node: {s_edge.left_node}")
    jt_logger.info(f"Tree 2 Node: {s_edge.right_node}")

    # Map the pair of edge types to the corresponding function call.
    conditions = {
        ("divergent", "divergent"): lambda: set_rule_based_algorithm(s_edge),
        ("divergent", "intermediate"): lambda: set_rule_based_algorithm(s_edge),
        ("intermediate", "divergent"): lambda: set_rule_based_algorithm(s_edge),
        ("intermediate", "intermediate"): lambda: set_rule_based_algorithm(s_edge),
        ("intermediate", "collapsed"): lambda: set_rule_based_algorithm(s_edge),
        ("collapsed", "intermediate"): lambda: set_rule_based_algorithm(s_edge),
    }

    # Create the key from the s-edge's types.
    key = s_edge.get_edge_types()

    result = None
    if key in conditions:
        result = conditions[key]()
        for comp in result:
            for c in comp:
                jt_logger.info(f"Component: {format_set({c})}")
    else:
        raise Exception(f"We forgot one case: {s_edge}")

    return result


@jt_logger.log_execution
def algorithm_five(input_tree1: Node, input_tree2: Node, leaf_order: list[str]):
    """
    Runs 'algorithm five' on two trees, pruning iteratively based on discovered components.

    Args:
        input_tree1 (Node): The first input tree (root node).
        input_tree2 (Node): The second input tree (root node).
        leaf_order (list[str]): The list of leaf labels in a certain (sorted) order.

    Returns:
        list[int]: A list of unique components (encoded as integer indices) discovered by the algorithm.
    """

    try:
        global_components: list[int] = []

        pruned_original_tree1, pruned_original_tree2 = input_tree1, input_tree2
        current_s_edges = compare_tree_splits(input_tree1, input_tree2)
        remaining_leaves = len(leaf_order)

        while True:
            iteration_components = []
            iteration_taxa = set()

            for current_s_edge in current_s_edges.values():
                jt_logger.section(f"\nProcessing split: {current_s_edge.split}")

                new_components = algorithm_5_for_sedge(current_s_edge)
                iteration_components.extend(list(new_components))

                # Update iteration_taxa with proper set operations
                for comp in new_components:
                    iteration_taxa.update(tuple(comp))

                global_components.extend(new_components)

                jt_logger.info(f"Global Components: {global_components}")

            current_proposed_deletions = len(iteration_taxa)

            jt_logger.info(f"Current Proposed Deletions: {current_proposed_deletions}")

            if (
                iteration_components
                and (remaining_leaves - current_proposed_deletions) > 3
            ):
                remaining_leaves -= current_proposed_deletions

                pruned_original_tree1, pruned_original_tree2 = delete_leaves(
                    pruned_original_tree1, pruned_original_tree2, list(iteration_taxa)
                )

                current_s_edges = compare_tree_splits(
                    pruned_original_tree1, pruned_original_tree2
                )

                jt_logger.info(
                    f"Remaining leaves: {remaining_leaves}"
                )  # Change print to print_if_enabled

            elif not current_s_edges:
                jt_logger.info("No more common splits found. Algorithm completed.")
                break
            else:
                jt_logger.info(
                    "Minimum leaf count reached or no new components found. Algorithm completed."
                )
                break

        return list(global_components)
    except Exception as e:
        from brancharchitect.jumping_taxa.debug import log_stacktrace

        # Log to HTML output
        log_stacktrace(e)
        # Also print to console
        raise Exception(f"Error in algorithm_five: {str(e)}")


# ============================================== Pruning ====================================================== #
def delete_leaves(
    original_tree_one: Node, original_tree_two: Node, to_be_deleted_leaves=[]
):
    jt_logger.section("Deleting Leaves")
    jt_logger.info(f"Deleting Leaves: {to_be_deleted_leaves}")

    pruned_tree_one = original_tree_one.delete_taxa(to_be_deleted_leaves)
    jt_logger.info(f"Pruned Tree One: {pruned_tree_one.to_newick()}")

    pruned_tree_two = original_tree_two.delete_taxa(to_be_deleted_leaves)
    jt_logger.info(f"Pruned Tree Two: {pruned_tree_one.to_newick()}")
    return pruned_tree_one, pruned_tree_two
