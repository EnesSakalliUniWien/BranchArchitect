import collections
from functools import lru_cache
from typing import Dict, Hashable, List


def calculate_mobius_function(
    poset_covers: Dict[Hashable, List[Hashable]],
) -> Dict[tuple[Hashable, Hashable], int]:
    """
    Calculates the Mobius function for all pairs (x, y) in a poset.

    Args:
        poset_covers: A dictionary representing the Hasse diagram of the poset.
                      Keys are elements, values are the list of elements they cover.

    Returns:
        A dictionary where keys are tuples (x, y) and values are mu(x, y).
    """
    elements = set(poset_covers.keys())
    for children in poset_covers.values():
        elements.update(children)

    # Build a successor mapping (parent -> children) to find all elements >= x
    successors = {el: set() for el in elements}
    for parent, children in poset_covers.items():
        for child in children:
            successors[parent].add(child)

    @lru_cache(maxsize=None)
    def get_all_successors(node):
        """Get all nodes reachable from a given node (reflexive)."""
        all_succs = {node}
        for succ in successors.get(node, []):
            all_succs.update(get_all_successors(succ))
        return all_succs

    # Pre-calculate the full <= relation (the Zeta function)
    # zeta[x] contains all y such that x <= y
    zeta = {el: get_all_successors(el) for el in elements}

    memo = {}
    sorted_elements = topological_sort(poset_covers)

    for y in sorted_elements:
        for x in sorted_elements:
            # Condition for non-zero Mobius function is x <= y
            if y not in zeta.get(x, set()):
                memo[(x, y)] = 0
                continue

            if x == y:
                memo[(x, y)] = 1
                continue

            # mu(x, y) = - sum(mu(x, z) for z where x <= z < y)
            sum_val = 0
            # Find all z such that x <= z < y
            for z in zeta.get(x, set()):
                if z != y and y in zeta.get(z, set()):  # x <= z and z < y
                    sum_val += memo.get((x, z), 0)

            memo[(x, y)] = -sum_val

    return memo


def topological_sort(poset_covers: Dict[Hashable, List[Hashable]]) -> List[Hashable]:
    """Performs a topological sort on the poset."""
    elements = set(poset_covers.keys())
    for children in poset_covers.values():
        elements.update(children)

    # Successor map for traversal
    successors = {el: [] for el in elements}
    for parent, children in poset_covers.items():
        successors[parent].extend(children)

    # In-degree for starting points
    in_degree = {el: 0 for el in elements}
    for children in poset_covers.values():
        for child in children:
            in_degree[child] += 1

    queue = collections.deque([el for el in elements if in_degree[el] == 0])
    sorted_list = []

    while queue:
        node = queue.popleft()
        sorted_list.append(node)

        for neighbor in successors.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(sorted_list) != len(elements):
        raise ValueError("Poset has a cycle and cannot be topologically sorted.")
    return sorted_list


if __name__ == "__main__":
    # 1. Define the poset from our example
    poset_name = "Boolean Lattice on {E, F, G}"
    poset_covers = {
        "∅": ["E", "F", "G"],
        "E": ["EF", "EG"],
        "F": ["EF", "FG"],
        "G": ["EG", "FG"],
        "EF": ["EFG"],
        "EG": ["EFG"],
        "FG": ["EFG"],
        "EFG": [],
    }

    print(f"--- Calculating Möbius Function for: {poset_name} ---")
    mobius_values = calculate_mobius_function(poset_covers)

    print("\nCalculated Möbius Function Values μ(∅, S):")
    # Sort the output for consistent viewing
    for end_node in topological_sort(poset_covers):
        val = mobius_values.get(("∅", end_node))
        print(f"  μ(∅, {end_node:<5}) = {val}")

    # 2. Demonstrate Inversion
    print("\n--- Demonstrating Möbius Inversion ---")
    f = collections.defaultdict(int)
    f["E"] = 1
    f["F"] = 1
    print(f"\nIntrinsic conflict function f: {dict(f)}")

    # Build predecessor map for g and recovered_f calculation
    elements = set(poset_covers.keys())
    for children in poset_covers.values():
        elements.update(children)
    predecessors = {el: set() for el in elements}
    for parent, children in poset_covers.items():
        for child in children:
            predecessors[child].add(parent)

    @lru_cache(maxsize=None)
    def get_all_predecessors(node):
        all_preds = {node}
        for pred in predecessors.get(node, []):
            all_preds.update(get_all_predecessors(pred))
        return all_preds

    # Calculate the aggregated function `g` where g(S) = sum(f(T) for T <= S)
    g = {}
    sorted_nodes = topological_sort(poset_covers)
    for y in sorted_nodes:
        sum_val = 0
        for t in get_all_predecessors(y):
            sum_val += f.get(t, 0)
        g[y] = sum_val
    print(f"Aggregated conflict function g: {g}")

    # 3. Recover `f` from `g` using the calculated Möbius values
    recovered_f = {}
    for y in sorted_nodes:
        sum_val = 0
        for t in get_all_predecessors(y):
            sum_val += mobius_values.get((t, y), 0) * g.get(t, 0)
        recovered_f[y] = sum_val

    print(f"\nRecovered intrinsic function f using inversion: {recovered_f}")
