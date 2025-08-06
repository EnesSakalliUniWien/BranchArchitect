from scipy.spatial.distance import pdist, squareform
from skbio import DistanceMatrix
from skbio.tree import nj


def collapse_short_branches(tree, threshold=1e-6):
    """Collapse short branches to create multifurcations."""
    while True:
        nodes_to_collapse = []
        for node in tree.postorder():
            if (
                not node.is_tip()
                and node.length is not None
                and node.length < threshold
            ):
                if node.parent:
                    nodes_to_collapse.append(node)
        if not nodes_to_collapse:
            break
        for node in nodes_to_collapse:
            if node.parent is None:
                continue
            parent = node.parent
            for child in node.children:
                child.length = (child.length or 0) + (node.length or 0)
            children = list(node.children)
            parent.remove(node)
            parent.extend(children)
    return tree


def apply_midpoint_rooting(tree):
    """Apply midpoint rooting to a scikit-bio tree."""
    try:
        # Use scikit-bio's built-in midpoint rooting
        rooted_tree = tree.root_at_midpoint()
        return str(rooted_tree)
    except Exception as e:
        print(f"Warning: Could not apply midpoint rooting: {e}")
        # Return original tree as fallback
        return str(tree)


def library_neighbor_joining(D, labels, collapse_threshold=1e-6, apply_rooting=True):
    """Construct NJ tree using scikit-bio with optional midpoint rooting."""
    dm = DistanceMatrix(D, labels)
    tree = nj(dm)
    tree = collapse_short_branches(tree, threshold=collapse_threshold)
    
    if apply_rooting:
        return apply_midpoint_rooting(tree)
    else:
        return str(tree)


def generate_nj_trees(full_traj, n_points, apply_rooting=True):
    """Generate NJ trees for each simulation step with optional midpoint rooting."""
    labels = [f"P{k}" for k in range(n_points)]
    actual_steps = full_traj.shape[1]

    nj_trees = [
        library_neighbor_joining(squareform(pdist(full_traj[:, s, :])), labels, apply_rooting=apply_rooting)
        for s in range(actual_steps)
    ]

    return nj_trees, labels
