from brancharchitect.jumping_taxa.tree_interpolation import interpolate_tree
from brancharchitect.jumping_taxa.algorithm_five import algorithm_five

def call_jumping_taxa(tree1, tree2):
    if tree1._order != tree2._order:
        raise ValueError('Trees have incompatible leaf order')

    it1, c1, c2, it2 = interpolate_tree(tree1, tree2)

    jumping_taxa = algorithm_five(it1, it2, tree1._order)
    return jumping_taxa
