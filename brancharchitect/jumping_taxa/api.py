from brancharchitect.jumping_taxa.tree_interpolation import interpolate_tree
from brancharchitect.jumping_taxa.algorithm_five import algorithm_five
from brancharchitect.tree import Node

def call_jumping_taxa(tree1 : Node, tree2: Node):
    if tree1._order != tree2._order:
        raise ValueError('Trees have incompatible leaf order')

    it1, c1, c2, it2 = interpolate_tree(tree1, tree2)

    jumping_taxa = algorithm_five(it1, it2, tree1._order)
    jumping_taxa = [tuple(tree1._order[i] for i in idx) for idx in jumping_taxa]
    return jumping_taxa
