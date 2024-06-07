from brancharchitect.tree_interpolation import interpolate_tree
from brancharchitect.algorithm_five import algorithm_five

def call_jumping_taxa(tree1, tree2):
    it1, c1, c2, it2 = interpolate_tree(tree1, tree2)

    print(it1.to_newick())
    print(it2.to_newick())
    jumping_taxa = algorithm_five(it1, it2, tree1._order)
    return jumping_taxa
