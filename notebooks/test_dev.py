from brancharchitect.io import parse_newick
from brancharchitect.jumping_taxa.lattice.lattice_solver import lattice_algorithm

t1, t2 = parse_newick(
    (
        "((((X,A1),A2),(A3,A4)),((B1,B2),(B3,B4)));"
        + "(X,(((A1,A2),(A3,A4)),((B1,B2),(B3,B4))));"
    )
)
print(t1.to_newick())
print(t2.to_newick())
lattice_algorithm(
    t1, t2, t1._order
)  # , lattice_algorithm="lattice_algorithm", lattice_solver="lattice_solver"
print("Lattice algorithm completed.")
print("Lattice algorithm completed.")