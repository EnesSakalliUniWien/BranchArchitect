# BranchArchitect

# usecase

## Parse Newick into Python Tree

```{python}
from brancharchitect.newick_parser import parse_newick

with open('newick_file.nwk') as f:
    tree = parse_newick(f.read())
```
