# Profiling fixtures

These files are checked in so profiling runs do not depend on temporary output.

## `norovirus_window_2500_step_1500_fasttree.newick`

- Source alignment: `test/data/current_testfiles/norovirus_full_genome_490taxa_aligned.fasta`
- Window size: `2500`
- Step size: `1500`
- Tree inference: `FastTreeConfig(use_gtr=True, use_gamma=True, no_ml=True)`
- Contents: 7 newline-delimited Newick trees, 490 taxa per tree

Profile this fixture with:

```bash
poetry run python scripts/profile_interpolation.py --fixture norovirus --max-pairs 1
```
