# BranchArchitect: Tree Transformation Toolkit

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.64.0-orange.svg)](https://github.com/EnesSakalliUniWien/BranchArchitect)

BranchArchitect is a phylogenetic tree transformation engine that computes SPR (Subtree Prune and Regraft) paths between trees. It powers the Phylo-Movies web visualization by identifying which subtrees move between positions and animating smooth transitions.

### How It Works

1. **Parse & Root** — Load Newick trees and apply midpoint rooting
2. **Lattice Solving** — Identify conflicting splits and compute the minimal set of "jumping taxa" (movers) needed to transform one tree into another
3. **Leaf Ordering** — Rotate subtrees to minimize visual crossings during animation
4. **4-Phase Interpolation** — Build intermediate frames via:
   - **Collapse**: Shrink source-unique branches to zero
   - **Reorder**: Move subtrees to destination positions  
   - **Expand**: Grow destination-unique branches from zero
   - **Snap**: Apply final branch lengths

## Citation

If you use BranchArchitect in your research, please cite the software using the metadata in [CITATION.cff](CITATION.cff).

## Examples

### MSA → Trees (Sliding Window)

```python
from msa_to_trees import run_pipeline, FastTreeConfig

# Generate trees from MSA using sliding windows + FastTree
result = run_pipeline(
    input_file="alignment.fasta",
    output_directory="./output",
    window_size=1000,
    step_size=250,
    fasttree_config=FastTreeConfig(use_gtr=True)
)

# Output: ./output/combined_trees.newick (one tree per line)
print(f"Trees written to: {result.tree_file}")
```

### Trees → Interpolation

```python
from brancharchitect.io import read_newick
from brancharchitect.movie_pipeline import TreeInterpolationPipeline, PipelineConfig

# Load trees from Newick file
trees = read_newick("trees.newick", treat_zero_as_epsilon=True)

# Run interpolation pipeline
config = PipelineConfig(enable_rooting=True)
pipeline = TreeInterpolationPipeline(config=config)
result = pipeline.process_trees(trees)

# Print interpolated frames as Newick
for i, tree in enumerate(result.interpolated_trees):
    print(f"Frame {i}: {tree.to_newick()}")
```

### Full Pipeline (MSA → Trees → Interpolation)

```bash
python examples/msa_to_movie.py alignment.fasta
```

## Installation

**Requirements:** Python 3.11+, Poetry 1.2+

```bash
git clone https://github.com/EnesSakalliUniWien/BranchArchitect.git
cd BranchArchitect
poetry install
```

## Web Application

Start the Phylo-Movies backend server:

```bash
./start_movie_server.sh
# Backend API at http://127.0.0.1:5002/
```

The Phylo-Movies frontend connects to this backend for tree interpolation and visualization.

**Manual setup:**
```bash
export PYTHONPATH="${PWD}:${PYTHONPATH}"
poetry run python webapp/run.py --port=5002
```

## Development

```bash
poetry install                           # Install all dependencies
poetry run pytest                        # Run tests (includes mypy)
poetry run pytest --cov=brancharchitect  # With coverage
poetry run python run_pipeline.py        # Demo pipeline
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Poetry command not found` | `curl -sSL https://install.python-poetry.org \| python3 -` |
| `Python version X.X.X not supported` | `poetry env use python3.11` |
| Port 5002 in use | `lsof -i :5002` then kill the process |
| `No module named 'brancharchitect'` | `export PYTHONPATH="${PWD}:${PYTHONPATH}"` |

## Project Structure

```
BranchArchitect/
├── brancharchitect/        # Core library (tree.py, parser/, jumping_taxa/)
├── msa_to_trees/           # MSA → Trees pipeline (sliding window + FastTree)
├── webapp/                 # Flask web application
├── examples/               # Example scripts (msa_to_movie.py)
├── test/                   # Test suite
├── notebooks/              # Jupyter notebooks
└── start_movie_server.sh   # Launch web app (http://127.0.0.1:5002/)
```

## Support

- [GitHub Issues](https://github.com/EnesSakalliUniWien/BranchArchitect/issues)
- See `docs/` for algorithm details

## License

MIT — see [LICENSE](LICENSE)
