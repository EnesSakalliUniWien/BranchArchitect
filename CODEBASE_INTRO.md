# BranchArchitect Codebase Introduction

## 1. Getting Started
- **Install dependencies** with Poetry (Python 3.11+): `poetry install`
- **Run quick demo**: `poetry run python run_pipeline.py` (uses `test-data/current_testfiles/small_example.newick`)
- **Launch server**: `chmod +x start_movie_server.sh && ./start_movie_server.sh` (serves at http://127.0.0.1:5002)
- **Run tests**: `poetry run pytest` (includes mypy checks)

## 2. Environment Overview
- **Dependencies** (`pyproject.toml`): Python ^3.11, NumPy/SciPy/pandas, tskit/msprime, Flask, joblib.
- **Runtime**: `start_movie_server.sh` manages the environment, setting `PYTHONPATH` and logging to `logs/backend.log`.

## 3. Input Handling
Rules for `read_newick` to avoid common errors:
- **Format**: Well-formed Newick with terminal semicolons. Parentheses must balance.
- **Taxa**: Stick to ASCII. Quote special characters (`'Taxon,1'`).
- **Consistency**: All trees in a batch **must** share the exact same taxa (names and count).
- **Loading**: Use `read_newick("batch.newick", force_list=True, treat_zero_as_epsilon=True)`.

## 4. Architecture Map
- **Parsing/IO**: `brancharchitect/io.py`, `brancharchitect/parser/newick_parser.py` -> Produces `Node` trees.
- **Tree Model**: `brancharchitect/tree.py` -> The core object model.
- **Core Elements**: `brancharchitect/elements/` -> `Partition` and `PartitionSet` primitives.
- **Pipeline Driver**: `brancharchitect/movie_pipeline/` -> Orchestrates the full process (Rooting -> Order -> Interpolate).
- **Interpolation Logic**: `brancharchitect/tree_interpolation/` -> The "Movie" generation logic.
- **Lattice Solvers**: `brancharchitect/jumping_taxa/lattice/` -> Solves the "Moving Taxa" problem.
- **Web Backend**: `webapp/` -> Flask app wrapping the library.

## 5. Core Data Structures
### Partition & PartitionSet
The fundamental atoms of the system (in `brancharchitect/elements/`):
- **`Partition`**: Represents a Split/Clade using a **bitmask**. Fast, hashable, and supports set algebra.
- **`PartitionSet`**: A collection of partitions (e.g., a whole tree's topology). Optimized for union/intersection/difference and subset queries.

### Encoding
The "Rosetta Stone" for bitmasks:
- Maps `Taxon Name <-> Integer Index`.
- stored on `Node.taxa_encoding`.
- **Critical Constraint**: `Partition` and `PartitionSet` operations generally require identical encodings.

## 6. The Interpolation Pipeline

The system morphs Tree A into Tree B via a "Geodesic Path" in tree space. The process follows a strict dependency chain:

### 1. Analysis & Pre-computation

- **Rooting**: Trees are midpoint-rooted (`tree_rooting.py`) to establish a consistent Up/Down orientation.
- **Lattice Solver**: The core engine (`lattice/`) solves the geometric path between trees *before* any morphing happens. It identifies:
  - **Anchors**: Stable subtrees that exist in both trees.
  - **Movers**: Taxa that "jump" locations.

### 2. Visual Optimization (Static Layout)

- **Leaf Ordering**: `TreeOrderOptimizer` uses the pre-computed **Anchor/Mover** (Lattice) info to rotate leaves.
  - Anchors are placed in the center.
  - Movers are placed on the edges (Left/Right bands) to prevent "hairball" crossings.

### 3. Execution (The Dynamic Morph)

With the trees aligned, `pivot_sequence_orchestrator` plans the move:
- **Filtering**:
  - **Common Splits**: Preservation (Mean Averaging).
  - **Unique Splits**: Target for topological change.
- **5-Step Surgery** (`step_executor.py`):
  1.  **Zeroing**: Source-unique branches shrink to 0.
  2.  **Collapse**: Nodes removed (transient polytomy).
  3.  **Reorder**: Logical move of the subtree.
  4.  **Graft**: New nodes inserted at destination.
  5.  **Snap/Expand**: New branches grow.

## 7. Existing Tests
- **IO**: `test/io/`, `test/tree/`
- **Core Elements**: `test/element/`
- **Lattice logic**: `test/lattice/`, `test/jumping_taxa/`
- **Interpolation**: `test/tree_interpolation/`
- **Integration**: `tests/` (e.g., `test_patristic_compensation.py`, `test_rooting/`)
- **Distances**: `test/distances/`
