# BranchArchitect: Tree Transformation Toolkit

BranchArchitect is the backend engine that powers the Phylo-Movies experience. It studies transformations between phylogenetic trees on the same taxa set by decomposing each tree into splits (bipartitions induced by single edges) and detecting conflicts whenever a split appears in one tree but not the other. Instead of resolving splits independently, the method focuses on pivotal subtrees whose rearrangement eliminates several conflicts at once, producing a finite sequence of intermediate trees that share the same labeled leaves and valid bipartitions.

**Interpolation checklist (conceptual):**
- Identify the common taxa set and the splits shared by both trees.
- Flag conflicting splits and group them by the pivotal subtrees they influence.
- Select a depth order so rewiring proceeds from leaf-near edits toward the root.
- Rewire each pivotal subtree to resolve several conflicts at once.
- After every edit, verify that the tree remains connected, acyclic, and free of duplicate leaves; collapse zero-length edges before continuing.

The interpolation procedure starts from splits common to both trees. For every shared branch, we compare the groupings of its immediate children across the two trees, identify mismatched child pairs that cover the same leaves, and choose the rewiring that restores consistency. After each edit we validate that the structure is still connected, acyclic, and free of duplicate leaves, collapsing any zero-length edges that appear. Because the edits are applied from the tips inward, this depth-aware rewiring keeps every intermediate topology valid while moving smoothly from the source to the target tree.

This description mirrors the actual `TreeInterpolationPipeline` implementation (`brancharchitect/movie_pipeline/tree_interpolation_pipeline.py`). The pipeline roots the trees (when configured), optimizes their leaf order, discovers active-changing splits via the lattice algorithm (`iterate_lattice_algorithm`), orders those splits from leaves to root (`edge_sorting_utils.sort_edges_by_depth`), and builds the five-step microsequences defined in `tree_interpolation/subtree_paths`. Each microstep collapses zero-length branches only when they are absent from the destination topology (`consensus_tree.create_collapsed_consensus_tree`) and reorders taxa before grafting the next subtree, so every intermediate tree produced by the pipeline satisfies the invariants stated above.

To inspect these interpolations through the browser-based Phylo-Movies interface, run `./start_movie_server.sh` from the repository root (after `chmod +x start_movie_server.sh` if needed). The script installs dependencies via Poetry, sets the required environment variables, and launches the Flask server on `http://127.0.0.1:5002/`.

## Movie Server Overview

BranchArchitect bundles a Flask-based “movie server” that surfaces these results through the Phylo-Movies web client. The backend streams tree frames produced by the Python pipelines and exposes REST endpoints consumed by a lightweight front end served from the same application. Out of the box the server provides:

- Interactive playback of interpolated trees, with controls to scrub through anchor and intermediate states.
- Linked charts (e.g., Robinson–Foulds distances, branch-length summaries) that stay synchronized with the current frame.
- Optional multiple-sequence alignment views that highlight the taxa selected in the tree.
- Recording and export helpers that capture SVG snapshots or WebM videos for presentations.

The server runs entirely with the Python dependencies defined in this repository—no separate Node.js build is required. Use `start_movie_server.sh` (described below) for the managed startup sequence or follow the manual instructions if you need to customize ports, logging, or environment variables.

---

## Installation

BranchArchitect is a Python-first project that targets Python 3.11+ and uses Poetry for dependency management. The library ships scientific-computing dependencies (NumPy, SciPy, scikit-bio) plus visualization tooling (Matplotlib, Plotly, Toytree) needed by both the CLI pipelines and the optional Flask movie server.

### Installation Research & Validation Checklist
- Review `pyproject.toml`/`poetry.lock` to capture the supported Python version, dependency groups, and optional extras before documenting commands.
- Inspect helper tooling (`start_movie_server.sh`, `webapp/pyproject.toml`) so that any referenced environment setup matches the actual scripts committed to the repo.
- Confirm distributable artifacts under `dist/` to justify pip-based installs alongside Poetry workflows.
- Cross-check verification commands (`poetry run python …`, `poetry run pytest`) with the enforced settings in `pyproject.toml`/`pytest.ini` to keep instructions aligned with repository guidelines.

### Prerequisites

**System Requirements:**
- Python 3.11 or higher
- Poetry (Python dependency manager)
- Cairo library (optional, only needed for advanced SVG rendering features)

**Install Poetry:**

```bash
# Linux, macOS, Windows (WSL)
curl -sSL https://install.python-poetry.org | python3 -

# Add to PATH (if needed)
export PATH="$HOME/.local/bin:$PATH"

# Verify installation
poetry --version
```

**Install Cairo (Optional - for SVG rendering):**

Cairo is only required if you plan to use advanced SVG export features in the plotting modules. The web application and core functionality work without it.

```bash
# macOS (Homebrew)
brew install cairo

# Ubuntu/Debian
sudo apt-get install libcairo2-dev

# Fedora/RHEL
sudo dnf install cairo-devel

# Windows
# Download from: https://www.cairographics.org/download/
```

### Installation Methods

#### Method 1: Development Installation with Poetry (Recommended for Contributors)

**Purpose:** Provision the full development toolchain (tests, type checks, movie pipeline helpers) exactly as defined in `pyproject.toml`.

**Minimal requirements:** Python 3.11+, Poetry 1.2+ on your `PATH`, and Cairo libraries if you plan to use the optional plotting extras.

**When to use:** Active feature work, debugging, or running the complete pytest+mypy suite locally.

**Steps:**

```bash
git clone https://github.com/EnesSakalliUniWien/BranchArchitect.git
cd BranchArchitect

# Install core + dev/test deps (add --extras plotting if you need Cairo-backed SVG exports)
poetry install

# Activate the Poetry-managed virtualenv (optional)
poetry shell

# Validate the install and test suite
python -c "import brancharchitect; print('BranchArchitect ready')"
poetry run pytest
```

#### Method 2: Runtime Installation with Poetry (For Analysts/Users)

**Purpose:** Install only the production dependency group for running BranchArchitect as a library or CLI tool.

**Minimal requirements:** Python 3.11+, Poetry 1.2+, Cairo only if you enable the plotting extras.

**When to use:** Consuming BranchArchitect inside analysis notebooks, scripts, or pipelines without the development overhead.

**Steps:**

```bash
git clone https://github.com/EnesSakalliUniWien/BranchArchitect.git
cd BranchArchitect

# Install the runtime dependency set
poetry install --only main

# Run your workflow inside the Poetry environment
poetry run python run_pipeline.py  # swap in your own entry point as needed
```

#### Method 3: Editable Installation with pip (Virtualenv or Conda)

**Purpose:** Use pip’s editable mode while still leveraging the Poetry build backend for teams standardized on pip/venv or Conda tooling.

**Minimal requirements:** Python 3.11+, pip 23+, and an activated virtual environment (created via `python -m venv`, Conda, etc.); Cairo remains optional for the plotting extras.

**When to use:** Integrating BranchArchitect into an existing pip-based mono-repo, IDE, or workflow where Poetry is unavailable.

**Steps:**

```bash
git clone https://github.com/EnesSakalliUniWien/BranchArchitect.git
cd BranchArchitect

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

pip install --upgrade pip
pip install -e .[plotting]  # drop [plotting] if Cairo/pycairo are unnecessary

python -c "import brancharchitect; print('BranchArchitect ready')"
```

### Installation Validation Notes
- `pyproject.toml` declares Python `^3.11`, dependency groups (`main`, `dev`, `test`), and the `plotting` extra; each method above maps directly onto those definitions.
- Wheels and source distributions in `dist/brancharchitect-0.64.0*` verify that pip-based installs (`pip install -e .` or from wheel) use the officially published artifacts.
- The verification commands (`python -c …`, `poetry run pytest`) mirror the enforced `pytest --mypy` settings from `pyproject.toml`, ensuring the documented checks align with the repository’s CI expectations.

### Running the Web Application

BranchArchitect includes a Flask-based web interface for interactive tree visualization and movie generation.

#### Quick Start (Using the Startup Script)

**Purpose:** Automated server startup with proper environment configuration.
**When to use:** Primary method for running the web interface locally.

```bash
# Ensure you're in the project root
cd BranchArchitect

# Make the script executable (first time only)
chmod +x start_movie_server.sh

# Start the server
./start_movie_server.sh
```

**What the script does:**
1. Checks for and kills any processes on port 5002
2. Installs webapp dependencies via Poetry
3. Sets up required environment variable (`PYTHONPATH`)
4. Starts the Flask backend on `http://127.0.0.1:5002/`
5. Monitors the server health and creates logs (`backend.log`, `backend_startup.log`)
6. Handles graceful shutdown on Ctrl+C

**Accessing the application:**

Once the script reports `[backend] Backend is ready!`, open your browser to:
- **Main interface:** http://127.0.0.1:5002/
- **Health check:** http://127.0.0.1:5002/about

**Stopping the server:**

Press `Ctrl+C` in the terminal running the script. The cleanup function will automatically terminate the backend process.

#### Manual Web Application Setup

**Purpose:** Manual control over server configuration and troubleshooting.
**When to use:** Custom deployment, debugging, or when the startup script fails.

**Step 1: Install dependencies**

```bash
# Install BranchArchitect core library (if not already installed)
cd BranchArchitect
poetry install

# Install webapp-specific dependencies
cd webapp
poetry install
cd ..
```

**Step 2: Set environment variables**

```bash
# Set PYTHONPATH to include project root
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Optional: Set library path for Cairo (only if you installed the 'plotting' extras)
# export DYLD_LIBRARY_PATH="/opt/homebrew/lib:${DYLD_LIBRARY_PATH}"

# Optional: Enable Flask debug mode
export FLASK_ENV=development
```

**Step 3: Start the Flask server**

```bash
# From project root
poetry run python webapp/run.py --host=127.0.0.1 --port=5002

# Or from webapp directory
cd webapp
poetry run python run.py --host=127.0.0.1 --port=5002
```

**Step 4: Access the application**

Open http://127.0.0.1:5002/ in your browser.

#### Web Application Architecture

The web application consists of:
- **Backend:** Flask server (`webapp/run.py`) providing REST API
- **Core library:** BranchArchitect modules imported via `PYTHONPATH`
- **Dependencies:** Separate Poetry configuration in `webapp/pyproject.toml`

**Port configuration:**
- Default: `5002`
- To use a different port, modify the `--port` argument or update `start_movie_server.sh`

**Log files:**
- `backend.log`: Server runtime logs and error messages
- `backend_startup.log`: Startup diagnostics and environment configuration

### Verification

After installation, verify BranchArchitect is working correctly:

```bash
# Check package installation
poetry run python -c "import brancharchitect; from brancharchitect.tree import Node; print('Core module loaded')"

# Check critical dependencies
poetry run python -c "import numpy, scipy, skbio, matplotlib; print('Scientific dependencies loaded')"

# Run a simple test
poetry run python -c "
from brancharchitect.parser.newick_parser import parse_newick
tree = parse_newick('(A:1,B:1)C:0;')
print(f'Parsed tree with {len(tree.get_leaves())} leaves')
"

# Run the test suite
poetry run pytest -v
```

### Troubleshooting

#### Cairo library not found

**Symptom:**
```
ImportError: cairo library not found
```

**Solution:**
1. Install Cairo system library (see Prerequisites - optional)
2. Install plotting extras: `poetry install --extras plotting`
3. On macOS, set library path: `export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"`

#### Poetry command not found

**Solution:**
```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Add to PATH
export PATH="$HOME/.local/bin:$PATH"

# Add to shell profile for persistence
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc  # or ~/.bashrc
```

#### Python version mismatch

**Symptom:**
```
The currently activated Python version X.X.X is not supported by the project (^3.11)
```

**Solution:**
```bash
# Install Python 3.11+ (if needed)
# macOS: brew install python@3.11
# Ubuntu: sudo apt-get install python3.11

# Configure Poetry to use correct version
poetry env use python3.11
poetry install
```

#### Web server won't start

**Symptoms:**
- Server doesn't respond
- Port already in use error
- Backend process dies immediately

**Solutions:**

1. **Check port availability:**
   ```bash
   lsof -i :5002
   # If port is in use, kill the process or choose a different port
   ```

2. **Review logs:**
   ```bash
   cat backend.log
   cat backend_startup.log
   ```

3. **Verify environment:**
   ```bash
   echo $PYTHONPATH  # Should include project root
   # DYLD_LIBRARY_PATH only needed if using plotting extras with Cairo
   ```

4. **Manual dependency check:**
   ```bash
   cd webapp
   poetry install
   poetry run python -c "import flask; print('Flask OK')"
   ```

5. **Test without the script:**
   ```bash
   export PYTHONPATH="${PWD}:${PYTHONPATH}"
   cd webapp
   poetry run python run.py --host=127.0.0.1 --port=5002
   ```

#### Permission denied on startup script

**Symptom:**
```
-bash: ./start_movie_server.sh: Permission denied
```

**Solution:**
```bash
chmod +x start_movie_server.sh
```

#### Import errors in webapp

**Symptom:**
```
ModuleNotFoundError: No module named 'brancharchitect'
```

**Solution:**
Ensure `PYTHONPATH` includes the project root:
```bash
export PYTHONPATH="/path/to/BranchArchitect:${PYTHONPATH}"
```

### Configuration

BranchArchitect uses `pyproject.toml` for configuration. Key settings:

- **Python version:** `^3.11` (minimum)
- **Package version:** `0.64.0`
- **Type checking:** MyPy enabled with strict settings
- **Testing:** pytest with 10-second default timeout

### Development Workflow

```bash
# Install with dev dependencies
poetry install

# Run tests
poetry run pytest

# Run tests with type checking
poetry run pytest --mypy

# Run specific test file
poetry run pytest test/test_tree.py -v

# Run tests with coverage
poetry run pytest --cov=brancharchitect

# Type check without running tests
poetry run mypy brancharchitect/

# Build distribution
poetry build

# View built packages
ls -l dist/
```

### Environment Variables

**Core library:**
- None required for basic usage

**Web application:**
- `PYTHONPATH`: **Required.** Must include BranchArchitect project root
- `DYLD_LIBRARY_PATH` (macOS): **Optional.** Only needed if using plotting extras with Cairo (typically `/opt/homebrew/lib`)
- `FLASK_ENV`: Optional. Set to `development` for debug mode and auto-reload
- `FLASK_DEBUG`: Optional. Set to `1` for enhanced debugging

**Example configuration:**
```bash
export PYTHONPATH="/Users/username/BranchArchitect:${PYTHONPATH}"
# Only set this if you installed plotting extras:
# export DYLD_LIBRARY_PATH="/opt/homebrew/lib:${DYLD_LIBRARY_PATH}"
export FLASK_ENV=development
```

### System Dependencies Summary

| Dependency   | Purpose               | Required | Installation                                                           |
| ------------ | --------------------- | -------- | ---------------------------------------------------------------------- |
| Python 3.11+ | Runtime environment   | Yes      | [python.org](https://www.python.org/downloads/)                        |
| Poetry 1.2+  | Dependency management | Yes      | `curl -sSL https://install.python-poetry.org \| python3 -`             |
| Cairo        | SVG rendering         | Optional | `brew install cairo` (macOS), `apt-get install libcairo2-dev` (Ubuntu) |
| curl         | Health checks         | Yes      | Pre-installed on most systems                                          |
| lsof         | Port checking         | Yes      | Pre-installed on macOS/Linux                                           |

### Project Structure

```
BranchArchitect/
├── brancharchitect/          # Core library package
│   ├── tree.py               # Tree data structures
│   ├── parser/               # Newick parser
│   ├── consensus/            # Consensus algorithms
│   ├── jumping_taxa/         # Lattice algorithms
│   └── ...
├── webapp/                   # Flask web application
│   ├── run.py               # Server entry point
│   ├── routes/              # API endpoints
│   ├── pyproject.toml       # Webapp dependencies
│   └── ...
├── test/                     # Test suite
├── notebooks/                # Jupyter notebooks
├── docs/                     # Documentation
├── pyproject.toml           # Main project configuration
├── poetry.lock              # Locked dependencies
├── start_movie_server.sh    # Server startup script
└── README.md                # This file
```

### Next Steps

After installation:

1. **Explore examples:** Review test files in `test/` for usage patterns
2. **Read documentation:** Check `docs/` for algorithm details
3. **Try notebooks:** Explore interactive examples in `notebooks/`
4. **Run the pipeline:** Execute `poetry run python run_pipeline.py`
5. **Start the webapp:** Use `./start_movie_server.sh` for visual exploration

### Support

For questions, issues, or contributions:

- **Issues:** [GitHub Issues](https://github.com/EnesSakalliUniWien/BranchArchitect/issues)
- **Repository:** [github.com/EnesSakalliUniWien/BranchArchitect](https://github.com/EnesSakalliUniWien/BranchArchitect)
- **Documentation:** See `docs/` directory in the repository

### License

See the LICENSE file in the repository for licensing information.
