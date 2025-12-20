# Split Alignment Module

A modularized tool for splitting multiple sequence alignments into sub-alignments.

## Module Structure

The code has been refactored from a single 551-line script into a well-organized module:

```
split_alignment/
├── __init__.py          # Public API exports
├── __main__.py          # CLI entry point
├── constants.py         # Configuration constants
├── models.py            # Data classes (WindowInfo, SequenceType)
├── io_utils.py          # Alignment I/O operations
├── windowing.py         # Window generation and sliding window logic
├── sequence_analysis.py # Sequence type detection and filtering
└── validators.py        # Argument parsing validators
```

## Usage

### As a Python module:

```python
from split_alignment import (
    load_alignment,
    detect_sequence_type,
    create_windows_from_parameters,
    apply_sliding_window,
)

# Load alignment
alignment, fmt = load_alignment("input.fasta")

# Detect sequence type
seq_type = detect_sequence_type(alignment)

# Create windows
windows = create_windows_from_parameters(
    window_size=300,
    step_size=25,
    alignment_length=alignment.get_alignment_length()
)

# Process windows
for sub_alignment, window in apply_sliding_window(alignment, windows, seq_type):
    # Process each sub-alignment
    pass
```

### From command line:

```bash
# Run as module
python -m scripts.split_alignment -i input.fasta -o output_dir/

# Or use the wrapper script
python scripts/split_alignment_cli.py -i input.fasta -o output_dir/

# With custom windows from CSV
python -m scripts.split_alignment -i input.fasta -o output_dir/ --split-file windows.csv

# Generate log files
python -m scripts.split_alignment -i input.fasta -o output_dir/ --log
```

## Improvements from Refactoring

### Better Organization
- **Separation of concerns**: Each module handles a specific responsibility
- **Testability**: Individual functions can be tested in isolation
- **Maintainability**: Changes to one area don't affect unrelated code

### Improved Naming
- `WindowInfo.start_pos` instead of `WindowInfo.strtpos`
- `WindowInfo.reverse_complement` instead of `WindowInfo.rev_comp`
- `create_windows_from_parameters()` instead of `get_windows_from_parameters()`
- `detect_sequence_type()` instead of `check_sequence_type()`
- `PositiveIntegerAction` instead of `GreaterEqualOne`

### Better Documentation
- Comprehensive docstrings for all modules and functions
- Type hints throughout the codebase
- Clear parameter and return value descriptions

### Enhanced CLI
- Organized argument groups (windowing, filtering, output options)
- Better help text and descriptions
- Progress output during processing
- Summary statistics at completion

## Migration Notes

The refactored code maintains **100% functional compatibility** with the original script.
All features work identically, including:

- Sliding window generation
- Custom window ranges from CSV
- Sequence type detection (nucleotide/amino acid/other)
- Reverse complement handling
- Ambiguous sequence filtering
- Log file generation

## File Naming Conventions

All module and function names follow Python conventions:
- `snake_case` for functions, variables, and module names
- `PascalCase` for classes
- Descriptive names that clearly indicate purpose
- No abbreviations unless widely recognized
