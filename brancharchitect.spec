# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for BranchArchitect Flask server.

This bundles the webapp with all its dependencies into a standalone executable.
"""
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

def is_runtime_module(module_name):
    """Keep PyInstaller away from package test suites and local benchmarks."""
    excluded_parts = {"test", "tests", "testing", "benchmark", "benchmarks"}
    return not any(part in excluded_parts for part in module_name.split("."))


all_datas = []
all_binaries = []
all_hiddenimports = []

for pkg in [
    "brancharchitect",
    "webapp",
    "msa_to_trees",
    "split_alignment",
    "Bio.AlignIO",
]:
    all_hiddenimports.extend(collect_submodules(pkg, filter=is_runtime_module))

# Additional hidden imports that might be missed
additional_hiddenimports = [
    'flask',
    'flask.json',
    'flask.cli',
    'flask_cors',
    'flask_compress',
    'flask_compress.flask_compress',
    'werkzeug',
    'werkzeug.serving',
    'jinja2',
    'markupsafe',
    'itsdangerous',
    'click',
    'blinker',
    'brotli',
    'Bio',
    'Bio.Align',
    'Bio.AlignIO',
    'Bio.Phylo.BaseTree',
    'Bio.Phylo.Newick',
    'Bio.Phylo.NewickIO',
    'Bio.SeqIO.FastaIO',
    'Bio.Seq',
    'Bio.SeqRecord',
    'numpy',
    'joblib',
    'orjson',
    'waitress',
    # Multiprocessing support for PyInstaller
    'multiprocessing',
    'multiprocessing.pool',
    'multiprocessing.process',
    'multiprocessing.spawn',
    'multiprocessing.popen_spawn_posix',
    'multiprocessing.popen_fork',
]

a = Analysis(
    ['webapp/run.py'],
    pathex=['.'],
    binaries=all_binaries,
    datas=[
        ('webapp', 'webapp'),
        ('brancharchitect', 'brancharchitect'),
        ('msa_to_trees/msa_to_trees', 'msa_to_trees'),
        ('msa_to_trees/split_alignment', 'split_alignment'),
        ('bin', 'bin'),
    ] + all_datas,
    hiddenimports=all_hiddenimports + additional_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude large unnecessary packages
        'matplotlib',
        'plotly',
        'seaborn',
        'IPython',
        'ipywidgets',
        'notebook',
        'jupyter',
        'pytest',
        'tkinter',
        'torch',
        'tqdm',
        'Bio.PDB',
        'Bio.PDB.mmtf',
        'brancharchitect.leaforder.benchmark',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='brancharchitect-server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='brancharchitect-server',
)
