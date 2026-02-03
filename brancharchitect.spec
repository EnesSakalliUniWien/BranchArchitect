# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for BranchArchitect Flask server.

This bundles the webapp with all its dependencies into a standalone executable.
"""
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

# Collect all required packages
packages_to_collect = [
    'flask',
    'flask_cors',
    'flask_compress',
    'werkzeug',
    'jinja2',
    'click',
    'blinker',
    'itsdangerous',
    'markupsafe',
    'brotli',
    'Bio',
    'numpy',
    'scipy',
    'pandas',
    'joblib',
    'skbio',
    'typing_extensions',
    'waitress',
]

all_datas = []
all_binaries = []
all_hiddenimports = []

for pkg in packages_to_collect:
    try:
        datas, binaries, hiddenimports = collect_all(pkg)
        all_datas.extend(datas)
        all_binaries.extend(binaries)
        all_hiddenimports.extend(hiddenimports)
    except Exception as e:
        print(f"Warning: Could not collect {pkg}: {e}")

# Additional hidden imports that might be missed
additional_hiddenimports = [
    'flask',
    'flask.json',
    'flask.cli',
    'flask_cors',
    'flask_compress',
    'werkzeug',
    'werkzeug.serving',
    'werkzeug.debug',
    'jinja2',
    'markupsafe',
    'itsdangerous',
    'click',
    'blinker',
    'brotli',
    'Bio',
    'Bio.Phylo',
    'Bio.Phylo.Newick',
    'Bio.Phylo.NewickIO',
    'Bio.SeqIO',
    'Bio.Seq',
    'Bio.SeqRecord',
    'numpy',
    'scipy',
    'scipy.sparse',
    'pandas',
    'joblib',
    'skbio',
    'skbio.io',
    'skbio.io.format',
    'skbio.tree',
    'tqdm',
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
        ('msa_to_trees', 'msa_to_trees'),
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
