# Bundled Tree-Inference Binaries

This directory contains third-party command-line tools used by the
BranchArchitect backend when Phylo-Movies infers trees from alignments.

## Runtime Policy

BranchArchitect should use these binaries for reproducible source and packaged
runs unless a developer explicitly sets an override:

1. `IQTREE_PATH` or `FASTTREE_PATH`, when set.
2. Bundled binary under `bin/<platform>/`.
3. System `PATH` fallback only when the bundled binary is missing.

## Bundled Tools

| Tool | Bundled command | Version currently bundled | Upstream | License |
| --- | --- | --- | --- | --- |
| IQ-TREE | `iqtree3` / `iqtree3.exe` | 3.1.1 | <https://github.com/iqtree/iqtree3> | GPL-2.0 |
| FastTree | `fasttree` / `fasttree.exe` | 2.2.0 source build | <https://github.com/morgannprice/fasttree> | GPL-3.0 |

License texts are stored in `bin/licenses/`.

## Platform Coverage

| Platform directory | Included files |
| --- | --- |
| `darwin/` | `iqtree3`, `fasttree` |
| `linux/` | `iqtree3`, `iqtree3_intel`, `iqtree3_arm`, `fasttree` |
| `win32/` | `iqtree3.exe`, `libiomp5md.dll`, `fasttree.exe` |

## Checksums

Regenerate these values after replacing any binary:

```bash
for f in $(find engine/BranchArchitect/bin -maxdepth 2 -type f | sort); do
  shasum -a 256 "$f"
done
```

Current SHA-256 values:

```text
65b894a28c5fee05f327b43ac958d43095d7cac450651e01154a2701c1a594fb  bin/darwin/fasttree
0e9ca422e17338554b6d7c537bb6b70fd674a915634206ea496c3829058dca93  bin/darwin/iqtree3
7b348493c779a385a95a54ff49d5057ffe23517aac781afaa3a9ced3cdc4f181  bin/linux/fasttree
4ba12225d8f52a727b5e6fa172d300cd9302510730ea763795b2bd7a6ed4a834  bin/linux/iqtree3
7ec9c737898d2efb29903fbb0818beed313069d811c8c9a972610cd748a37f49  bin/linux/iqtree3_arm
aeef0a1baf3b3cf24d8175b7ea864c76c5eabbcde4e2491cde6b6a93f5dacbe0  bin/linux/iqtree3_intel
5bd2c3a7c2bf06c89d85f42ceef6982001b06dcb71bfc7810ef6ecc6be3e3abc  bin/win32/fasttree.exe
01447a010dd7ff0330715ac3f8988bbb7ea1f1ded7b02b95af100d5d2be98289  bin/win32/iqtree3.exe
946269ecbcce6950571f1c2405044810dfb8983fff35c633705adf799dfb6cb8  bin/win32/libiomp5md.dll
```

## Update Procedure

1. Download or build replacement binaries from the upstream project.
2. Replace only the files for the intended platform directory.
3. Verify the local version:

   ```bash
   engine/BranchArchitect/bin/darwin/iqtree3 --version
   engine/BranchArchitect/bin/darwin/fasttree -expert 2>&1 | head -1
   ```

4. Refresh license files from upstream if the upstream license changed.
5. Update the bundled version, platform table, and checksums in this file.
6. Run the backend IQ-TREE/FastTree smoke tests and the Electron build script.
