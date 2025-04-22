import argparse
import shutil
from pathlib import Path
from typing import Optional

def normalize_name(fname: str) -> str:
    # Remove a prefix of "reverse_" or "reversed_" (case-insensitive)
    lower = fname.lower()
    if lower.startswith("reverse_"):
        fname = fname[len("reverse_"):]
    elif lower.startswith("reversed_"):
        fname = fname[len("reversed_"):]
    # Remove a suffix of "_reverse"
    if fname.lower().endswith("_reverse"):
        fname = fname[:-len("_reverse")]
    return fname

def group_json_files_in_directory(directory_path: str) -> None:
    """
    Group JSON files that have the same base name.
    A file can be "reversed" if its filename either begins with 'reverse_' (or 'reversed_')
    or ends with '_reverse'. Both files with and without the marker are grouped together.
    """
    base_dir = Path(directory_path)
    if not base_dir.is_dir():
        print(f"Directory does not exist: {directory_path}")
        return

    files_by_base: dict[str, dict[str, Optional[Path]]] = {}  # Added type annotation
    for f in base_dir.glob("*.json"):
        norm = normalize_name(f.stem)
        files_by_base.setdefault(norm, {"normal": None, "reverse": None})
        # Mark file as reversed if its stem starts with the marker or ends with _reverse
        if f.stem.lower().startswith("reverse_") or f.stem.lower().startswith("reversed_") or f.stem.lower().endswith("_reverse"):
            files_by_base[norm]["reverse"] = f
        else:
            files_by_base[norm]["normal"] = f

    # For each base, create a folder only if a normal file exists;
    # if only a reversed file exists, leave it in place.
    for base, file_map in files_by_base.items():
        if file_map["normal"]:  # Only create folder when a normal file exists.
            subfolder = base_dir / base
            subfolder.mkdir(exist_ok=True)
            shutil.move(str(file_map["normal"]), str(subfolder / file_map["normal"].name))
            if file_map["reverse"]:
                shutil.move(str(file_map["reverse"]), str(subfolder / file_map["reverse"].name))
        # Otherwise, do nothing for reversed-only files.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Group JSON files by base name if a reverse file exists."
    )
    parser.add_argument(
        "directory",
        metavar="DIRECTORY_PATH",
        help="Path to the directory containing JSON files."
    )
    args = parser.parse_args()
    group_json_files_in_directory(args.directory)
