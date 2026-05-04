from pathlib import Path
import importlib.util


def _load_profile_module():
    script_path = Path("scripts/profile_interpolation.py")
    spec = importlib.util.spec_from_file_location("profile_interpolation", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_iter_pairs_honors_max_pairs():
    profile_interpolation = _load_profile_module()
    trees = [object(), object(), object(), object()]

    pairs = list(profile_interpolation._iter_pairs(trees, max_pairs=2))

    assert [(index, source, destination) for index, source, destination in pairs] == [
        (0, trees[0], trees[1]),
        (1, trees[1], trees[2]),
    ]
