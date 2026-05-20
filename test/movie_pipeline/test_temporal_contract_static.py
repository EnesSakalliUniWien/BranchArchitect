from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_temporal_contract_pipeline_does_not_export_legacy_intermediate_types():
    legacy_type_names = [
        "".join(["Distance", "Metrics"]),
        "".join(["Tree", "Pair", "Solution"]),
        "".join(["Pair", "Interpolation", "Context"]),
    ]
    legacy_file_names = [
        "_".join(["distance", "metrics"]) + ".py",
        "_".join(["tree", "pair", "solution"]) + ".py",
    ]
    legacy_builder_name = "_".join(["build", "pair", "solutions"])
    duplicate_bridge_name = "_".join(["build", "pair", "contexts"])
    checked_paths = [
        REPO_ROOT / "brancharchitect" / "movie_pipeline",
        REPO_ROOT / "brancharchitect" / "tree_interpolation" / "types",
    ]

    offenders = []
    for checked_path in checked_paths:
        for path in checked_path.rglob("*.py"):
            source = path.read_text(encoding="utf8")
            for legacy_name in legacy_type_names:
                if legacy_name in source:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}: {legacy_name}")
            if legacy_builder_name in source:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {legacy_builder_name}")
            if duplicate_bridge_name in source:
                offenders.append(
                    f"{path.relative_to(REPO_ROOT)}: {duplicate_bridge_name}"
                )

    for checked_path in checked_paths:
        for legacy_file_name in legacy_file_names:
            if list(checked_path.rglob(legacy_file_name)):
                offenders.append(
                    f"{checked_path.relative_to(REPO_ROOT)}: {legacy_file_name}"
                )

    assert offenders == []
