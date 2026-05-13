from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "msa_to_trees"))

from msa_to_trees import pipeline


def test_iqtree_config_builds_dna_model_and_output_prefix() -> None:
    IQTreeConfig = pipeline.IQTreeConfig
    config = IQTreeConfig(use_gtr=True, use_gamma=True, threads=1)

    args = config.build_command_args("window.fasta", Path("/tmp/window_0"))

    assert args == [
        "-s",
        "window.fasta",
        "-st",
        "DNA",
        "-m",
        "GTR+G",
        "-nt",
        "1",
        "-fast",
        "-quiet",
        "-redo",
        "-pre",
        "/tmp/window_0",
    ]


def test_iqtree_config_can_disable_fast_search() -> None:
    IQTreeConfig = pipeline.IQTreeConfig
    config = IQTreeConfig(fast_search=False)

    args = config.build_command_args("window.fasta", Path("/tmp/window_0"))

    assert "-fast" not in args


def test_iqtree_config_is_exported_from_package() -> None:
    from msa_to_trees import IQTreeConfig

    assert IQTreeConfig is pipeline.IQTreeConfig


def test_run_iqtree_reads_treefile_from_iqtree_prefix(
    tmp_path: Path, monkeypatch
) -> None:
    IQTreeConfig = pipeline.IQTreeConfig
    alignment = tmp_path / "0.fasta"
    alignment.write_text(">A\nACGT\n>B\nACGA\n>C\nACGG\n", encoding="utf-8")

    def fake_run(cmd, check, capture_output, text, env):
        prefix = Path(cmd[cmd.index("-pre") + 1])
        Path(f"{prefix}.treefile").write_text("(A:0.1,B:0.2,C:0.3);\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setenv("IQTREE_PATH", "iqtree-test")
    monkeypatch.setattr(subprocess, "run", fake_run)

    tree = pipeline.run_iqtree(str(alignment), IQTreeConfig())

    assert tree == "(A:0.1,B:0.2,C:0.3);\n"


def test_iqtree_discovery_prefers_iqtree3_binary(monkeypatch) -> None:
    monkeypatch.delenv("IQTREE_PATH", raising=False)
    monkeypatch.setattr(
        pipeline.shutil,
        "which",
        lambda name: "/usr/local/bin/iqtree3" if name == "iqtree3" else None,
    )

    assert pipeline._get_iqtree_exe() == "/usr/local/bin/iqtree3"


def test_iqtree_discovery_accepts_iqtree2_binary(monkeypatch) -> None:
    monkeypatch.delenv("IQTREE_PATH", raising=False)
    monkeypatch.setattr(
        pipeline.shutil,
        "which",
        lambda name: "/usr/local/bin/iqtree2" if name == "iqtree2" else None,
    )

    assert pipeline._get_iqtree_exe() == "/usr/local/bin/iqtree2"


def test_iqtree_discovery_prefers_bundled_binary_when_frozen(
    tmp_path: Path, monkeypatch
) -> None:
    bundled = tmp_path / "bin" / "darwin" / "iqtree3"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.delenv("IQTREE_PATH", raising=False)
    monkeypatch.setattr(pipeline.sys, "frozen", True, raising=False)
    monkeypatch.setattr(pipeline.sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.setattr(pipeline.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: None)

    assert pipeline._get_iqtree_exe() == str(bundled)


def test_infer_trees_parallel_uses_iqtree_runner_for_iqtree_config(
    tmp_path: Path, monkeypatch
) -> None:
    IQTreeConfig = pipeline.IQTreeConfig
    windows_dir = tmp_path / "windows"
    trees_dir = tmp_path / "trees"
    windows_dir.mkdir()
    trees_dir.mkdir()
    (windows_dir / "0.fasta").write_text(">A\nACGT\n>B\nACGA\n>C\nACGG\n")
    (windows_dir / "1.fasta").write_text(">A\nCGTA\n>B\nCGAA\n>C\nCGGA\n")

    def fake_run_iqtree(alignment_file: str, config) -> str:
        return f"({Path(alignment_file).stem});\n"

    monkeypatch.setattr("msa_to_trees.pipeline.run_iqtree", fake_run_iqtree)
    monkeypatch.setattr("sys.frozen", True, raising=False)

    master = pipeline.infer_trees_parallel(
        windows_dir=windows_dir,
        trees_dir=trees_dir,
        output_tree_filename="all.newick",
        config=IQTreeConfig(),
    )

    assert master.read_text() == "(0);\n(1);\n"


def test_run_pipeline_defaults_to_iqtree_config(tmp_path: Path, monkeypatch) -> None:
    captured_config = None

    class FakeRecord:
        def __init__(self, record_id: str) -> None:
            self.id = record_id

    class FakeAlignment:
        def get_alignment_length(self) -> int:
            return 10

        def __iter__(self):
            return iter([FakeRecord("A"), FakeRecord("B"), FakeRecord("C")])

    def fake_infer_trees_parallel(windows_dir, trees_dir, output_tree_filename, config):
        nonlocal captured_config
        captured_config = config
        output = trees_dir / "combined.newick"
        output.write_text("(A,B,C);\n", encoding="utf-8")
        return output

    monkeypatch.setattr(
        pipeline, "load_alignment", lambda *args, **kwargs: FakeAlignment()
    )
    monkeypatch.setattr(pipeline, "scan_invalid_taxa", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        pipeline,
        "create_windows_from_parameters",
        lambda *args, **kwargs: [object()],
    )
    monkeypatch.setattr(
        pipeline,
        "generate_filtered_window_alignments",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(pipeline, "infer_trees_parallel", fake_infer_trees_parallel)

    pipeline.run_pipeline(
        input_file="alignment.fasta",
        output_directory=str(tmp_path),
        window_size=5,
        step_size=5,
    )

    assert isinstance(captured_config, pipeline.IQTreeConfig)
