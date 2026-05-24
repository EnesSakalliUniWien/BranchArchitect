from pathlib import Path

from msa_to_trees.pipeline import IQTreeConfig


def test_iqtree_config_adds_support_arguments_when_enabled():
    args = IQTreeConfig(
        fast_search=False,
        support_mode="sh_alrt_ufboot",
        ufboot_replicates=1000,
        sh_alrt_replicates=1000,
        bnni=True,
    ).build_command_args("window.fasta", Path("window"))

    assert "-B" in args
    assert args[args.index("-B") + 1] == "1000"
    assert "-alrt" in args
    assert args[args.index("-alrt") + 1] == "1000"
    assert "-bnni" in args


def test_iqtree_config_omits_fast_search_for_ufboot_support():
    args = IQTreeConfig(
        fast_search=True,
        support_mode="sh_alrt_ufboot",
    ).build_command_args("window.fasta", Path("window"))

    assert "-fast" not in args
    assert "-B" in args
    assert "-alrt" in args


def test_iqtree_config_keeps_fast_search_for_sh_alrt_only():
    args = IQTreeConfig(
        fast_search=True,
        support_mode="sh_alrt",
    ).build_command_args("window.fasta", Path("window"))

    assert "-fast" in args
    assert "-B" not in args
    assert "-alrt" in args


def test_iqtree_config_omits_support_arguments_by_default():
    args = IQTreeConfig().build_command_args("window.fasta", Path("window"))

    assert "-B" not in args
    assert "-alrt" not in args
    assert "-bnni" not in args
