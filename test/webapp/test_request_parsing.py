import io
import importlib.util
import sys
from pathlib import Path

import pytest
from werkzeug.datastructures import FileStorage, MultiDict
from werkzeug.test import EnvironBuilder
from werkzeug.wrappers import Request

HELPERS_PATH = Path(__file__).resolve().parents[2] / "webapp" / "routes" / "helpers.py"
spec = importlib.util.spec_from_file_location("request_helpers", HELPERS_PATH)
assert spec and spec.loader
request_helpers = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = request_helpers
spec.loader.exec_module(request_helpers)
parse_tree_data_request = request_helpers.parse_tree_data_request


def _request_with_msa(form: dict[str, str]) -> Request:
    builder = EnvironBuilder(
        method="POST",
        data=MultiDict(
            [
                (
                    "msaFile",
                    FileStorage(
                        filename="alignment.fasta",
                        stream=io.BytesIO(b">A\nACGT\n>B\nACGA\n>C\nACGG\n"),
                    ),
                ),
                *form.items(),
            ]
        ),
    )
    return Request(builder.get_environ())


def test_parse_tree_data_request_preserves_tree_engine_and_iqtree_fast_search() -> None:
    request = _request_with_msa(
        {
            "treeInferenceEngine": "iqtree",
            "iqtreeFastSearch": "on",
        }
    )

    parsed = parse_tree_data_request(request)

    assert parsed.tree_inference_engine == "iqtree"
    assert parsed.iqtree_fast_search is True


def test_parse_tree_data_request_accepts_fasttree_engine() -> None:
    request = _request_with_msa(
        {
            "treeInferenceEngine": "fasttree",
            "iqtreeFastSearch": "",
        }
    )

    parsed = parse_tree_data_request(request)

    assert parsed.tree_inference_engine == "fasttree"
    assert parsed.iqtree_fast_search is False


def test_parse_tree_data_request_rejects_out_of_range_iqtree_replicates() -> None:
    request = _request_with_msa(
        {
            "iqtreeSupportMode": "ufboot",
            "iqtreeUfbootReplicates": "999",
        }
    )

    with pytest.raises(
        ValueError,
        match="iqtreeUfbootReplicates must be between 1000 and 100000",
    ):
        parse_tree_data_request(request)


def test_parse_tree_data_request_allows_sh_alrt_replicates_below_ufboot_minimum() -> None:
    request = _request_with_msa(
        {
            "iqtreeSupportMode": "sh_alrt",
            "iqtreeUfbootReplicates": "1000",
            "iqtreeShAlrtReplicates": "100",
        }
    )

    parsed = parse_tree_data_request(request)

    assert parsed.iqtree_sh_alrt_replicates == 100


def test_parse_tree_data_request_rejects_non_integer_iqtree_replicates() -> None:
    request = _request_with_msa(
        {
            "iqtreeSupportMode": "sh_alrt",
            "iqtreeShAlrtReplicates": "many",
        }
    )

    with pytest.raises(ValueError, match="iqtreeShAlrtReplicates must be an integer"):
        parse_tree_data_request(request)
