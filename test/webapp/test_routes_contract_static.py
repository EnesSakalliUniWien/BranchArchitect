from pathlib import Path


ROUTES_SOURCE = Path(__file__).resolve().parents[2] / "webapp" / "routes" / "routes.py"


def test_backend_does_not_expose_legacy_synchronous_treedata_route() -> None:
    source = ROUTES_SOURCE.read_text(encoding="utf-8")

    assert '@bp.route("/treedata", methods=["POST"])' not in source
    assert "def treedata(" not in source
    assert '@bp.route("/treedata/stream", methods=["POST"])' in source
