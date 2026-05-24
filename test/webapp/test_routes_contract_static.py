from pathlib import Path

from flask import Flask

from webapp.routes.routes import bp


ROUTES_SOURCE = Path(__file__).resolve().parents[2] / "webapp" / "routes" / "routes.py"


def test_backend_does_not_expose_legacy_synchronous_treedata_route() -> None:
    source = ROUTES_SOURCE.read_text(encoding="utf-8")

    assert '@bp.route("/treedata", methods=["POST"])' not in source
    assert "def treedata(" not in source
    assert '@bp.route("/treedata/stream", methods=["POST"])' in source


def test_backend_root_route_serves_direct_visit_landing_page() -> None:
    app = Flask(__name__)
    app.register_blueprint(bp)

    response = app.test_client().get("/")

    assert response.status_code == 200
    assert response.mimetype == "text/html"
    assert b"Phylo-Movies Backend" in response.data
    assert b"/about" in response.data
