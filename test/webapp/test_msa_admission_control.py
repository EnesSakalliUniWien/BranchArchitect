from __future__ import annotations

from io import BytesIO
import threading
import time
from typing import Any

from webapp import create_app
from webapp.routes import routes


def _msa_upload() -> dict[str, Any]:
    return {
        "msaFile": (
            BytesIO(b">A\nACGTACGT\n>B\nACGTACGT\n"),
            "example.fasta",
        ),
        "windowSize": "4",
        "windowStepSize": "4",
    }


def _wait_until(predicate: Any, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_msa_tree_inference_rejects_concurrent_jobs(monkeypatch: Any) -> None:
    app = create_app()
    started = threading.Event()
    release = threading.Event()

    def slow_msa_analysis(*_args: Any, **_kwargs: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        started.set()
        assert release.wait(timeout=2.0)
        return {"file_name": "example.nwk"}, []

    monkeypatch.setattr(routes, "_run_msa_analysis_and_interpolate", slow_msa_analysis)

    with app.test_client() as client:
        first = client.post(
            "/treedata/stream",
            data=_msa_upload(),
            content_type="multipart/form-data",
        )
        assert first.status_code == 200
        assert started.wait(timeout=2.0)

        health = client.get("/health").get_json()
        assert health["jobs"]["msa_tree_inference"]["busy"] is True

        second = client.post(
            "/treedata/stream",
            data=_msa_upload(),
            content_type="multipart/form-data",
        )
        assert second.status_code == 409
        assert "already running" in second.get_json()["error"]

        release.set()
        assert _wait_until(
            lambda: client.get("/health").get_json()["jobs"]["msa_tree_inference"]["busy"]
            is False
        )


def test_tree_file_upload_is_not_blocked_by_msa_tree_inference(monkeypatch: Any) -> None:
    app = create_app()
    started = threading.Event()
    release = threading.Event()

    def slow_msa_analysis(*_args: Any, **_kwargs: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        started.set()
        assert release.wait(timeout=2.0)
        return {"file_name": "example.nwk"}, []

    monkeypatch.setattr(routes, "_run_msa_analysis_and_interpolate", slow_msa_analysis)

    with app.test_client() as client:
        first = client.post(
            "/treedata/stream",
            data=_msa_upload(),
            content_type="multipart/form-data",
        )
        assert first.status_code == 200
        assert started.wait(timeout=2.0)

        tree_response = client.post(
            "/treedata/stream",
            data={
                "treeFile": (
                    BytesIO(b"((A:1,B:1):1,C:1);((A:1,C:1):1,B:1);"),
                    "trees.nwk",
                )
            },
            content_type="multipart/form-data",
        )
        assert tree_response.status_code == 200
        assert "channel_id" in tree_response.get_json()

        release.set()
        assert _wait_until(
            lambda: client.get("/health").get_json()["jobs"]["msa_tree_inference"]["busy"]
            is False
        )
