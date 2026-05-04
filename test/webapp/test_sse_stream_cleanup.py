from flask import Flask

from webapp.routes import routes
from webapp.services.sse.channels import ChannelRegistry


def test_stream_progress_removes_channel_after_stream_completion(monkeypatch):
    registry = ChannelRegistry()
    channel = registry.create()
    channel.complete({"ok": True})
    monkeypatch.setattr(routes, "channels", registry)

    app = Flask(__name__)
    with app.test_request_context(f"/stream/progress/{channel.channel_id}"):
        response = routes.stream_progress(channel.channel_id)
        list(response.response)

    assert registry.get(channel.channel_id) is None
