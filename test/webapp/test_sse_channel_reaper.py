import time

from webapp.services.sse.channels import ChannelRegistry
from webapp.services.sse.reaper import ChannelReaper


def test_reaper_removes_closed_channel_nobody_ever_streamed() -> None:
    """
    Reproduces the leak this reaper exists to fix: a channel whose background
    job finished (closing it) but whose client never opened
    /stream/progress/<channel_id> at all, so nothing ever called
    registry.remove() for it.
    """
    registry = ChannelRegistry()
    channel = registry.create()
    channel.complete({"ok": True})
    assert registry.count() == 1

    reaper = ChannelReaper(registry, interval_seconds=0.02)
    reaper.start()
    try:
        deadline = time.monotonic() + 2.0
        while registry.count() != 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert registry.get(channel.channel_id) is None
        assert registry.count() == 0
    finally:
        reaper.stop()


def test_reaper_leaves_unclosed_channels_alone() -> None:
    """A long-running job's still-open channel must never be swept."""
    registry = ChannelRegistry()
    channel = registry.create()  # never completed/closed

    reaper = ChannelReaper(registry, interval_seconds=0.02)
    reaper.start()
    try:
        time.sleep(0.1)
        assert registry.get(channel.channel_id) is channel
    finally:
        reaper.stop()


def test_reaper_start_is_idempotent() -> None:
    registry = ChannelRegistry()
    reaper = ChannelReaper(registry, interval_seconds=0.02)
    reaper.start()
    first_thread = reaper._thread
    reaper.start()
    assert reaper._thread is first_thread
    reaper.stop()
