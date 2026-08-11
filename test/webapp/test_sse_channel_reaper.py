import time

from webapp.services.sse.channels import ChannelRegistry
from webapp.services.sse.reaper import DEFAULT_RETAIN_CLOSED_SECONDS, ChannelReaper


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

    reaper = ChannelReaper(registry, interval_seconds=0.02, retain_closed_seconds=0.0)
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


def test_reaper_keeps_a_just_completed_channel_the_client_has_not_reached_yet() -> None:
    """
    The race this retention window exists to close: processing can finish
    before the client's EventSource connects. Sweeping that channel on age
    alone throws away the completion event and the client gets a 404.
    """
    registry = ChannelRegistry()
    channel = registry.create()
    channel.complete({"movie": "ready"})

    reaper = ChannelReaper(registry, interval_seconds=0.02)
    reaper.start()
    try:
        time.sleep(0.1)
        assert registry.get(channel.channel_id) is channel
    finally:
        reaper.stop()


def test_cleanup_closed_evicts_only_channels_older_than_the_retention_window() -> None:
    registry = ChannelRegistry()
    channel = registry.create()
    channel.complete({"ok": True})

    assert registry.cleanup_closed(min_closed_age_seconds=60.0) == 0
    assert registry.get(channel.channel_id) is channel

    assert registry.cleanup_closed(min_closed_age_seconds=0.0) == 1
    assert registry.get(channel.channel_id) is None


def test_completed_channel_still_replays_its_result_to_a_late_client() -> None:
    """A client that connects after completion must still receive the payload."""
    registry = ChannelRegistry()
    channel = registry.create()
    channel.complete({"movie": "ready"})

    registry.cleanup_closed(min_closed_age_seconds=DEFAULT_RETAIN_CLOSED_SECONDS)

    late_channel = registry.get(channel.channel_id)
    assert late_channel is not None

    replayed = "".join(late_channel.stream(timeout=0.1))
    assert "event: complete" in replayed
    assert '{"movie":"ready"}' in replayed


def test_reaper_start_is_idempotent() -> None:
    registry = ChannelRegistry()
    reaper = ChannelReaper(registry, interval_seconds=0.02)
    reaper.start()
    first_thread = reaper._thread
    reaper.start()
    assert reaper._thread is first_thread
    reaper.stop()
