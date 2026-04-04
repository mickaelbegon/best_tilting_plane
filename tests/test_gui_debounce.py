"""Tests for the GUI debounce helper."""

from best_tilting_plane.gui.debounce import DebouncedRunner


class FakeScheduler:
    """Very small scheduler stub exposing the `after` Tk-like API."""

    def __init__(self) -> None:
        self._next_handle = 0
        self.pending: dict[int, object] = {}
        self.cancelled: list[int] = []

    def after(self, _delay_ms: int, callback):
        self._next_handle += 1
        self.pending[self._next_handle] = callback
        return self._next_handle

    def after_cancel(self, handle) -> None:
        self.cancelled.append(handle)
        self.pending.pop(handle, None)

    def run(self, handle: int) -> None:
        callback = self.pending.pop(handle)
        callback()


def test_debounced_runner_cancels_previous_pending_run() -> None:
    """Only the latest scheduled callback should survive repeated scheduling."""

    scheduler = FakeScheduler()
    calls = []
    runner = DebouncedRunner(scheduler, lambda: calls.append("run"), delay_ms=250)

    runner.schedule()
    first_handle = next(iter(scheduler.pending))
    runner.schedule()
    second_handle = next(iter(scheduler.pending))

    assert first_handle in scheduler.cancelled
    assert second_handle != first_handle

    scheduler.run(second_handle)

    assert calls == ["run"]


def test_debounced_runner_cancel_clears_pending_callback() -> None:
    """Cancelling should prevent the pending callback from being executed."""

    scheduler = FakeScheduler()
    calls = []
    runner = DebouncedRunner(scheduler, lambda: calls.append("run"), delay_ms=250)

    runner.schedule()
    pending_handle = next(iter(scheduler.pending))
    runner.cancel()

    assert pending_handle in scheduler.cancelled
    assert calls == []
