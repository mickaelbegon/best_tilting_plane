"""Small debounce helper used by the GUI to avoid recomputing on every slider tick."""

from __future__ import annotations

from typing import Callable, Protocol


class AfterScheduler(Protocol):
    """Minimal protocol implemented by Tk widgets exposing `after` scheduling."""

    def after(self, delay_ms: int, callback: Callable[[], None]) -> object:
        """Schedule a callback and return an opaque handle."""

    def after_cancel(self, handle: object) -> None:
        """Cancel a previously scheduled callback."""


class DebouncedRunner:
    """Run a callback only after changes have been quiet for a short delay."""

    def __init__(
        self, scheduler: AfterScheduler, callback: Callable[[], None], *, delay_ms: int = 250
    ) -> None:
        """Store the scheduler, callback, and debounce delay."""

        self._scheduler = scheduler
        self._callback = callback
        self._delay_ms = delay_ms
        self._pending_handle: object | None = None

    def schedule(self) -> None:
        """Schedule the callback and cancel any older pending run."""

        if self._pending_handle is not None:
            self._scheduler.after_cancel(self._pending_handle)
        self._pending_handle = self._scheduler.after(self._delay_ms, self._run)

    def cancel(self) -> None:
        """Cancel the pending callback, if any."""

        if self._pending_handle is not None:
            self._scheduler.after_cancel(self._pending_handle)
            self._pending_handle = None

    def _run(self) -> None:
        """Run the debounced callback and clear the pending handle."""

        self._pending_handle = None
        self._callback()
