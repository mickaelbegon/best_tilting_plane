"""Tests for GUI animation playback controls."""

from __future__ import annotations

import numpy as np

from best_tilting_plane.gui.app import BestTiltingPlaneApp


class FakeScheduler:
    """Very small scheduler stub exposing the `after` Tk-like API."""

    def __init__(self) -> None:
        self._next_handle = 0
        self.pending: dict[int, object] = {}
        self.cancelled: list[int] = []

    def after(self, _delay_ms: int, callback):
        """Schedule a callback and return a handle."""

        self._next_handle += 1
        self.pending[self._next_handle] = callback
        return self._next_handle

    def after_cancel(self, handle) -> None:
        """Cancel a pending callback."""

        self.cancelled.append(handle)
        self.pending.pop(handle, None)


class FakeVar:
    """Tiny variable stub exposing `get` and `set`."""

    def __init__(self, value) -> None:
        self._value = value

    def get(self):
        """Return the stored value."""

        return self._value

    def set(self, value) -> None:
        """Update the stored value."""

        self._value = value


class FakeScale:
    """Minimal scale stub storing the latest configuration."""

    def __init__(self) -> None:
        self.options: dict[str, float] = {}

    def configure(self, **kwargs) -> None:
        """Record the latest configuration values."""

        self.options.update(kwargs)


def _build_app_for_animation() -> tuple[BestTiltingPlaneApp, list[int], FakeScheduler]:
    """Create a minimal app instance for animation-control tests."""

    scheduler = FakeScheduler()
    drawn_frames: list[int] = []
    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app.root = scheduler
    app.play_pause_label = FakeVar("Play")
    app.time_slider_var = FakeVar(0.0)
    app.time_value_var = FakeVar("0.00 s")
    app.time_slider = FakeScale()
    app._animation_after_id = None
    app._animation_frame_index = 0
    app._animation_playing = False
    app._time_slider_updating = False
    app._visualization_data = {
        "result": type("Result", (), {"time": np.array([0.0, 0.5, 1.0], dtype=float)})()
    }
    app._draw_animation_frame = drawn_frames.append
    return app, drawn_frames, scheduler


def test_start_animation_loop_draws_current_frame_and_schedules_next() -> None:
    """Starting playback should draw immediately, sync the slider, and schedule a callback."""

    app, drawn_frames, scheduler = _build_app_for_animation()

    app._start_animation_loop()

    assert drawn_frames == [0]
    assert app._animation_playing
    assert app.play_pause_label.get() == "Pause"
    assert app.time_slider_var.get() == 0.0
    assert app.time_value_var.get() == "0.00 s"
    assert app._animation_frame_index == 1
    assert app._animation_after_id in scheduler.pending


def test_time_slider_change_pauses_and_jumps_to_nearest_frame() -> None:
    """Moving the time slider should pause playback and display the nearest frame."""

    app, drawn_frames, scheduler = _build_app_for_animation()
    app._start_animation_loop()
    scheduled_handle = app._animation_after_id

    app._on_time_slider_change("0.74")

    assert scheduled_handle in scheduler.cancelled
    assert not app._animation_playing
    assert app.play_pause_label.get() == "Play"
    assert drawn_frames[-1] == 1
    assert app._animation_frame_index == 1
    assert app.time_slider_var.get() == 0.5
    assert app.time_value_var.get() == "0.50 s"


def test_configure_time_slider_uses_simulation_time_bounds() -> None:
    """The time slider range should match the current simulation span."""

    app, _drawn_frames, _scheduler = _build_app_for_animation()
    app._animation_frame_index = 2

    app._configure_time_slider()

    assert app.time_slider.options == {"from_": 0.0, "to": 1.0}
    assert app.time_slider_var.get() == 1.0
    assert app.time_value_var.get() == "1.00 s"
