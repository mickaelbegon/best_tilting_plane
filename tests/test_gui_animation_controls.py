"""Tests for GUI animation playback controls."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from best_tilting_plane.gui.app import (
    DEFAULT_CAMERA_AZIMUTH_DEG,
    DEFAULT_CAMERA_ELEVATION_DEG,
    ROOT_INITIAL_OPTIONS,
    TOP_VIEW_CAMERA_AZIMUTH_DEG,
    TOP_VIEW_CAMERA_ELEVATION_DEG,
    BestTiltingPlaneApp,
)


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

    def update_idletasks(self) -> None:
        """Mirror the Tk API used by the GUI during optimization."""


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


class FakeRunner:
    """Minimal debounced-runner stub exposing `cancel`."""

    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        """Record that pending automatic simulations were cancelled."""

        self.cancelled = True


class FakeAxis:
    """Minimal 3D axis stub exposing only `view_init` for camera tests."""

    def __init__(self) -> None:
        self.camera: tuple[float, float] | None = None

    def view_init(self, *, elev: float, azim: float) -> None:
        """Store the requested camera orientation."""

        self.camera = (elev, azim)


def _build_app_for_animation() -> tuple[BestTiltingPlaneApp, list[int], FakeScheduler]:
    """Create a minimal app instance for animation-control tests."""

    scheduler = FakeScheduler()
    drawn_frames: list[int] = []
    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app.root = scheduler
    app.play_pause_label = FakeVar("Play")
    app.plot_mode_var = FakeVar("Courbe")
    app.root_initial_mode = FakeVar(ROOT_INITIAL_OPTIONS[1])
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


def test_apply_camera_view_uses_top_view_when_root_is_zeroed() -> None:
    """The `q(root)=0` display mode should orient the camera on the `xOy` plane."""

    app, _drawn_frames, _scheduler = _build_app_for_animation()
    app._animation_axis = FakeAxis()
    app.root_initial_mode = FakeVar(ROOT_INITIAL_OPTIONS[0])

    app._apply_camera_view()

    assert app._animation_axis.camera == (
        TOP_VIEW_CAMERA_ELEVATION_DEG,
        TOP_VIEW_CAMERA_AZIMUTH_DEG,
    )


def test_apply_camera_view_uses_default_perspective_otherwise() -> None:
    """The standard display mode should keep the default 3D perspective."""

    app, _drawn_frames, _scheduler = _build_app_for_animation()
    app._animation_axis = FakeAxis()

    app._apply_camera_view()

    assert app._animation_axis.camera == (
        DEFAULT_CAMERA_ELEVATION_DEG,
        DEFAULT_CAMERA_AZIMUTH_DEG,
    )


def test_apply_optimized_values_updates_sliders_and_reruns_simulation() -> None:
    """Applying an optimized strategy should refresh the controls, then rerun the simulation."""

    scheduler = FakeScheduler()
    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app.root = scheduler
    calls: list[tuple[str, object]] = []
    app._set_values = lambda values: calls.append(("set", dict(values)))
    app._run_simulation = lambda: calls.append(("run", None))

    app._apply_optimized_values({"right_arm_start": 0.25})

    assert calls == [("set", {"right_arm_start": 0.25}), ("run", None)]


def test_optimize_strategy_applies_optimized_values_and_reruns_animation(monkeypatch) -> None:
    """The optimization button should push the optimum to the sliders before rerunning the simulation."""

    class FakeOptimizer:
        def optimize_right_arm_start_only(self, initial_right_arm_start, **_kwargs):
            assert initial_right_arm_start == 0.1
            return type(
                "OptimizationResult",
                (),
                {
                    "variables": type(
                        "Variables",
                        (),
                        {
                            "right_arm_start": 0.35,
                            "left_plane_initial": 0.0,
                            "left_plane_final": 0.0,
                            "right_plane_initial": 0.0,
                            "right_plane_final": 0.0,
                        },
                    )(),
                },
            )()

    monkeypatch.setattr(
        "best_tilting_plane.gui.app.TwistStrategyOptimizer.from_builder",
        lambda *_args, **_kwargs: FakeOptimizer(),
    )

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app.root = FakeScheduler()
    app.result_var = FakeVar("")
    app.optimization_mode_var = FakeVar("Optimize 2D")
    app._auto_runner = FakeRunner()
    app._current_values = lambda: {
        "right_arm_start": 0.1,
        "left_plane_initial": 0.0,
        "left_plane_final": 0.0,
        "right_plane_initial": 0.0,
        "right_plane_final": 0.0,
    }
    app._model_path = lambda: Path("/tmp/reduced.bioMod")
    applied: list[dict[str, float]] = []
    app._apply_optimized_values = lambda values: applied.append(dict(values))

    app._optimize_strategy()

    assert app._auto_runner.cancelled
    assert applied == [
        {
            "right_arm_start": 0.35,
            "left_plane_initial": 0.0,
            "left_plane_final": 0.0,
            "right_plane_initial": 0.0,
            "right_plane_final": 0.0,
        }
    ]
