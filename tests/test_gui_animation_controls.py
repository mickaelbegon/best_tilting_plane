"""Tests for GUI animation playback controls."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tkinter as tk

import numpy as np

from best_tilting_plane.gui.app import (
    ANIMATION_MODE_OPTIONS,
    ANIMATION_REFERENCE_OPTIONS,
    DEFAULT_CAMERA_AZIMUTH_DEG,
    DEFAULT_CAMERA_ELEVATION_DEG,
    ALL_OPTIMIZATION_MODE_OPTIONS,
    OPTIMIZATION_MODE_ARM1_2D,
    OPTIMIZATION_MODE_ARM1_3D,
    OPTIMIZATION_MODE_ARM2_2D,
    OPTIMIZATION_MODE_ARM2_3D,
    OPTIMIZATION_MODE_BOTH_3D,
    OPTIMIZATION_MODE_OPTIONS,
    ROOT_VIEW_CAMERA_AZIMUTH_DEG,
    ROOT_VIEW_CAMERA_ELEVATION_DEG,
    ROOT_INITIAL_OPTIONS,
    SKELETON_CONNECTIONS,
    BestTiltingPlaneApp,
)
from best_tilting_plane.simulation import AerialSimulationResult, SimulationConfiguration


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

    def after_idle(self, callback):
        """Schedule one idle callback with the same semantics as `after(0, ...)`."""

        return self.after(0, callback)

    def after_cancel(self, handle) -> None:
        """Cancel a pending callback."""

        self.cancelled.append(handle)
        self.pending.pop(handle, None)

    def update_idletasks(self) -> None:
        """Mirror the Tk API used by the GUI during optimization."""

    def protocol(self, *_args) -> None:
        """Mirror the Tk API used to register the close callback."""

    def destroy(self) -> None:
        """Mirror the Tk API used when closing the window."""

    def run_pending(self) -> None:
        """Execute and clear the currently pending callbacks."""

        pending = list(self.pending.items())
        self.pending.clear()
        for _handle, callback in pending:
            callback()


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


class FakeCombobox:
    """Small combobox stub storing the last configured values."""

    def __init__(self) -> None:
        self.options: dict[str, object] = {}

    def configure(self, **kwargs) -> None:
        """Record the latest combobox configuration."""

        self.options.update(kwargs)


class FakeProgressbar:
    """Small progressbar stub storing its latest configuration."""

    def __init__(self) -> None:
        self.options: dict[str, float] = {}

    def configure(self, **kwargs) -> None:
        """Record the latest progressbar values."""

        self.options.update(kwargs)


class FakeWindow:
    """Very small window stub that records whether it was destroyed."""

    def __init__(self) -> None:
        self.destroyed = False

    def destroy(self) -> None:
        """Mirror one popup close."""

        self.destroyed = True


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


class Fake3DLine:
    """Minimal 3D line artist stub storing the latest coordinates."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.x = []
        self.y = []
        self.z = []

    def set_data(self, x, y) -> None:
        """Mirror matplotlib line updates."""

        self.x = list(x)
        self.y = list(y)

    def set_3d_properties(self, z) -> None:
        """Mirror matplotlib line updates."""

        self.z = list(z)


class Fake3DAxis(FakeAxis):
    """Small 3D axis stub that records created lines and the title."""

    def __init__(self) -> None:
        super().__init__()
        self.plot_calls: list[Fake3DLine] = []
        self.title: str | None = None

    def clear(self) -> None:
        """Mirror matplotlib."""

    def set_xlabel(self, _value) -> None:
        """Mirror matplotlib."""

    def set_ylabel(self, _value) -> None:
        """Mirror matplotlib."""

    def set_zlabel(self, _value) -> None:
        """Mirror matplotlib."""

    def set_box_aspect(self, _value) -> None:
        """Mirror matplotlib."""

    def set_xlim(self, *_args) -> None:
        """Mirror matplotlib."""

    def set_ylim(self, *_args) -> None:
        """Mirror matplotlib."""

    def set_zlim(self, *_args) -> None:
        """Mirror matplotlib."""

    def plot(self, _x, _y, _z, **kwargs):
        """Record one 3D line creation."""

        line = Fake3DLine(**kwargs)
        self.plot_calls.append(line)
        return [line]

    def add_collection3d(self, _collection) -> None:
        """Mirror matplotlib."""

    def set_title(self, value: str) -> None:
        """Store the latest title."""

        self.title = value

    def legend(self, *_args, **_kwargs) -> None:
        """Mirror matplotlib."""

    legend_ = None


class FakePlotAxis:
    """Minimal 2D axis stub recording plotted lines and labels."""

    def __init__(self) -> None:
        self.plot_calls: list[dict[str, object]] = []
        self.axvline_calls: list[dict[str, object]] = []

    def clear(self) -> None:
        """Mirror matplotlib."""

    def plot(self, x, y, **kwargs):
        """Record a plotted line."""

        self.plot_calls.append({"x": list(x), "y": list(y), **kwargs})
        return [None]

    def step(self, x, y, **kwargs):
        """Record a plotted stepped line."""

        self.plot_calls.append({"x": list(x), "y": list(y), "step": True, **kwargs})
        return [None]

    def set_xlabel(self, _value) -> None:
        """Mirror matplotlib."""

    def set_ylabel(self, _value) -> None:
        """Mirror matplotlib."""

    def set_title(self, _value) -> None:
        """Mirror matplotlib."""

    def grid(self, *_args, **_kwargs) -> None:
        """Mirror matplotlib."""

    def legend(self, *_args, **_kwargs) -> None:
        """Mirror matplotlib."""

    def axhline(self, *_args, **_kwargs) -> None:
        """Mirror matplotlib."""

    def axvline(self, x, **kwargs):
        """Record the time-indicator line."""

        payload = {"x": float(x), "kwargs": dict(kwargs), "xdata": [float(x), float(x)]}
        self.axvline_calls.append(payload)

        class FakeLine:
            def __init__(self, data) -> None:
                self.data = data

            def set_xdata(self, values) -> None:
                self.data["xdata"] = list(values)

        return FakeLine(payload)

    def set_aspect(self, *_args, **_kwargs) -> None:
        """Mirror matplotlib."""


def _build_app_for_animation() -> tuple[BestTiltingPlaneApp, list[int], FakeScheduler]:
    """Create a minimal app instance for animation-control tests."""

    scheduler = FakeScheduler()
    drawn_frames: list[int] = []
    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app.root = scheduler
    app.play_pause_label = FakeVar("Play")
    app.animation_mode_var = FakeVar(ANIMATION_MODE_OPTIONS[0])
    app.plot_mode_var = FakeVar("Courbe")
    app.root_initial_mode = FakeVar(ROOT_INITIAL_OPTIONS[1])
    app.time_slider_var = FakeVar(0.0)
    app.time_value_var = FakeVar("0.00 s")
    app.time_slider = FakeScale()
    app._animation_after_id = None
    app._animation_frame_index = 0
    app._animation_playing = False
    app._time_slider_updating = False
    app._is_closing = False
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


def test_stop_animation_loop_ignores_tcl_error_during_shutdown() -> None:
    """Cancelling one pending callback should stay silent if Tk is already shutting down."""

    app, _drawn_frames, _scheduler = _build_app_for_animation()

    class FailingScheduler(FakeScheduler):
        def after_cancel(self, handle) -> None:
            raise tk.TclError(f"invalid command name {handle!r}")

    app.root = FailingScheduler()
    app.play_pause_label = FakeVar("Pause")
    app._animation_after_id = "5158428800_animate_next_frame"
    app._animation_playing = True

    app._stop_animation_loop()

    assert app._animation_after_id is None
    assert not app._animation_playing
    assert app.play_pause_label.get() == "Play"


def test_on_close_stops_animation_before_destroying_root() -> None:
    """Closing the GUI should cancel the animation callback before destroying Tk."""

    app, _drawn_frames, _scheduler = _build_app_for_animation()
    destroyed: list[str] = []

    class ClosingScheduler(FakeScheduler):
        def destroy(self) -> None:
            destroyed.append("destroyed")

    app.root = ClosingScheduler()
    app.play_pause_label = FakeVar("Pause")
    app._animation_after_id = 1
    app._animation_playing = True

    app._on_close()

    assert app._is_closing
    assert app._animation_after_id is None
    assert not app._animation_playing
    assert destroyed == ["destroyed"]


def test_on_close_cancels_pending_matplotlib_idle_draws() -> None:
    """Closing the GUI should cancel pending Tk idle draws scheduled by Matplotlib canvases."""

    app, _drawn_frames, _scheduler = _build_app_for_animation()
    cancelled: list[object] = []

    class FakeTkCanvas:
        def after_cancel(self, handle) -> None:
            cancelled.append(handle)

    class FakeCanvas:
        def __init__(self, handle) -> None:
            self._idle_draw_id = handle
            self._tkcanvas = FakeTkCanvas()

    app._animation_canvas = FakeCanvas("6309343552idle_draw")
    app._plot_canvas = FakeCanvas("6309343553idle_draw")

    app._on_close()

    assert cancelled == ["6309343552idle_draw", "6309343553idle_draw"]
    assert app._animation_canvas._idle_draw_id is None
    assert app._plot_canvas._idle_draw_id is None


def test_on_close_closes_external_matplotlib_figures(monkeypatch) -> None:
    """Closing the GUI should also close any external matplotlib windows opened from the app."""

    app, _drawn_frames, _scheduler = _build_app_for_animation()
    closed: list[str] = []

    monkeypatch.setattr(
        "best_tilting_plane.gui.app.close_external_figures",
        lambda: closed.append("closed"),
    )

    app._on_close()

    assert closed == ["closed"]


def test_configure_time_slider_uses_simulation_time_bounds() -> None:
    """The time slider range should match the current simulation span."""

    app, _drawn_frames, _scheduler = _build_app_for_animation()
    app._animation_frame_index = 2

    app._configure_time_slider()

    assert app.time_slider.options == {"from_": 0.0, "to": 1.0}
    assert app.time_slider_var.get() == 1.0
    assert app.time_value_var.get() == "1.00 s"


def test_apply_camera_view_uses_root_side_view_when_root_is_zeroed() -> None:
    """The root display mode should orient the camera in the `xOz` plane."""

    app, _drawn_frames, _scheduler = _build_app_for_animation()
    app._animation_axis = FakeAxis()
    app.root_initial_mode = FakeVar(ROOT_INITIAL_OPTIONS[0])

    app._apply_camera_view()

    assert app._animation_axis.camera == (
        ROOT_VIEW_CAMERA_ELEVATION_DEG,
        ROOT_VIEW_CAMERA_AZIMUTH_DEG,
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


def test_apply_animation_reference_maps_popup_choices_to_internal_modes() -> None:
    """The single animation popup should drive the internal global/root/BTP display settings."""

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app.root_initial_mode = FakeVar(ROOT_INITIAL_OPTIONS[1])
    app.animation_mode_var = FakeVar(ANIMATION_MODE_OPTIONS[0])

    app._apply_animation_reference(ANIMATION_REFERENCE_OPTIONS[0])
    assert app.root_initial_mode.get() == ROOT_INITIAL_OPTIONS[1]
    assert app.animation_mode_var.get() == ANIMATION_MODE_OPTIONS[0]

    app._apply_animation_reference(ANIMATION_REFERENCE_OPTIONS[1])
    assert app.root_initial_mode.get() == ROOT_INITIAL_OPTIONS[0]
    assert app.animation_mode_var.get() == ANIMATION_MODE_OPTIONS[0]

    app._apply_animation_reference(ANIMATION_REFERENCE_OPTIONS[2])
    assert app.root_initial_mode.get() == ROOT_INITIAL_OPTIONS[0]
    assert app.animation_mode_var.get() == ANIMATION_MODE_OPTIONS[0]

    app._apply_animation_reference(ANIMATION_REFERENCE_OPTIONS[3])
    assert app.root_initial_mode.get() == ROOT_INITIAL_OPTIONS[1]
    assert app.animation_mode_var.get() == ANIMATION_MODE_OPTIONS[1]


def test_apply_camera_view_uses_root_side_view_for_root_hand_mode() -> None:
    """The root-hand mode should reuse the root side camera."""

    app, _drawn_frames, _scheduler = _build_app_for_animation()
    app._animation_axis = FakeAxis()
    app.animation_reference_var = FakeVar(ANIMATION_REFERENCE_OPTIONS[2])

    app._apply_camera_view()

    assert app._animation_axis.camera == (
        ROOT_VIEW_CAMERA_ELEVATION_DEG,
        ROOT_VIEW_CAMERA_AZIMUTH_DEG,
    )


def test_draw_animation_frame_dispatches_to_btp_animation_mode() -> None:
    """The animation panel should use the dedicated BTP renderer when that mode is selected."""

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app._animation_mode = lambda: ANIMATION_MODE_OPTIONS[1]
    app._visualization_data = {"result": object()}
    dispatched: list[int] = []
    app._draw_btp_animation_frame = dispatched.append

    app._draw_animation_frame(2)

    assert dispatched == [2]


def test_apply_optimized_values_updates_sliders_and_reruns_simulation() -> None:
    """Applying an optimized strategy should refresh the controls, then rerun the simulation."""

    scheduler = FakeScheduler()
    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app.root = scheduler
    app.result_var = FakeVar("Vrilles finales: 1.23 tours")
    calls: list[tuple[str, object]] = []
    app._set_values = lambda values: calls.append(("set", dict(values)))
    app._run_simulation = lambda: calls.append(("run", None))

    app._apply_optimized_values({"right_arm_start": 0.25})

    assert calls == [("set", {"right_arm_start": 0.25}), ("run", None)]


def test_apply_optimized_values_can_rerun_with_a_custom_prescribed_motion() -> None:
    """A custom optimized motion should bypass the default motion rebuild and be replayed directly."""

    scheduler = FakeScheduler()
    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app.root = scheduler
    app.result_var = FakeVar("Vrilles finales: 1.23 tours")
    calls: list[tuple[str, object]] = []
    app._set_values = lambda values: calls.append(("set", dict(values)))
    app._run_simulation = lambda: calls.append(("run_default", None))
    app._run_simulation_with_motion = lambda motion: calls.append(("run_motion", motion))

    app._apply_optimized_values({"right_arm_start": 0.25}, prescribed_motion="custom-motion")

    assert calls == [
        ("set", {"right_arm_start": 0.25}),
        ("run_motion", "custom-motion"),
    ]


def test_optimize_strategy_applies_optimized_values_and_reruns_animation(
    monkeypatch, tmp_path: Path
) -> None:
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
                    "final_twist_turns": -0.75,
                    "solver_status": "Solve_Succeeded",
                },
            )()

    monkeypatch.setattr(
        "best_tilting_plane.gui.app.TwistStrategyOptimizer.from_builder",
        lambda *_args, **_kwargs: FakeOptimizer(),
    )
    monkeypatch.setattr(
        "best_tilting_plane.gui.app.show_right_arm_start_sweep_figure",
        lambda **_kwargs: None,
    )

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app.root = FakeScheduler()
    app.result_var = FakeVar("")
    app.plot_y_var = FakeVar("Twist")
    app.optimization_mode_var = FakeVar("Optimize 2D")
    app._auto_runner = FakeRunner()
    app._current_values = lambda: {
        "right_arm_start": 0.1,
        "left_plane_initial": 0.0,
        "left_plane_final": 0.0,
        "right_plane_initial": 0.0,
        "right_plane_final": 0.0,
    }
    app._model_path = lambda: tmp_path / "reduced.bioMod"
    refresh_calls: list[str] = []
    app._refresh_plot = lambda: refresh_calls.append("refresh")
    applied: list[tuple[dict[str, float], str | None]] = []
    app._apply_optimized_values = lambda values, prescribed_motion=None, status_suffix=None: applied.append(
        (dict(values), status_suffix)
    )

    app._optimize_strategy()

    assert app._auto_runner.cancelled
    assert applied == [
        (
            {
                "first_arm_start": 0.0,
                "right_arm_start": 0.35,
                "left_plane_initial": 0.0,
                "left_plane_final": 0.0,
                "right_plane_initial": 0.0,
                "right_plane_final": 0.0,
            },
            "optimum Optimize 2D bras 1: -0.75 tours (Solve_Succeeded)",
        )
    ]
    assert app.plot_y_var.get() == "Twist"
    assert refresh_calls == []


def test_on_scan_plot_click_replays_the_nearest_candidate() -> None:
    """Clicking the dedicated scan figure should replay the nearest cached candidate solution."""

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app._scan_axis = object()
    app._selected_scan_solutions = []
    refreshed_scan: list[str] = []
    refreshed_plot: list[str] = []
    replayed: list[dict[str, object]] = []
    app._refresh_scan_plot = lambda: refreshed_scan.append("scan")
    app._refresh_plot = lambda: refreshed_plot.append("plot")
    app._apply_scan_candidate_solution = lambda candidate: replayed.append(candidate)
    candidate = {
        "mode": "Optimize 3D",
        "values": {
            "right_arm_start": 0.24,
            "left_plane_initial": 0.0,
            "left_plane_final": 0.0,
            "right_plane_initial": 0.0,
            "right_plane_final": 0.0,
            "contact_twist_turns_per_second": -0.6,
        },
        "final_twist_turns": -1.2,
        "objective": -1.19,
        "solver_status": "Solve_Succeeded",
        "success": True,
        "simulation": None,
    }
    app._scan_plot_datasets = lambda: [
        {
            "mode": "Optimize 3D",
            "start_times": [0.24, 0.26],
            "final_twist_turns": [-1.2, -1.0],
            "objective_values": [-1.19, -0.99],
            "success_mask": [True, True],
            "best_start_time": 0.24,
            "candidate_solutions": [candidate, None],
        }
    ]

    event = SimpleNamespace(inaxes=app._scan_axis, xdata=0.241, ydata=-1.19)
    app._on_scan_plot_click(event)

    assert app._selected_scan_solutions == [("Optimize 3D", 0)]
    assert refreshed_scan == ["scan"]
    assert refreshed_plot == ["plot"]
    assert replayed == [candidate]


def test_refresh_plot_overlays_two_selected_scan_conditions_with_solid_and_dashed_lines() -> None:
    """Two selected scan conditions should be shown as solid and dashed curves on the right plot."""

    simulation_a = AerialSimulationResult(
        time=np.array([0.0, 0.2, 0.4]),
        q=np.zeros((3, 10)),
        qdot=np.zeros((3, 10)),
        qddot=np.zeros((3, 10)),
        integrator_method="rk4",
        rk4_step=0.005,
        integration_seconds=None,
    )
    simulation_b = AerialSimulationResult(
        time=np.array([0.0, 0.2, 0.4]),
        q=np.zeros((3, 10)),
        qdot=np.zeros((3, 10)),
        qddot=np.zeros((3, 10)),
        integrator_method="rk4",
        rk4_step=0.005,
        integration_seconds=None,
    )

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app._visualization_data = {"result": simulation_a}
    app.plot_mode_var = FakeVar("Courbe")
    app.plot_y_var = FakeVar("Twist")
    app.plot_x_var = FakeVar("Temps")
    app._plot_axis = FakePlotAxis()
    app._plot_canvas = SimpleNamespace(draw_idle=lambda: None)
    app._selected_scan_solutions = [("Optimize 3D", 0), ("Optimize 3D BTP", 0)]
    app._scan_plot_datasets = lambda: [
        {
            "mode": "Optimize 3D",
            "candidate_solutions": [
                {
                    "mode": "Optimize 3D",
                    "values": {"right_arm_start": 0.24},
                    "simulation": simulation_a,
                }
            ],
        },
        {
            "mode": "Optimize 3D BTP",
            "candidate_solutions": [
                {
                    "mode": "Optimize 3D BTP",
                    "values": {"right_arm_start": 0.30},
                    "simulation": simulation_b,
                }
            ],
        },
    ]
    app._plot_data = lambda: (
        np.array([0.0, 0.2, 0.4]),
        np.array([0.0, 0.0, 0.0]),
        "Temps (s)",
        "Twist (deg)",
        "Twist en fonction de temps",
        None,
    )

    def fake_plot_data_for_result(result):
        if result is simulation_a:
            return (
                np.array([0.0, 0.2, 0.4]),
                np.array([1.0, 2.0, 3.0]),
                "Temps (s)",
                "Twist (deg)",
                "Twist en fonction de temps",
                None,
            )
        return (
            np.array([0.0, 0.2, 0.4]),
            np.array([1.5, 2.5, 3.5]),
            "Temps (s)",
            "Twist (deg)",
            "Twist en fonction de temps",
            None,
        )

    app._plot_data_for_result = fake_plot_data_for_result

    app._refresh_plot()

    assert len(app._plot_axis.plot_calls) == 2
    assert app._plot_axis.plot_calls[0]["linestyle"] == "-"
    assert app._plot_axis.plot_calls[1]["linestyle"] == "--"


def test_refresh_plot_labels_all_arm_velocity_curves_for_selected_solution() -> None:
    """Selected arm-velocity comparisons should keep one legend entry per DoF."""

    simulation = AerialSimulationResult(
        time=np.array([0.0, 0.2, 0.4]),
        q=np.zeros((3, 10)),
        qdot=np.zeros((3, 10)),
        qddot=np.zeros((3, 10)),
        integrator_method="rk4",
        rk4_step=0.005,
        integration_seconds=None,
    )

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app._visualization_data = {"result": simulation}
    app.plot_mode_var = FakeVar("Courbe")
    app.plot_y_var = FakeVar("Vitesses bras")
    app.plot_x_var = FakeVar("Temps")
    app._plot_axis = FakePlotAxis()
    app._plot_canvas = SimpleNamespace(draw_idle=lambda: None)
    app._selected_scan_solutions = [("Optimize 3D", 0)]
    app._scan_plot_datasets = lambda: [
        {
            "mode": "Optimize 3D",
            "candidate_solutions": [
                {
                    "mode": "Optimize 3D",
                    "values": {"right_arm_start": 0.24},
                    "simulation": simulation,
                }
            ],
        }
    ]
    app._plot_data = lambda: (
        np.array([0.0, 0.2, 0.4]),
        np.zeros((3, 4)),
        "Temps (s)",
        "Vitesses bras (deg/s)",
        "Vitesses bras en fonction de temps",
        (
            "Plan bras gauche",
            "Elevation bras gauche",
            "Plan bras droit",
            "Elevation bras droit",
        ),
    )
    app._plot_data_for_result = lambda _result: (
        np.array([0.0, 0.2, 0.4]),
        np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.5, 2.5, 3.5, 4.5],
                [2.0, 3.0, 4.0, 5.0],
            ]
        ),
        "Temps (s)",
        "Vitesses bras (deg/s)",
        "Vitesses bras en fonction de temps",
        (
            "Plan bras gauche",
            "Elevation bras gauche",
            "Plan bras droit",
            "Elevation bras droit",
        ),
    )

    app._refresh_plot()

    assert [call["label"] for call in app._plot_axis.plot_calls] == [
        "Plan bras gauche | t1=0.24 s",
        "Elevation bras gauche | t1=0.24 s",
        "Plan bras droit | t1=0.24 s",
        "Elevation bras droit | t1=0.24 s",
    ]


def test_refresh_plot_can_filter_selected_shoulder_torque_curves() -> None:
    """The shoulder-torque plot should honor the GUI curve selection."""

    simulation = AerialSimulationResult(
        time=np.array([0.0, 0.2, 0.4]),
        q=np.zeros((3, 10)),
        qdot=np.zeros((3, 10)),
        qddot=np.zeros((3, 10)),
        integrator_method="rk4",
        rk4_step=0.005,
        integration_seconds=None,
    )

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app._visualization_data = {"result": simulation}
    app.plot_mode_var = FakeVar("Courbe")
    app.plot_y_var = FakeVar("Couples epaules detail")
    app.plot_x_var = FakeVar("Temps")
    app._plot_axis = FakePlotAxis()
    app._plot_canvas = SimpleNamespace(draw_idle=lambda: None)
    app._curve_selection_by_plot = {"Couples epaules detail": ("Plan bras gauche | Total", "Plan bras droit | Mqddot")}
    app._selected_scan_solutions = []
    app._plot_data = lambda: (
        np.array([0.0, 0.2, 0.4]),
        np.arange(48, dtype=float).reshape(3, 16),
        "Temps (s)",
        "Couples epaules detail (N.m)",
        "Couples epaules detail en fonction de temps",
        BestTiltingPlaneApp._shoulder_torque_curve_labels(),
    )

    app._refresh_plot()

    assert [call["label"] for call in app._plot_axis.plot_calls] == [
        "Plan bras gauche | Total",
        "Plan bras droit | Mqddot",
    ]


def test_prepare_standard_animation_scene_adds_light_gray_dashed_overlay_for_second_condition() -> None:
    """A second selected condition should create dashed darker-gray artists in the 3D scene."""

    marker_names = {name for connection in SKELETON_CONNECTIONS for name in connection}
    trajectories = {name: np.zeros((2, 3)) for name in marker_names}

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app._visualization_data = {"trajectories": trajectories}
    app._secondary_visualization_data = {"trajectories": trajectories}
    app._animation_axis = Fake3DAxis()
    app._apply_camera_view = lambda: None
    app._draw_animation_frame = lambda _frame_index: None
    app.show_btp = FakeVar(False)
    app._animation_frame_index = 0

    app._prepare_standard_animation_scene()

    primary_calls = app._animation_axis.plot_calls[: len(SKELETON_CONNECTIONS)]
    secondary_calls = app._animation_axis.plot_calls[
        len(SKELETON_CONNECTIONS) : 2 * len(SKELETON_CONNECTIONS)
    ]
    assert primary_calls[1].kwargs["color"] == "tab:red"
    assert primary_calls[4].kwargs["color"] == "tab:blue"
    assert all(line.kwargs["color"] == "0.65" for line in secondary_calls)
    assert all(line.kwargs["linestyle"] == "--" for line in secondary_calls)


def test_prepare_standard_animation_scene_builds_root_hand_paths_when_requested() -> None:
    """The root-hand mode should create persistent left/right hand trajectory artists."""

    marker_names = {name for connection in SKELETON_CONNECTIONS for name in connection}
    trajectories = {name: np.zeros((12, 3)) for name in marker_names}

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app._visualization_data = {"trajectories": trajectories}
    app._secondary_visualization_data = None
    app._animation_axis = Fake3DAxis()
    app._apply_camera_view = lambda: None
    app._draw_animation_frame = lambda _frame_index: None
    app.show_btp = FakeVar(False)
    app._animation_frame_index = 0
    app.animation_reference_var = FakeVar(ANIMATION_REFERENCE_OPTIONS[2])

    app._prepare_standard_animation_scene()

    assert set(app._root_hand_path_artists) == {"left", "right"}


def test_draw_animation_frame_updates_secondary_overlay_from_selected_condition() -> None:
    """The 3D animation should advance the comparison overlay with the current frame."""

    marker_names = {name for connection in SKELETON_CONNECTIONS for name in connection}
    primary_trajectories = {
        name: np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=float)
        for name in marker_names
    }
    secondary_trajectories = {
        name: np.array([[1.0, 0.0, 0.0], [2.0, 0.5, 0.0]], dtype=float)
        for name in marker_names
    }
    first_connection = SKELETON_CONNECTIONS[0]

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app.animation_mode_var = FakeVar(ANIMATION_MODE_OPTIONS[0])
    app._secondary_line_artists = tuple(Fake3DLine() for _ in SKELETON_CONNECTIONS)
    app._line_artists = tuple(Fake3DLine() for _ in SKELETON_CONNECTIONS)
    app._frame_artists = {}
    app._angular_momentum_artist = Fake3DLine()
    app._plane_artist = None
    app._animation_canvas = SimpleNamespace(draw_idle=lambda: None)
    app._animation_axis = Fake3DAxis()
    app.show_btp = FakeVar(False)
    app._visualization_data = {
        "result": SimpleNamespace(time=np.array([0.0, 0.2])),
        "display_q": np.zeros((2, 10)),
        "trajectories": primary_trajectories,
        "frames": {},
        "observables": {
            "center_of_mass": np.zeros((2, 3)),
            "angular_momentum": np.zeros((2, 3)),
        },
    }
    app._secondary_visualization_data = {"trajectories": secondary_trajectories}

    app._draw_animation_frame(1)

    expected_segment = np.vstack(
        (secondary_trajectories[first_connection[0]][1], secondary_trajectories[first_connection[1]][1])
    )
    assert app._secondary_line_artists[0].x == expected_segment[:, 0].tolist()
    assert app._secondary_line_artists[0].y == expected_segment[:, 1].tolist()


def test_optimization_mode_options_hide_the_btp_mode_from_the_menu() -> None:
    """The GUI menu should expose the staged 2D/3D modes without the BTP mode."""

    assert OPTIMIZATION_MODE_OPTIONS == (
        OPTIMIZATION_MODE_ARM1_2D,
        OPTIMIZATION_MODE_ARM1_3D,
    )


def test_refresh_optimization_mode_options_unlocks_arm2_after_one_arm1_scan() -> None:
    """The staged workflow should unlock arm-2, then the combined mode, in sequence."""

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app.optimization_mode_var = FakeVar(OPTIMIZATION_MODE_ARM2_3D)
    app._optimization_mode_box = FakeCombobox()
    app._selected_first_arm_scan_solution = None
    app._load_cached_scan_bundle_for_mode = lambda mode: None

    app._refresh_optimization_mode_options()

    assert app._optimization_mode_box.options["values"] == OPTIMIZATION_MODE_OPTIONS
    assert app.optimization_mode_var.get() == OPTIMIZATION_MODE_ARM1_2D

    app._selected_first_arm_scan_solution = ("Arm1 3D", 0)

    app._refresh_optimization_mode_options()

    assert app._optimization_mode_box.options["values"] == (
        OPTIMIZATION_MODE_ARM1_2D,
        OPTIMIZATION_MODE_ARM1_3D,
        OPTIMIZATION_MODE_ARM2_2D,
        OPTIMIZATION_MODE_ARM2_3D,
    )
    assert OPTIMIZATION_MODE_ARM2_2D in app._optimization_mode_box.options["values"]
    assert OPTIMIZATION_MODE_BOTH_3D not in app._optimization_mode_box.options["values"]

    app._load_cached_scan_bundle_for_mode = (
        lambda mode: {"mode": mode} if mode == OPTIMIZATION_MODE_ARM2_3D else None
    )

    app._refresh_optimization_mode_options()

    assert app._optimization_mode_box.options["values"] == (
        OPTIMIZATION_MODE_ARM1_2D,
        OPTIMIZATION_MODE_ARM1_3D,
        OPTIMIZATION_MODE_ARM2_2D,
        OPTIMIZATION_MODE_ARM2_3D,
        OPTIMIZATION_MODE_BOTH_3D,
    )


def test_optimization_progress_updates_the_popup_with_completed_over_total() -> None:
    """The progress popup should track completed evaluations against the total."""

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app.root = FakeScheduler()
    app.result_var = FakeVar("")
    app._optimization_progress_window = FakeWindow()
    app._optimization_progress_var = FakeVar("")
    app._optimization_progress_bar = FakeProgressbar()

    app._optimization_progress(
        {
            "message": "Optimisation 2D bras 1... t0=0.20 s",
            "completed": 3,
            "total": 12,
        }
    )

    assert app.result_var.get() == "Optimisation 2D bras 1... t0=0.20 s"
    assert app._optimization_progress_var.get() == "Optimisation 2D bras 1... t0=0.20 s (3/12)"
    assert app._optimization_progress_bar.options["maximum"] == 12.0
    assert app._optimization_progress_bar.options["value"] == 3.0


def test_load_cached_optimized_values_reads_matching_record(tmp_path: Path) -> None:
    """A matching cache entry should be returned as GUI-ready float values."""

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app.optimization_mode_var = FakeVar("Optimize 2D")
    app._model_path = lambda: tmp_path / "reduced.bioMod"

    cache_path = tmp_path / "optimization_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "records": {
                    "optimize_2d": {
                        "signature": app._optimization_cache_signature(),
                        "values": {
                            "first_arm_start": 0.0,
                            "right_arm_start": 0.35,
                            "left_plane_initial": 0.0,
                            "left_plane_final": 0.0,
                            "right_plane_initial": 0.0,
                            "right_plane_final": 0.0,
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert app._load_cached_optimized_values() == {
        "contact_twist_turns_per_second": 0.0,
        "first_arm_start": 0.0,
        "right_arm_start": 0.35,
        "left_plane_initial": 0.0,
        "left_plane_final": 0.0,
        "right_plane_initial": 0.0,
        "right_plane_final": 0.0,
    }


def test_optimize_strategy_uses_cache_without_running_ipopt(monkeypatch, tmp_path: Path) -> None:
    """A cached optimum should be applied directly without rebuilding the optimizer."""

    monkeypatch.setattr(
        "best_tilting_plane.gui.app.TwistStrategyOptimizer.from_builder",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("optimizer should not run")),
    )
    monkeypatch.setattr(
        "best_tilting_plane.gui.app.show_right_arm_start_sweep_figure",
        lambda **_kwargs: None,
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
    app._model_path = lambda: tmp_path / "reduced.bioMod"
    (tmp_path / "optimization_cache.json").write_text(
        json.dumps(
            {
                "records": {
                    "arm1_2d": {
                        "signature": app._optimization_cache_signature_for_mode("Arm1 2D"),
                        "values": {
                            "first_arm_start": 0.0,
                            "right_arm_start": 0.42,
                            "left_plane_initial": 0.0,
                            "left_plane_final": 0.0,
                            "right_plane_initial": 0.0,
                            "right_plane_final": 0.0,
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    applied: list[tuple[dict[str, float], str | None]] = []
    app._apply_optimized_values = lambda values, prescribed_motion=None, status_suffix=None: applied.append(
        (dict(values), status_suffix)
    )

    app._optimize_strategy()

    assert app._auto_runner.cancelled
    assert applied == [
        (
            {
                "contact_twist_turns_per_second": 0.0,
                "first_arm_start": 0.0,
                "right_arm_start": 0.42,
                "left_plane_initial": 0.0,
                "left_plane_final": 0.0,
                "right_plane_initial": 0.0,
                "right_plane_final": 0.0,
            },
            "optimum Optimize 2D bras 1 charge depuis le cache",
        )
    ]


def test_optimize_strategy_can_ignore_cached_ipopt_solution(monkeypatch, tmp_path: Path) -> None:
    """The force-recompute option should bypass a stored 2D optimum and run IPOPT again."""

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
                    "final_twist_turns": -0.75,
                    "solver_status": "Solve_Succeeded",
                },
            )()

    monkeypatch.setattr(
        "best_tilting_plane.gui.app.TwistStrategyOptimizer.from_builder",
        lambda *_args, **_kwargs: FakeOptimizer(),
    )
    monkeypatch.setattr(
        "best_tilting_plane.gui.app.show_right_arm_start_sweep_figure",
        lambda **_kwargs: None,
    )

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app.root = FakeScheduler()
    app.result_var = FakeVar("")
    app.optimization_mode_var = FakeVar("Optimize 2D")
    app.ignore_optimization_cache_var = FakeVar(True)
    app._auto_runner = FakeRunner()
    app._current_values = lambda: {
        "right_arm_start": 0.1,
        "left_plane_initial": 0.0,
        "left_plane_final": 0.0,
        "right_plane_initial": 0.0,
        "right_plane_final": 0.0,
    }
    app._model_path = lambda: tmp_path / "reduced.bioMod"
    (tmp_path / "optimization_cache.json").write_text(
        json.dumps(
            {
                "records": {
                    "arm1_2d": {
                        "signature": app._optimization_cache_signature_for_mode("Arm1 2D"),
                        "values": {
                            "first_arm_start": 0.0,
                            "right_arm_start": 0.42,
                            "left_plane_initial": 0.0,
                            "left_plane_final": 0.0,
                            "right_plane_initial": 0.0,
                            "right_plane_final": 0.0,
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    applied: list[tuple[dict[str, float], str | None]] = []
    app._apply_optimized_values = lambda values, prescribed_motion=None, status_suffix=None: applied.append(
        (dict(values), status_suffix)
    )

    app._optimize_strategy()

    assert applied == [
        (
            {
                "first_arm_start": 0.0,
                "right_arm_start": 0.35,
                "left_plane_initial": 0.0,
                "left_plane_final": 0.0,
                "right_plane_initial": 0.0,
                "right_plane_final": 0.0,
            },
            "optimum Optimize 2D bras 1: -0.75 tours (Solve_Succeeded)",
        )
    ]


def test_optimize_strategy_reports_errors_instead_of_raising(monkeypatch, tmp_path: Path) -> None:
    """A failing optimizer should leave a readable error in the GUI instead of crashing Tkinter."""

    class FakeOptimizer:
        def optimize_right_arm_start_only(self, initial_right_arm_start, **_kwargs):
            assert initial_right_arm_start == 0.1
            raise ValueError("boom")

    monkeypatch.setattr(
        "best_tilting_plane.gui.app.TwistStrategyOptimizer.from_builder",
        lambda *_args, **_kwargs: FakeOptimizer(),
    )
    monkeypatch.setattr(
        "best_tilting_plane.gui.app.show_right_arm_start_sweep_figure",
        lambda **_kwargs: None,
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
    app._model_path = lambda: tmp_path / "reduced.bioMod"

    app._optimize_strategy()

    assert app._auto_runner.cancelled
    assert app.result_var.get() == "Erreur optimisation: boom"


def test_optimize_strategy_writes_cache_after_ipopt(monkeypatch, tmp_path: Path) -> None:
    """A newly optimized strategy should be persisted for later GUI sessions."""

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
                    "final_twist_turns": -0.75,
                    "solver_status": "Solve_Succeeded",
                },
            )()

    monkeypatch.setattr(
        "best_tilting_plane.gui.app.TwistStrategyOptimizer.from_builder",
        lambda *_args, **_kwargs: FakeOptimizer(),
    )
    monkeypatch.setattr(
        "best_tilting_plane.gui.app.show_right_arm_start_sweep_figure",
        lambda **_kwargs: None,
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
    app._model_path = lambda: tmp_path / "reduced.bioMod"
    applied: list[tuple[dict[str, float], str | None]] = []
    app._apply_optimized_values = lambda values, prescribed_motion=None, status_suffix=None: applied.append(
        (dict(values), status_suffix)
    )

    app._optimize_strategy()

    stored = json.loads((tmp_path / "optimization_cache.json").read_text(encoding="utf-8"))
    record = stored["records"]["arm1_2d"]
    assert record["values"] == {
        "contact_twist_turns_per_second": 0.0,
        "first_arm_start": 0.0,
        "right_arm_start": 0.35,
        "left_plane_initial": 0.0,
        "left_plane_final": 0.0,
        "right_plane_initial": 0.0,
        "right_plane_final": 0.0,
    }
    assert record["final_twist_turns"] == -0.75
    assert record["solver_status"] == "Solve_Succeeded"
    assert applied == [
        (
            {
                "first_arm_start": 0.0,
                "right_arm_start": 0.35,
                "left_plane_initial": 0.0,
                "left_plane_final": 0.0,
                "right_plane_initial": 0.0,
                "right_plane_final": 0.0,
            },
            "optimum Optimize 2D bras 1: -0.75 tours (Solve_Succeeded)",
        )
    ]


def test_load_cached_dms_solution_rebuilds_the_prescribed_motion(tmp_path: Path) -> None:
    """A matching DMS cache entry should rebuild the jerk-driven prescribed motion."""

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app.optimization_mode_var = FakeVar("Optimize DMS")
    app._model_path = lambda: tmp_path / "reduced.bioMod"
    app._standard_optimization_configuration = lambda: SimulationConfiguration(
        final_time=1.0,
        integrator="rk4",
        rk4_step=0.005,
    )

    cache_path = tmp_path / "optimization_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "records": {
                    "optimize_dms": {
                        "signature": app._optimization_cache_signature(),
                        "values": {
                            "right_arm_start": 0.28,
                            "left_plane_initial": 0.0,
                            "left_plane_final": 0.0,
                            "right_plane_initial": 0.0,
                            "right_plane_final": 0.0,
                        },
                        "left_plane_jerk": [0.0] * 15,
                        "right_plane_jerk": [0.0] * 15,
                        "scan_start_times": [0.10, 0.12, 0.14],
                        "scan_final_twist_turns": [-0.40, -0.55, -0.63],
                        "scan_objective_values": [-0.39, -0.54, -0.62],
                        "scan_success_mask": [True, True, True],
                        "final_twist_turns": -0.63,
                        "solver_status": "Solve_Succeeded",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    cached = app._load_cached_dms_solution()

    assert cached is not None
    values, motion, final_twist_turns, solver_status, scan_data = cached
    assert values["right_arm_start"] == 0.28
    assert motion.left_arm_start == 0.28
    assert motion.right_arm_start == 0.0
    assert motion.left_plane.jerks.shape == (15,)
    assert motion.right_plane.jerks.shape == (15,)
    assert final_twist_turns == -0.63
    assert solver_status == "Solve_Succeeded"
    assert scan_data is not None
    assert scan_data["start_times"] == [0.10, 0.12, 0.14]


def test_cached_simulation_result_reuses_qddot_when_present(tmp_path: Path) -> None:
    """Cached simulation replay should preserve stored accelerations when available."""

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app._standard_optimization_configuration = lambda: SimulationConfiguration(
        final_time=1.0,
        integrator="rk4",
        rk4_step=0.005,
    )

    q = np.zeros((3, 10))
    qdot = np.ones((3, 10))
    qddot = 2.0 * np.ones((3, 10))
    result = app._cached_simulation_result_from_record(
        {
            "q": q.tolist(),
            "qdot": qdot.tolist(),
            "qddot": qddot.tolist(),
        }
    )

    assert result is not None
    np.testing.assert_allclose(result.qddot, qddot)


def test_optimization_cache_key_changes_with_contact_twist_slider() -> None:
    """Different contact-twist slider values should map to distinct cache entries."""

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app._entries = {"contact_twist_turns_per_second": FakeVar(-0.4)}

    assert app._optimization_cache_key_for_mode("Optimize 3D") == "optimize_3d__contact_twist_m0p4"


def test_load_cached_dms_solution_ignores_stale_signature(tmp_path: Path) -> None:
    """An obsolete DMS cache entry should not be replayed after formulation changes."""

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app.optimization_mode_var = FakeVar("Optimize DMS")
    app._model_path = lambda: tmp_path / "reduced.bioMod"
    app._standard_optimization_configuration = lambda: SimulationConfiguration(
        final_time=1.0,
        integrator="rk4",
        rk4_step=0.005,
    )

    stale_signature = dict(app._optimization_cache_signature())
    stale_signature["version"] = stale_signature["version"] - 1
    cache_path = tmp_path / "optimization_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "records": {
                    "optimize_dms": {
                        "signature": stale_signature,
                        "values": {
                            "right_arm_start": 0.28,
                            "left_plane_initial": 0.0,
                            "left_plane_final": 0.0,
                            "right_plane_initial": 0.0,
                            "right_plane_final": 0.0,
                        },
                        "left_plane_jerk": [100.0] * 15,
                        "right_plane_jerk": [-100.0] * 15,
                        "final_twist_turns": -0.63,
                        "solver_status": "Solve_Succeeded",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert app._load_cached_dms_solution() is None


def test_optimize_3d_can_reuse_legacy_optimize_dms_cache(tmp_path: Path) -> None:
    """The renamed 3D mode should still replay a compatible legacy `optimize_dms` cache entry."""

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app.optimization_mode_var = FakeVar("Optimize 3D")
    app._model_path = lambda: tmp_path / "reduced.bioMod"
    app._standard_optimization_configuration = lambda: SimulationConfiguration(
        final_time=1.0,
        integrator="rk4",
        rk4_step=0.005,
    )

    legacy_signature = dict(app._optimization_cache_signature_for_mode("Optimize 3D"))
    legacy_signature["version"] -= 1
    legacy_signature["mode"] = "optimize_dms"
    legacy_signature.pop("dms_objective_mode")
    legacy_signature.pop("dms_btp_deviation_weight")
    (tmp_path / "optimization_cache.json").write_text(
        json.dumps(
            {
                "records": {
                    "optimize_dms": {
                        "signature": legacy_signature,
                        "values": {
                            "right_arm_start": 0.28,
                            "left_plane_initial": 0.0,
                            "left_plane_final": 0.0,
                            "right_plane_initial": 0.0,
                            "right_plane_final": 0.0,
                        },
                        "left_plane_jerk": [0.0] * 15,
                        "right_plane_jerk": [0.0] * 15,
                        "final_twist_turns": -0.63,
                        "solver_status": "Solve_Succeeded",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    cached = app._load_cached_dms_solution()

    assert cached is not None
    assert cached[0]["right_arm_start"] == 0.28


def test_optimize_strategy_uses_cached_dms_solution_without_running_solver(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A cached DMS optimum should be replayed directly without rebuilding the DMS optimizer."""

    monkeypatch.setattr(
        "best_tilting_plane.gui.app.DirectMultipleShootingOptimizer.from_builder",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("DMS optimizer should not run")),
    )

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    scheduler = FakeScheduler()
    app.root = scheduler
    app.result_var = FakeVar("")
    app.optimization_mode_var = FakeVar("Optimize DMS")
    app._auto_runner = FakeRunner()
    app._current_values = lambda: {
        "right_arm_start": 0.1,
        "left_plane_initial": 0.0,
        "left_plane_final": 0.0,
        "right_plane_initial": 0.0,
        "right_plane_final": 0.0,
    }
    app._model_path = lambda: tmp_path / "reduced.bioMod"
    app._standard_optimization_configuration = lambda: SimulationConfiguration(
        final_time=1.0,
        integrator="rk4",
        rk4_step=0.005,
    )
    shown: list[dict[str, object]] = []
    app._show_dms_sweep_figure = lambda **kwargs: shown.append(dict(kwargs))
    (tmp_path / "optimization_cache.json").write_text(
        json.dumps(
            {
                "records": {
                    "optimize_dms": {
                        "signature": app._optimization_cache_signature(),
                        "values": {
                            "right_arm_start": 0.28,
                            "left_plane_initial": 0.0,
                            "left_plane_final": 0.0,
                            "right_plane_initial": 0.0,
                            "right_plane_final": 0.0,
                        },
                        "left_plane_jerk": [0.0] * 15,
                        "right_plane_jerk": [0.0] * 15,
                        "scan_start_times": [0.10, 0.12, 0.28],
                        "scan_final_twist_turns": [-0.40, -0.55, -0.63],
                        "scan_objective_values": [-0.39, -0.54, -0.62],
                        "scan_success_mask": [True, True, True],
                        "final_twist_turns": -0.63,
                        "solver_status": "Solve_Succeeded",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    applied: list[tuple[dict[str, float], object, str | None]] = []
    app._apply_optimized_values = lambda values, prescribed_motion=None, status_suffix=None: applied.append(
        (dict(values), prescribed_motion, status_suffix)
    )

    app._optimize_strategy()
    scheduler.run_pending()

    assert app._auto_runner.cancelled
    assert applied[0][0] == {
        "contact_twist_turns_per_second": 0.0,
        "first_arm_start": 0.0,
        "right_arm_start": 0.28,
        "left_plane_initial": 0.0,
        "left_plane_final": 0.0,
        "right_plane_initial": 0.0,
        "right_plane_final": 0.0,
    }
    assert applied[0][1] is not None
    assert applied[0][2] == "optimum DMS bras 2 charge depuis le cache: -0.63 tours (Solve_Succeeded)"
    assert len(shown) == 1
    np.testing.assert_allclose(shown[0]["start_times"], [0.10, 0.12, 0.28])
    np.testing.assert_allclose(shown[0]["final_twist_turns"], [-0.40, -0.55, -0.63])
    np.testing.assert_allclose(shown[0]["objective_values"], [-0.39, -0.54, -0.62])
    np.testing.assert_array_equal(shown[0]["success_mask"], [True, True, True])
    assert shown[0]["best_start_time"] == 0.28


def test_show_dms_sweep_figure_overlays_cached_2d_scan_when_available(tmp_path: Path) -> None:
    """The DMS sweep should forward the cached 2D scan to the comparison figure."""

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app.optimization_mode_var = FakeVar("Optimize DMS")
    app._model_path = lambda: tmp_path / "reduced.bioMod"
    app._standard_optimization_configuration = lambda: SimulationConfiguration(
        final_time=1.0,
        integrator="rk4",
        rk4_step=0.005,
    )
    (tmp_path / "optimization_cache.json").write_text(
        json.dumps(
            {
                "records": {
                    "optimize_2d": {
                        "signature": app._optimization_cache_signature_for_mode("Optimize 2D"),
                        "values": {
                            "right_arm_start": 0.20,
                            "left_plane_initial": 0.0,
                            "left_plane_final": 0.0,
                            "right_plane_initial": 0.0,
                            "right_plane_final": 0.0,
                        },
                        "scan_start_times": [0.10, 0.20],
                        "scan_final_twist_turns": [-0.30, -0.50],
                        "scan_objective_values": [-0.29, -0.49],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    captured: list[dict[str, object]] = []
    app._show_scan_comparison_figure = lambda **kwargs: captured.append(dict(kwargs))

    app._show_dms_sweep_figure(
        start_times=[0.16, 0.18],
        final_twist_turns=[-0.40, -0.55],
        objective_values=[-0.39, -0.54],
        success_mask=[True, True],
        best_start_time=0.18,
    )

    assert len(captured) == 1
    comparison_scan = captured[0]["comparison_scan"]
    assert comparison_scan["mode"] == "Optimize 2D"
    assert comparison_scan["best_start_time"] == 0.20


def test_optimize_strategy_runs_dms_and_replays_the_optimized_motion(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The DMS mode should push both slider values and the optimized motion."""

    class FakeDmsOptimizer:
        def candidate_start_times(self):
            return np.array([0.10, 0.28, 0.30])

        def __init__(self) -> None:
            self.interval_count = 50
            self.active_control_count = 15
            self.node_times = np.linspace(0.0, 1.0, self.interval_count + 1)

        def _global_jerk_bounds(self, *, right_start_node_index):
            lower = np.full(self.interval_count, -1.0, dtype=float)
            upper = np.full(self.interval_count, 1.0, dtype=float)
            return lower, upper, lower.copy(), upper.copy()

        def solve_fixed_start(self, initial_guess, *, right_arm_start, previous_result=None, **_kwargs):
            assert initial_guess.right_arm_start == 0.1
            assert _kwargs["show_jerk_diagnostics"] is False
            if previous_result is None:
                assert right_arm_start == 0.10
            elif np.isclose(right_arm_start, 0.28):
                assert previous_result.variables.right_arm_start == 0.10
            else:
                assert previous_result.variables.right_arm_start == 0.28
            return type(
                "DmsResult",
                (),
                {
                    "variables": type(
                        "Variables",
                        (),
                        {
                            "right_arm_start": right_arm_start,
                            "left_plane_initial": 0.0,
                            "left_plane_final": 0.0,
                            "right_plane_initial": 0.0,
                            "right_plane_final": 0.0,
                        },
                    )(),
                    "prescribed_motion": "dms-motion" if np.isclose(right_arm_start, 0.28) else f"dms-motion-{right_arm_start:.2f}",
                    "left_plane_jerk": np.zeros(15),
                    "right_plane_jerk": np.zeros(15),
                    "final_twist_turns": -0.40 if np.isclose(right_arm_start, 0.10) else -0.63 if np.isclose(right_arm_start, 0.28) else -0.61,
                    "objective": -0.39 if np.isclose(right_arm_start, 0.10) else -0.62 if np.isclose(right_arm_start, 0.28) else -0.60,
                    "solver_status": "Solve_Succeeded",
                    "success": True,
                    "warm_start_primal": np.full(20, right_arm_start),
                    "warm_start_lam_x": np.full(20, 1.0),
                    "warm_start_lam_g": np.full(10, 2.0),
                },
            )()

    monkeypatch.setattr(
        "best_tilting_plane.gui.app.DirectMultipleShootingOptimizer.from_builder",
        lambda *_args, **_kwargs: FakeDmsOptimizer(),
    )

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    scheduler = FakeScheduler()
    app.root = scheduler
    app.result_var = FakeVar("")
    app.optimization_mode_var = FakeVar("Optimize DMS")
    app._auto_runner = FakeRunner()
    app._current_values = lambda: {
        "right_arm_start": 0.1,
        "left_plane_initial": 0.0,
        "left_plane_final": 0.0,
        "right_plane_initial": 0.0,
        "right_plane_final": 0.0,
    }
    app._model_path = lambda: tmp_path / "reduced.bioMod"
    app._standard_optimization_configuration = lambda: SimulationConfiguration(
        final_time=1.0,
        integrator="rk4",
        rk4_step=0.005,
    )
    shown: list[dict[str, object]] = []
    app._show_dms_sweep_figure = lambda **kwargs: shown.append(dict(kwargs))
    jerk_diagnostics: list[dict[str, object]] = []
    app._schedule_dms_jerk_diagnostic_figure = lambda **kwargs: jerk_diagnostics.append(dict(kwargs))
    applied: list[tuple[dict[str, float], object, str | None]] = []
    app._apply_optimized_values = lambda values, prescribed_motion=None, status_suffix=None: applied.append(
        (dict(values), prescribed_motion, status_suffix)
    )

    app._optimize_strategy()
    scheduler.run_pending()

    assert app._auto_runner.cancelled
    assert applied == [
        (
            {
                "first_arm_start": 0.0,
                "right_arm_start": 0.28,
                "left_plane_initial": 0.0,
                "left_plane_final": 0.0,
                "right_plane_initial": 0.0,
                "right_plane_final": 0.0,
            },
            "dms-motion",
            "optimum DMS bras 2: -0.63 tours (Solve_Succeeded)",
        )
    ]
    assert len(shown) == 1
    np.testing.assert_allclose(shown[0]["start_times"], [0.10, 0.28, 0.30])
    np.testing.assert_allclose(shown[0]["final_twist_turns"], [-0.40, -0.63, -0.61])
    np.testing.assert_allclose(shown[0]["objective_values"], [-0.39, -0.62, -0.60])
    np.testing.assert_array_equal(shown[0]["success_mask"], [True, True, True])
    assert shown[0]["best_start_time"] == 0.28
    assert len(jerk_diagnostics) == 1


def test_optimize_strategy_uses_btp_objective_for_optimize_3d_btp(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The `Optimize 3D BTP` mode should request the DMS objective with the BTP Lagrange term."""

    captured: dict[str, object] = {}

    class FakeDmsOptimizer:
        def candidate_start_times(self):
            return np.array([0.10], dtype=float)

        def solve_fixed_start(self, initial_guess, *, right_arm_start, **_kwargs):
            assert initial_guess.right_arm_start == 0.1
            return type(
                "DmsResult",
                (),
                {
                    "variables": type(
                        "Variables",
                        (),
                        {
                            "right_arm_start": right_arm_start,
                            "left_plane_initial": 0.0,
                            "left_plane_final": 0.0,
                            "right_plane_initial": 0.0,
                            "right_plane_final": 0.0,
                        },
                    )(),
                    "prescribed_motion": "dms-btp-motion",
                    "left_plane_jerk": np.zeros(15),
                    "right_plane_jerk": np.zeros(15),
                    "final_twist_turns": -0.52,
                    "objective": -0.50,
                    "solver_status": "Solve_Succeeded",
                    "success": True,
                    "warm_start_primal": np.full(20, right_arm_start),
                    "warm_start_lam_x": np.full(20, 1.0),
                    "warm_start_lam_g": np.full(10, 2.0),
                },
            )()

    def fake_from_builder(*_args, **kwargs):
        captured.update(kwargs)
        return FakeDmsOptimizer()

    monkeypatch.setattr(
        "best_tilting_plane.gui.app.DirectMultipleShootingOptimizer.from_builder",
        fake_from_builder,
    )

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    scheduler = FakeScheduler()
    app.root = scheduler
    app.result_var = FakeVar("")
    app.optimization_mode_var = FakeVar("Optimize 3D BTP")
    app._auto_runner = FakeRunner()
    app._current_values = lambda: {
        "right_arm_start": 0.1,
        "left_plane_initial": 0.0,
        "left_plane_final": 0.0,
        "right_plane_initial": 0.0,
        "right_plane_final": 0.0,
    }
    app._model_path = lambda: tmp_path / "reduced.bioMod"
    app._standard_optimization_configuration = lambda: SimulationConfiguration(
        final_time=1.0,
        integrator="rk4",
        rk4_step=0.005,
    )
    app._show_dms_sweep_figure = lambda **kwargs: None
    app._schedule_dms_jerk_diagnostic_figure = lambda **kwargs: None
    app._apply_optimized_values = lambda values, prescribed_motion=None, status_suffix=None: None

    app._optimize_strategy()
    scheduler.run_pending()

    assert captured["objective_mode"] == "twist_btp"
    assert captured["btp_deviation_weight"] > 0.0


def test_optimize_strategy_uses_multistart_for_t1_equal_to_best_2d_minus_0_2(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The 3D GUI path should use the dedicated multistart helper at the snapped `t1_2D - 0.2` node."""

    calls: list[tuple[str, float]] = []

    class FakeDmsOptimizer:
        def candidate_start_times(self):
            return np.array([0.28, 0.30, 0.32], dtype=float)

        def solve_fixed_start(self, initial_guess, *, right_arm_start, previous_result=None, **_kwargs):
            assert initial_guess.right_arm_start == 0.1
            calls.append(("single", float(right_arm_start)))
            return type(
                "DmsResult",
                (),
                {
                    "variables": type(
                        "Variables",
                        (),
                        {
                            "right_arm_start": right_arm_start,
                            "left_plane_initial": 0.0,
                            "left_plane_final": 0.0,
                            "right_plane_initial": 0.0,
                            "right_plane_final": 0.0,
                        },
                    )(),
                    "prescribed_motion": f"dms-motion-{right_arm_start:.2f}",
                    "left_plane_jerk": np.zeros(15),
                    "right_plane_jerk": np.zeros(15),
                    "final_twist_turns": -0.55 if np.isclose(right_arm_start, 0.28) else -0.58,
                    "objective": -0.54 if np.isclose(right_arm_start, 0.28) else -0.57,
                    "solver_status": "Solve_Succeeded",
                    "success": True,
                    "right_arm_start_node_index": int(round(right_arm_start / 0.02)),
                    "warm_start_primal": np.full(20, right_arm_start),
                    "warm_start_lam_x": np.full(20, 1.0),
                    "warm_start_lam_g": np.full(10, 2.0),
                },
            )()

        def solve_fixed_start_multistart(self, initial_guess, *, right_arm_start, **_kwargs):
            assert initial_guess.right_arm_start == 0.1
            calls.append(("multi", float(right_arm_start)))
            return type(
                "DmsResult",
                (),
                {
                    "variables": type(
                        "Variables",
                        (),
                        {
                            "right_arm_start": right_arm_start,
                            "left_plane_initial": 0.0,
                            "left_plane_final": 0.0,
                            "right_plane_initial": 0.0,
                            "right_plane_final": 0.0,
                        },
                    )(),
                    "prescribed_motion": "dms-motion-0.30-best",
                    "left_plane_jerk": np.zeros(15),
                    "right_plane_jerk": np.zeros(15),
                    "final_twist_turns": -0.80,
                    "objective": -0.79,
                    "solver_status": "Solve_Succeeded",
                    "success": True,
                    "right_arm_start_node_index": 15,
                    "warm_start_primal": np.full(20, 0.30),
                    "warm_start_lam_x": np.full(20, 1.0),
                    "warm_start_lam_g": np.full(10, 2.0),
                },
            )()

    monkeypatch.setattr(
        "best_tilting_plane.gui.app.DirectMultipleShootingOptimizer.from_builder",
        lambda *_args, **_kwargs: FakeDmsOptimizer(),
    )

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app.root = FakeScheduler()
    app.result_var = FakeVar("")
    app.optimization_mode_var = FakeVar("Optimize DMS")
    app._auto_runner = FakeRunner()
    app._current_values = lambda: {
        "right_arm_start": 0.1,
        "left_plane_initial": 0.0,
        "left_plane_final": 0.0,
        "right_plane_initial": 0.0,
        "right_plane_final": 0.0,
    }
    app._model_path = lambda: tmp_path / "reduced.bioMod"
    app._standard_optimization_configuration = lambda: SimulationConfiguration(
        final_time=1.0,
        integrator="rk4",
        rk4_step=0.005,
    )
    app._show_dms_sweep_figure = lambda **kwargs: None
    app._schedule_dms_jerk_diagnostic_figure = lambda **kwargs: None
    app._schedule_scan_figure = lambda **kwargs: None
    app._apply_optimized_values = lambda values, prescribed_motion=None, status_suffix=None: None
    (tmp_path / "optimization_cache.json").write_text(
        json.dumps(
            {
                "records": {
                    "optimize_2d": {
                        "signature": app._optimization_cache_signature_for_mode("Optimize 2D"),
                        "values": {
                            "right_arm_start": 0.50,
                            "left_plane_initial": 0.0,
                            "left_plane_final": 0.0,
                            "right_plane_initial": 0.0,
                            "right_plane_final": 0.0,
                            "contact_twist_turns_per_second": 0.0,
                        },
                        "final_twist_turns": -0.70,
                        "solver_status": "Discrete_Sweep",
                        "scan_start_times": [0.10, 0.30, 0.50],
                        "scan_final_twist_turns": [-0.55, -0.65, -0.70],
                        "scan_objective_values": [-0.54, -0.64, -0.69],
                    }
                },
                "progress_records": {},
            }
        ),
        encoding="utf-8",
    )

    app._optimize_strategy()

    assert calls == [("single", 0.28), ("multi", 0.3), ("single", 0.32)]


def test_compute_optimization_outcome_3d_attaches_exact_simulation_to_prescribed_motion(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A fresh 3D optimization should reuse the exact optimized simulation when replaying in the GUI."""

    simulation = SimpleNamespace(
        q=np.array([[0.0] * 10, [0.1] * 10], dtype=float),
        qdot=np.array([[0.0] * 10, [0.2] * 10], dtype=float),
        final_twist_turns=-0.8,
    )

    class FakeDmsOptimizer:
        def candidate_start_times(self):
            return np.array([0.10], dtype=float)

        def solve_fixed_start(self, initial_guess, *, right_arm_start, previous_result=None, **_kwargs):
            del initial_guess, previous_result
            motion = SimpleNamespace(tag="fresh-motion")
            return SimpleNamespace(
                variables=SimpleNamespace(
                    right_arm_start=right_arm_start,
                    left_plane_initial=0.0,
                    left_plane_final=0.0,
                    right_plane_initial=0.0,
                    right_plane_final=0.0,
                ),
                prescribed_motion=motion,
                simulation=simulation,
                left_plane_jerk=np.zeros(15),
                right_plane_jerk=np.zeros(15),
                final_twist_turns=-0.8,
                objective=-0.79,
                solver_status="Solve_Succeeded",
                success=True,
                warm_start_primal=np.full(20, right_arm_start),
                warm_start_lam_x=np.full(20, 1.0),
                warm_start_lam_g=np.full(10, 2.0),
            )

    monkeypatch.setattr(
        "best_tilting_plane.gui.app.DirectMultipleShootingOptimizer.from_builder",
        lambda *_args, **_kwargs: FakeDmsOptimizer(),
    )

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app.root = FakeScheduler()
    app.result_var = FakeVar("")
    app.optimization_mode_var = FakeVar("Optimize 3D")
    app._current_values = lambda: {
        "right_arm_start": 0.1,
        "left_plane_initial": 0.0,
        "left_plane_final": 0.0,
        "right_plane_initial": 0.0,
        "right_plane_final": 0.0,
        "contact_twist_turns_per_second": -0.6,
    }
    app._model_path = lambda: tmp_path / "reduced.bioMod"
    app._standard_optimization_configuration = lambda: SimulationConfiguration(
        final_time=1.0,
        integrator="rk4",
        rk4_step=0.005,
        contact_twist_rate=-0.6 * 2.0 * np.pi,
    )

    outcome = app._compute_optimization_outcome(
        current_values=app._current_values(),
        mode="Optimize 3D",
        use_cache=False,
    )

    assert getattr(outcome["prescribed_motion"], "_cached_simulation_result") is simulation


def test_optimize_strategy_keeps_best_previous_warm_start_when_one_node_is_worse(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A bad intermediate node should not replace the better warm start used for later `t1` values."""

    calls: list[tuple[float, float | None]] = []

    class FakeDmsOptimizer:
        def candidate_start_times(self):
            return np.array([0.60, 0.62, 0.64], dtype=float)

        def solve_fixed_start(self, initial_guess, *, right_arm_start, previous_result=None, **_kwargs):
            assert initial_guess.right_arm_start == 0.1
            calls.append((right_arm_start, None if previous_result is None else float(previous_result.warm_start_primal[0])))
            objective = -0.80 if np.isclose(right_arm_start, 0.60) else -0.20 if np.isclose(right_arm_start, 0.62) else -0.79
            final_twist = -0.81 if np.isclose(right_arm_start, 0.60) else -0.21 if np.isclose(right_arm_start, 0.62) else -0.80
            return type(
                "DmsResult",
                (),
                {
                    "variables": type(
                        "Variables",
                        (),
                        {
                            "right_arm_start": right_arm_start,
                            "left_plane_initial": 0.0,
                            "left_plane_final": 0.0,
                            "right_plane_initial": 0.0,
                            "right_plane_final": 0.0,
                        },
                    )(),
                    "prescribed_motion": f"dms-motion-{right_arm_start:.2f}",
                    "left_plane_jerk": np.zeros(15),
                    "right_plane_jerk": np.zeros(15),
                    "final_twist_turns": final_twist,
                    "objective": objective,
                    "solver_status": "Solve_Succeeded",
                    "success": True,
                    "right_arm_start_node_index": int(round(right_arm_start / 0.02)),
                    "warm_start_primal": np.full(20, right_arm_start),
                    "warm_start_lam_x": np.full(20, 1.0),
                    "warm_start_lam_g": np.full(10, 2.0),
                },
            )()

    monkeypatch.setattr(
        "best_tilting_plane.gui.app.DirectMultipleShootingOptimizer.from_builder",
        lambda *_args, **_kwargs: FakeDmsOptimizer(),
    )

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app.root = FakeScheduler()
    app.result_var = FakeVar("")
    app.optimization_mode_var = FakeVar("Optimize 3D")
    app._auto_runner = FakeRunner()
    app._current_values = lambda: {
        "right_arm_start": 0.1,
        "left_plane_initial": 0.0,
        "left_plane_final": 0.0,
        "right_plane_initial": 0.0,
        "right_plane_final": 0.0,
        "contact_twist_turns_per_second": -0.6,
    }
    app._model_path = lambda: tmp_path / "reduced.bioMod"
    app._standard_optimization_configuration = lambda: SimulationConfiguration(
        final_time=1.0,
        integrator="rk4",
        rk4_step=0.005,
        contact_twist_rate=-0.6 * 2.0 * np.pi,
    )
    app._schedule_scan_figure = lambda **kwargs: None
    app._schedule_dms_jerk_diagnostic_figure = lambda **kwargs: None
    app._apply_optimized_values = lambda values, prescribed_motion=None, status_suffix=None: None
    (tmp_path / "optimization_cache.json").write_text(
        json.dumps(
            {
                "records": {
                    "optimize_2d__contact_twist_m0p6": {
                        "signature": app._optimization_cache_signature_for_mode("Optimize 2D"),
                        "values": {
                            "right_arm_start": 0.80,
                            "left_plane_initial": 0.0,
                            "left_plane_final": 0.0,
                            "right_plane_initial": 0.0,
                            "right_plane_final": 0.0,
                            "contact_twist_turns_per_second": -0.6,
                        },
                        "final_twist_turns": -0.90,
                        "solver_status": "Discrete_Sweep",
                    }
                },
                "progress_records": {},
            }
        ),
        encoding="utf-8",
    )

    app._optimize_strategy()

    assert calls == [(0.60, None), (0.62, 0.60), (0.64, 0.60)]


def test_schedule_dms_jerk_diagnostic_figure_can_recompute_missing_start_node_index() -> None:
    """The final DMS diagnostic should still work for checkpoint results rebuilt without the node index."""

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app.root = FakeScheduler()
    app.result_var = FakeVar("")
    forwarded: list[dict[str, object]] = []
    app._schedule_external_callback = lambda callback: forwarded.append({"callback": callback})

    class FakeOptimizer:
        interval_count = 50
        node_times = np.linspace(0.0, 1.0, 51)

        @staticmethod
        def _global_jerk_bounds(*, right_start_node_index: int):
            zeros = np.zeros(50, dtype=float)
            return (
                zeros.copy(),
                zeros.copy(),
                zeros.copy() + right_start_node_index,
                zeros.copy() + right_start_node_index,
            )

    result = SimpleNamespace(
        variables=SimpleNamespace(right_arm_start=0.30),
        left_plane_jerk=np.zeros(15),
        right_plane_jerk=np.zeros(15),
    )

    app._schedule_dms_jerk_diagnostic_figure(optimizer=FakeOptimizer(), result=result)

    assert len(forwarded) == 1


def test_report_callback_exception_updates_result_message() -> None:
    """Tkinter callback failures should be surfaced in the GUI status line."""

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app.result_var = FakeVar("")

    try:
        raise RuntimeError("callback boom")
    except RuntimeError as error:
        trace = error.__traceback__
        app._report_callback_exception(RuntimeError, error, trace)

    assert app.result_var.get() == "Erreur GUI: callback boom"


def test_optimize_strategy_can_ignore_cached_dms_solution_and_restart_sweep(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The force-recompute option should ignore both cached DMS optima and partial checkpoints."""

    calls: list[tuple[float, float | None]] = []

    class FakeDmsOptimizer:
        def candidate_start_times(self):
            return np.array([0.10, 0.12], dtype=float)

        def solve_fixed_start(self, initial_guess, *, right_arm_start, previous_result=None, **_kwargs):
            assert initial_guess.right_arm_start == 0.1
            calls.append(
                (
                    float(right_arm_start),
                    None if previous_result is None else float(previous_result.variables.right_arm_start),
                )
            )
            return type(
                "DmsResult",
                (),
                {
                    "variables": type(
                        "Variables",
                        (),
                        {
                            "right_arm_start": right_arm_start,
                            "left_plane_initial": 0.0,
                            "left_plane_final": 0.0,
                            "right_plane_initial": 0.0,
                            "right_plane_final": 0.0,
                        },
                    )(),
                    "prescribed_motion": f"dms-motion-{right_arm_start:.2f}",
                    "left_plane_jerk": np.zeros(15),
                    "right_plane_jerk": np.zeros(15),
                    "final_twist_turns": -0.40 if np.isclose(right_arm_start, 0.10) else -0.63,
                    "objective": -0.39 if np.isclose(right_arm_start, 0.10) else -0.62,
                    "solver_status": "Solve_Succeeded",
                    "success": True,
                    "warm_start_primal": np.full(20, right_arm_start),
                    "warm_start_lam_x": np.full(20, 1.0),
                    "warm_start_lam_g": np.full(10, 2.0),
                },
            )()

    monkeypatch.setattr(
        "best_tilting_plane.gui.app.DirectMultipleShootingOptimizer.from_builder",
        lambda *_args, **_kwargs: FakeDmsOptimizer(),
    )

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app.root = FakeScheduler()
    app.result_var = FakeVar("")
    app.optimization_mode_var = FakeVar("Optimize DMS")
    app.ignore_optimization_cache_var = FakeVar(True)
    app._auto_runner = FakeRunner()
    app._current_values = lambda: {
        "right_arm_start": 0.1,
        "left_plane_initial": 0.0,
        "left_plane_final": 0.0,
        "right_plane_initial": 0.0,
        "right_plane_final": 0.0,
    }
    app._model_path = lambda: tmp_path / "reduced.bioMod"
    app._standard_optimization_configuration = lambda: SimulationConfiguration(
        final_time=1.0,
        integrator="rk4",
        rk4_step=0.005,
    )
    app._show_dms_sweep_figure = lambda **kwargs: None
    applied: list[tuple[dict[str, float], object, str | None]] = []
    app._apply_optimized_values = lambda values, prescribed_motion=None, status_suffix=None: applied.append(
        (dict(values), prescribed_motion, status_suffix)
    )
    (tmp_path / "optimization_cache.json").write_text(
        json.dumps(
            {
                "records": {
                    "optimize_dms": {
                        "signature": app._optimization_cache_signature(),
                        "in_progress": True,
                        "values": {
                            "right_arm_start": 0.10,
                            "left_plane_initial": 0.0,
                            "left_plane_final": 0.0,
                            "right_plane_initial": 0.0,
                            "right_plane_final": 0.0,
                        },
                        "left_plane_jerk": [0.0] * 15,
                        "right_plane_jerk": [0.0] * 15,
                        "scan_start_times": [0.10],
                        "scan_final_twist_turns": [-0.40],
                        "scan_objective_values": [-0.39],
                        "scan_success_mask": [True],
                        "last_completed_index": 0,
                        "last_warm_start_primal": [0.10] * 20,
                        "last_warm_start_lam_x": [1.0] * 20,
                        "last_warm_start_lam_g": [2.0] * 10,
                        "final_twist_turns": -0.40,
                        "solver_status": "Solve_Succeeded",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    app._optimize_strategy()

    assert calls == [(0.10, None), (0.12, 0.10)]
    assert applied == [
        (
            {
                "first_arm_start": 0.0,
                "right_arm_start": 0.12,
                "left_plane_initial": 0.0,
                "left_plane_final": 0.0,
                "right_plane_initial": 0.0,
                "right_plane_final": 0.0,
            },
            "dms-motion-0.12",
            "optimum DMS bras 2: -0.63 tours (Solve_Succeeded)",
        )
    ]


def test_optimize_strategy_writes_cache_after_dms(monkeypatch, tmp_path: Path) -> None:
    """A newly optimized DMS solution should be persisted for later GUI sessions."""

    class FakeDmsOptimizer:
        def candidate_start_times(self):
            return np.array([0.10, 0.28, 0.30])

        def solve_fixed_start(self, initial_guess, *, right_arm_start, previous_result=None, **_kwargs):
            assert initial_guess.right_arm_start == 0.1
            return type(
                "DmsResult",
                (),
                {
                    "variables": type(
                        "Variables",
                        (),
                        {
                            "right_arm_start": right_arm_start,
                            "left_plane_initial": 0.0,
                            "left_plane_final": 0.0,
                            "right_plane_initial": 0.0,
                            "right_plane_final": 0.0,
                        },
                    )(),
                    "prescribed_motion": "dms-motion" if np.isclose(right_arm_start, 0.28) else f"dms-motion-{right_arm_start:.2f}",
                    "left_plane_jerk": np.zeros(15),
                    "right_plane_jerk": np.zeros(15),
                    "final_twist_turns": -0.40 if np.isclose(right_arm_start, 0.10) else -0.63 if np.isclose(right_arm_start, 0.28) else -0.61,
                    "objective": -0.39 if np.isclose(right_arm_start, 0.10) else -0.62 if np.isclose(right_arm_start, 0.28) else -0.60,
                    "solver_status": "Solve_Succeeded",
                    "success": True,
                    "warm_start_primal": np.full(20, right_arm_start),
                    "warm_start_lam_x": np.full(20, 1.0),
                    "warm_start_lam_g": np.full(10, 2.0),
                },
            )()

    monkeypatch.setattr(
        "best_tilting_plane.gui.app.DirectMultipleShootingOptimizer.from_builder",
        lambda *_args, **_kwargs: FakeDmsOptimizer(),
    )

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app.root = FakeScheduler()
    app.result_var = FakeVar("")
    app.optimization_mode_var = FakeVar("Optimize DMS")
    app._auto_runner = FakeRunner()
    app._current_values = lambda: {
        "right_arm_start": 0.1,
        "left_plane_initial": 0.0,
        "left_plane_final": 0.0,
        "right_plane_initial": 0.0,
        "right_plane_final": 0.0,
    }
    app._model_path = lambda: tmp_path / "reduced.bioMod"
    app._standard_optimization_configuration = lambda: SimulationConfiguration(
        final_time=1.0,
        integrator="rk4",
        rk4_step=0.005,
    )
    app._show_dms_sweep_figure = lambda **kwargs: None
    app._apply_optimized_values = lambda values, prescribed_motion=None, status_suffix=None: None

    app._optimize_strategy()

    stored = json.loads((tmp_path / "optimization_cache.json").read_text(encoding="utf-8"))
    record = stored["records"]["optimize_dms"]
    assert record["values"] == {
        "contact_twist_turns_per_second": 0.0,
        "first_arm_start": 0.0,
        "right_arm_start": 0.28,
        "left_plane_initial": 0.0,
        "left_plane_final": 0.0,
        "right_plane_initial": 0.0,
        "right_plane_final": 0.0,
    }
    assert record["left_plane_jerk"] == [0.0] * 15
    assert record["right_plane_jerk"] == [0.0] * 15
    assert record["scan_start_times"] == [0.10, 0.28, 0.30]
    assert record["scan_final_twist_turns"] == [-0.40, -0.63, -0.61]
    assert record["scan_objective_values"] == [-0.39, -0.62, -0.60]
    assert record["scan_success_mask"] == [True, True, True]
    assert record["final_twist_turns"] == -0.63
    assert record["solver_status"] == "Solve_Succeeded"
    assert stored.get("progress_records", {}) == {}


def test_store_cached_dms_progress_preserves_completed_dms_solution(tmp_path: Path) -> None:
    """A resumable DMS checkpoint should not overwrite the last completed optimum."""

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app.optimization_mode_var = FakeVar("Optimize DMS")
    app._model_path = lambda: tmp_path / "reduced.bioMod"
    app._standard_optimization_configuration = lambda: SimulationConfiguration(
        final_time=1.0,
        integrator="rk4",
        rk4_step=0.005,
    )

    cache_path = tmp_path / "optimization_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "records": {
                    "optimize_dms": {
                        "signature": app._optimization_cache_signature(),
                        "in_progress": False,
                        "values": {
                            "right_arm_start": 0.28,
                            "left_plane_initial": 0.0,
                            "left_plane_final": 0.0,
                            "right_plane_initial": 0.0,
                            "right_plane_final": 0.0,
                        },
                        "left_plane_jerk": [0.0] * 15,
                        "right_plane_jerk": [0.0] * 15,
                        "scan_start_times": [0.10, 0.12, 0.28],
                        "scan_final_twist_turns": [-0.40, -0.55, -0.63],
                        "scan_objective_values": [-0.39, -0.54, -0.62],
                        "scan_success_mask": [True, True, True],
                        "final_twist_turns": -0.63,
                        "solver_status": "Solve_Succeeded",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    app._store_cached_dms_progress(
        {
            "right_arm_start": 0.10,
            "left_plane_initial": 0.0,
            "left_plane_final": 0.0,
            "right_plane_initial": 0.0,
            "right_plane_final": 0.0,
        },
        left_plane_jerk=np.zeros(15),
        right_plane_jerk=np.zeros(15),
        scan_start_times=np.array([0.10], dtype=float),
        scan_final_twist_turns=np.array([-0.40], dtype=float),
        scan_objective_values=np.array([-0.39], dtype=float),
        scan_success_mask=np.array([True], dtype=bool),
        last_completed_index=0,
        last_warm_start_primal=np.full(20, 0.10),
        last_warm_start_lam_x=np.full(20, 1.0),
        last_warm_start_lam_g=np.full(10, 2.0),
        final_twist_turns=-0.40,
        solver_status="Solve_Succeeded",
    )

    stored = json.loads(cache_path.read_text(encoding="utf-8"))
    assert stored["records"]["optimize_dms"]["values"]["right_arm_start"] == 0.28
    assert stored["records"]["optimize_dms"]["final_twist_turns"] == -0.63
    assert stored["progress_records"]["optimize_dms"]["in_progress"] is True
    assert stored["progress_records"]["optimize_dms"]["values"]["right_arm_start"] == 0.10


def test_optimize_strategy_resumes_dms_from_partial_checkpoint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A partial DMS cache entry should resume from the next start time instead of restarting at the beginning."""

    calls: list[tuple[float, float | None]] = []

    class FakeDmsOptimizer:
        def candidate_start_times(self):
            return np.array([0.10, 0.12, 0.14])

        def solve_fixed_start(self, initial_guess, *, right_arm_start, previous_result=None, **_kwargs):
            assert initial_guess.right_arm_start == 0.1
            calls.append((right_arm_start, None if previous_result is None else float(previous_result.warm_start_primal[0])))
            return type(
                "DmsResult",
                (),
                {
                    "variables": type(
                        "Variables",
                        (),
                        {
                            "right_arm_start": right_arm_start,
                            "left_plane_initial": 0.0,
                            "left_plane_final": 0.0,
                            "right_plane_initial": 0.0,
                            "right_plane_final": 0.0,
                        },
                    )(),
                    "prescribed_motion": f"dms-motion-{right_arm_start:.2f}",
                    "left_plane_jerk": np.zeros(15),
                    "right_plane_jerk": np.zeros(15),
                    "final_twist_turns": -0.55 if np.isclose(right_arm_start, 0.12) else -0.63,
                    "objective": -0.54 if np.isclose(right_arm_start, 0.12) else -0.62,
                    "solver_status": "Solve_Succeeded",
                    "success": True,
                    "warm_start_primal": np.full(20, right_arm_start),
                    "warm_start_lam_x": np.full(20, 1.0),
                    "warm_start_lam_g": np.full(10, 2.0),
                },
            )()

    monkeypatch.setattr(
        "best_tilting_plane.gui.app.DirectMultipleShootingOptimizer.from_builder",
        lambda *_args, **_kwargs: FakeDmsOptimizer(),
    )

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app.root = FakeScheduler()
    app.result_var = FakeVar("")
    app.optimization_mode_var = FakeVar("Optimize DMS")
    app._auto_runner = FakeRunner()
    app._current_values = lambda: {
        "right_arm_start": 0.1,
        "left_plane_initial": 0.0,
        "left_plane_final": 0.0,
        "right_plane_initial": 0.0,
        "right_plane_final": 0.0,
    }
    app._model_path = lambda: tmp_path / "reduced.bioMod"
    app._standard_optimization_configuration = lambda: SimulationConfiguration(
        final_time=1.0,
        integrator="rk4",
        rk4_step=0.005,
    )
    app._show_dms_sweep_figure = lambda **kwargs: None
    applied: list[tuple[dict[str, float], object, str | None]] = []
    app._apply_optimized_values = lambda values, prescribed_motion=None, status_suffix=None: applied.append(
        (dict(values), prescribed_motion, status_suffix)
    )
    (tmp_path / "optimization_cache.json").write_text(
        json.dumps(
            {
                "records": {
                    "optimize_dms": {
                        "signature": app._optimization_cache_signature(),
                        "in_progress": True,
                        "values": {
                            "right_arm_start": 0.10,
                            "left_plane_initial": 0.0,
                            "left_plane_final": 0.0,
                            "right_plane_initial": 0.0,
                            "right_plane_final": 0.0,
                        },
                        "left_plane_jerk": [0.0] * 15,
                        "right_plane_jerk": [0.0] * 15,
                        "scan_start_times": [0.10],
                        "scan_final_twist_turns": [-0.40],
                        "scan_objective_values": [-0.39],
                        "scan_success_mask": [True],
                        "last_completed_index": 0,
                        "last_warm_start_primal": [0.10] * 20,
                        "last_warm_start_lam_x": [1.0] * 20,
                        "last_warm_start_lam_g": [2.0] * 10,
                        "final_twist_turns": -0.40,
                        "solver_status": "Solve_Succeeded",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    app._optimize_strategy()

    assert calls == [(0.12, 0.10), (0.14, 0.12)]
    assert applied == [
        (
            {
                "first_arm_start": 0.0,
                "right_arm_start": 0.14,
                "left_plane_initial": 0.0,
                "left_plane_final": 0.0,
                "right_plane_initial": 0.0,
                "right_plane_final": 0.0,
            },
            "dms-motion-0.14",
            "optimum DMS bras 2: -0.63 tours (Solve_Succeeded)",
        )
    ]
    stored = json.loads((tmp_path / "optimization_cache.json").read_text(encoding="utf-8"))
    assert stored["records"]["optimize_dms"]["in_progress"] is False
    assert stored.get("progress_records", {}) == {}
