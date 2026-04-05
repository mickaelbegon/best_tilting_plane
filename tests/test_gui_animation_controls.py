"""Tests for GUI animation playback controls."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from best_tilting_plane.gui.app import (
    ANIMATION_MODE_OPTIONS,
    ANIMATION_REFERENCE_OPTIONS,
    DEFAULT_CAMERA_AZIMUTH_DEG,
    DEFAULT_CAMERA_ELEVATION_DEG,
    ROOT_INITIAL_OPTIONS,
    TOP_VIEW_CAMERA_AZIMUTH_DEG,
    TOP_VIEW_CAMERA_ELEVATION_DEG,
    BestTiltingPlaneApp,
)
from best_tilting_plane.simulation import SimulationConfiguration


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
    assert app.root_initial_mode.get() == ROOT_INITIAL_OPTIONS[1]
    assert app.animation_mode_var.get() == ANIMATION_MODE_OPTIONS[1]


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


def test_show_first_arm_jerk_comparison_uses_external_helper(monkeypatch) -> None:
    """The dedicated GUI button should open the external comparison window with the GUI values."""

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "best_tilting_plane.gui.app.show_first_arm_piecewise_constant_comparison",
        lambda variables, **kwargs: captured.update({"variables": variables, "kwargs": kwargs}),
    )

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app._current_values = lambda: {
        "right_arm_start": 0.1,
        "left_plane_initial": -15.0,
        "left_plane_final": 5.0,
        "right_plane_initial": 10.0,
        "right_plane_final": 20.0,
    }
    app._standard_optimization_configuration = lambda: SimulationConfiguration(
        final_time=1.0,
        integrator="rk4",
        rk4_step=0.005,
    )

    app._show_first_arm_jerk_comparison()

    assert captured["variables"].right_arm_start == 0.1
    assert captured["kwargs"] == {"total_time": 1.0, "jerk_step": 0.02, "sample_step": 0.005}


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
    app._apply_optimized_values = lambda values, status_suffix=None: applied.append(
        (dict(values), status_suffix)
    )

    app._optimize_strategy()

    assert app._auto_runner.cancelled
    assert applied == [
        (
            {
                "right_arm_start": 0.35,
                "left_plane_initial": 0.0,
                "left_plane_final": 0.0,
                "right_plane_initial": 0.0,
                "right_plane_final": 0.0,
            },
            "optimum IPOPT: -0.75 tours (Solve_Succeeded)",
        )
    ]


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
                    "optimize_2d": {
                        "signature": app._optimization_cache_signature(),
                        "values": {
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
    app._apply_optimized_values = lambda values, status_suffix=None: applied.append(
        (dict(values), status_suffix)
    )

    app._optimize_strategy()

    assert app._auto_runner.cancelled
    assert applied == [
        (
            {
                "right_arm_start": 0.42,
                "left_plane_initial": 0.0,
                "left_plane_final": 0.0,
                "right_plane_initial": 0.0,
                "right_plane_final": 0.0,
            },
            "optimum charge depuis le cache",
        )
    ]


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
    app._apply_optimized_values = lambda values, status_suffix=None: applied.append(
        (dict(values), status_suffix)
    )

    app._optimize_strategy()

    stored = json.loads((tmp_path / "optimization_cache.json").read_text(encoding="utf-8"))
    record = stored["records"]["optimize_2d"]
    assert record["values"] == {
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
                "right_arm_start": 0.35,
                "left_plane_initial": 0.0,
                "left_plane_final": 0.0,
                "right_plane_initial": 0.0,
                "right_plane_final": 0.0,
            },
            "optimum IPOPT: -0.75 tours (Solve_Succeeded)",
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
    assert motion.right_arm_start == 0.28
    assert motion.left_plane.jerks.shape == (15,)
    assert motion.right_plane.jerks.shape == (15,)
    assert final_twist_turns == -0.63
    assert solver_status == "Solve_Succeeded"
    assert scan_data is not None
    assert scan_data["start_times"] == [0.10, 0.12, 0.14]


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

    assert app._auto_runner.cancelled
    assert applied[0][0] == {
        "right_arm_start": 0.28,
        "left_plane_initial": 0.0,
        "left_plane_final": 0.0,
        "right_plane_initial": 0.0,
        "right_plane_final": 0.0,
    }
    assert applied[0][1] is not None
    assert applied[0][2] == "optimum DMS charge depuis le cache: -0.63 tours (Solve_Succeeded)"
    assert shown == [
        {
            "start_times": [0.10, 0.12, 0.28],
            "final_twist_turns": [-0.40, -0.55, -0.63],
            "objective_values": [-0.39, -0.54, -0.62],
            "success_mask": [True, True, True],
            "best_start_time": 0.28,
        }
    ]


def test_optimize_strategy_runs_dms_and_replays_the_optimized_motion(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The DMS mode should push both slider values and the optimized motion."""

    class FakeDmsOptimizer:
        def candidate_start_times(self):
            return np.array([0.10, 0.28, 0.30])

        def solve_fixed_start(self, initial_guess, *, right_arm_start, previous_result=None, **_kwargs):
            assert initial_guess.right_arm_start == 0.1
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
    shown: list[dict[str, object]] = []
    app._show_dms_sweep_figure = lambda **kwargs: shown.append(dict(kwargs))
    applied: list[tuple[dict[str, float], object, str | None]] = []
    app._apply_optimized_values = lambda values, prescribed_motion=None, status_suffix=None: applied.append(
        (dict(values), prescribed_motion, status_suffix)
    )

    app._optimize_strategy()

    assert app._auto_runner.cancelled
    assert applied == [
        (
            {
                "right_arm_start": 0.28,
                "left_plane_initial": 0.0,
                "left_plane_final": 0.0,
                "right_plane_initial": 0.0,
                "right_plane_final": 0.0,
            },
            "dms-motion",
            "optimum DMS: -0.63 tours (Solve_Succeeded)",
        )
    ]
    assert len(shown) == 1
    np.testing.assert_allclose(shown[0]["start_times"], [0.10, 0.28, 0.30])
    np.testing.assert_allclose(shown[0]["final_twist_turns"], [-0.40, -0.63, -0.61])
    np.testing.assert_allclose(shown[0]["objective_values"], [-0.39, -0.62, -0.60])
    np.testing.assert_array_equal(shown[0]["success_mask"], [True, True, True])
    assert shown[0]["best_start_time"] == 0.28


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
                "right_arm_start": 0.14,
                "left_plane_initial": 0.0,
                "left_plane_final": 0.0,
                "right_plane_initial": 0.0,
                "right_plane_final": 0.0,
            },
            "dms-motion-0.14",
            "optimum DMS: -0.63 tours (Solve_Succeeded)",
        )
    ]
    stored = json.loads((tmp_path / "optimization_cache.json").read_text(encoding="utf-8"))
    assert stored["records"]["optimize_dms"]["in_progress"] is False
