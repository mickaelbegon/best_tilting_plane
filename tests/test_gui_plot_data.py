"""Tests for GUI plot-data selection."""

from __future__ import annotations

import json

import numpy as np

from best_tilting_plane.gui.app import BestTiltingPlaneApp, ROOT_INITIAL_OPTIONS
from best_tilting_plane.simulation import SimulationConfiguration


class _FakeVar:
    """Tiny stand-in exposing the Tk variable `get` API used by the GUI."""

    def __init__(self, value: str) -> None:
        self._value = value

    def get(self) -> str:
        """Return the stored value."""

        return self._value


class _FakeAxis:
    """Very small axis stub storing the drawn lines and bounds."""

    def __init__(self) -> None:
        self.plot_calls: list[dict[str, object]] = []
        self.axhline_calls: list[dict[str, object]] = []

    def clear(self) -> None:
        """Mirror the matplotlib API."""

    def plot(self, x, y, **kwargs) -> None:
        """Record one plotted curve."""

        self.plot_calls.append({"x": np.asarray(x), "y": np.asarray(y), "kwargs": dict(kwargs)})

    def axhline(self, y, **kwargs) -> None:
        """Record one horizontal bound."""

        self.axhline_calls.append({"y": float(y), "kwargs": dict(kwargs)})

    def legend(self, **_kwargs) -> None:
        """Mirror the matplotlib API."""

    def set_xlabel(self, _label: str) -> None:
        """Mirror the matplotlib API."""

    def set_ylabel(self, _label: str) -> None:
        """Mirror the matplotlib API."""

    def set_title(self, _title: str) -> None:
        """Mirror the matplotlib API."""

    def grid(self, *_args, **_kwargs) -> None:
        """Mirror the matplotlib API."""


class _FakeCanvas:
    """Small canvas stub recording refreshes."""

    def __init__(self) -> None:
        self.draw_idle_calls = 0

    def draw_idle(self) -> None:
        """Record one redraw."""

        self.draw_idle_calls += 1


def _build_app_for_plotting(
    *,
    plot_x: str = "Temps",
    plot_y: str = "Twist",
    root_mode: str = ROOT_INITIAL_OPTIONS[0],
) -> BestTiltingPlaneApp:
    """Create a minimal app instance without constructing the Tk window."""

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app.plot_x_var = _FakeVar(plot_x)
    app.plot_y_var = _FakeVar(plot_y)
    app.plot_mode_var = _FakeVar("Courbe")
    app.root_initial_mode = _FakeVar(root_mode)
    app._animation_playing = False
    app._animation_after_id = None
    app._animation_frame_index = 0
    app._visualization_data = {
        "result": type(
            "Result",
            (),
            {
                "time": np.array([0.0, 0.5, 1.0], dtype=float),
                "q": np.array(
                    [
                        [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, -0.2, -3.0, 0.2, 3.0],
                        [0.0, 0.0, 0.0, 0.4, 0.5, 0.6, -0.1, -1.5, 0.1, 1.5],
                        [0.0, 0.0, 0.0, 0.9, 1.0, 1.1, 0.0, 0.0, 0.0, 0.0],
                    ],
                    dtype=float,
                ),
                "qdot": np.array(
                    [
                        [0.0, 0.0, 0.0, 0.7, 0.8, 0.9, -0.4, -6.0, 0.4, 6.0],
                        [0.0, 0.0, 0.0, 1.0, 1.1, 1.2, -0.2, -3.0, 0.2, 3.0],
                        [0.0, 0.0, 0.0, 1.3, 1.4, 1.5, 0.0, 0.0, 0.0, 0.0],
                    ],
                    dtype=float,
                ),
            },
        )(),
        "display_q": np.array(
            [
                [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, -0.2, -3.0, 0.2, 3.0],
                [0.0, 0.0, 0.0, 0.4, 0.5, 0.6, -0.1, -1.5, 0.1, 1.5],
                [0.0, 0.0, 0.0, 0.9, 1.0, 1.1, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        "display_qdot": np.array(
            [
                [0.0, 0.0, 0.0, 0.7, 0.8, 0.9, -0.4, -6.0, 0.4, 6.0],
                [0.0, 0.0, 0.0, 1.0, 1.1, 1.2, -0.2, -3.0, 0.2, 3.0],
                [0.0, 0.0, 0.0, 1.3, 1.4, 1.5, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        "deviations": {
            "left": np.array([0.0, 0.1, 0.2], dtype=float),
            "right": np.array([0.0, -0.1, -0.2], dtype=float),
        },
        "observables": {
            "angular_momentum": np.array(
                [
                    [1.0, 2.0, 2.0],
                    [3.0, 4.0, 0.0],
                    [5.0, 12.0, 0.0],
                ],
                dtype=float,
            )
        },
        "frames": {
            "pelvis": {
                "axes": np.repeat(np.eye(3)[None, :, :], 3, axis=0),
            }
        },
        "trajectories": {
            "pelvis_origin": np.array(
                [[1.0, 2.0, 0.0], [1.0, 2.1, 0.0], [1.0, 2.2, 0.0]],
                dtype=float,
            ),
            "shoulder_left": np.array(
                [[0.8, 2.0, 0.0], [0.8, 2.1, 0.0], [0.8, 2.2, 0.0]],
                dtype=float,
            ),
            "elbow_left": np.array(
                [[0.7, 2.1, 0.0], [0.7, 2.3, 0.0], [0.7, 2.4, 0.0]],
                dtype=float,
            ),
            "wrist_left": np.array(
                [[0.6, 2.2, 0.0], [0.6, 2.4, 0.0], [0.6, 2.6, 0.0]],
                dtype=float,
            ),
            "hand_left": np.array(
                [[0.5, 2.3, 0.0], [0.5, 2.5, 0.0], [0.5, 2.7, 0.0]],
                dtype=float,
            ),
            "shoulder_right": np.array(
                [[1.2, 2.0, 0.0], [1.2, 2.1, 0.0], [1.2, 2.2, 0.0]],
                dtype=float,
            ),
            "elbow_right": np.array(
                [[1.3, 2.1, 0.0], [1.3, 2.3, 0.0], [1.3, 2.4, 0.0]],
                dtype=float,
            ),
            "wrist_right": np.array(
                [[1.4, 2.2, 0.0], [1.4, 2.4, 0.0], [1.4, 2.6, 0.0]],
                dtype=float,
            ),
            "hand_right": np.array(
                [[1.5, 2.3, 0.0], [1.5, 2.5, 0.0], [1.5, 2.7, 0.0]],
                dtype=float,
            ),
        },
    }
    return app


def test_plot_data_returns_twist_against_somersault() -> None:
    """The standard plot mode should still expose root-angle curves."""

    app = _build_app_for_plotting(plot_x="Somersault", plot_y="Twist")

    x_data, y_data, x_label, y_label, title, curve_labels = app._plot_data()

    np.testing.assert_allclose(x_data, np.rad2deg(np.array([0.1, 0.4, 0.9])))
    np.testing.assert_allclose(y_data, np.rad2deg(np.array([0.3, 0.6, 1.1])))
    assert x_label == "Somersault (deg)"
    assert y_label == "Twist (deg)"
    assert title == "Twist en fonction de somersault"
    assert curve_labels is None


def test_plot_data_can_return_the_four_arm_kinematic_curves() -> None:
    """The plot selector should expose the four arm DoFs as one 4-curve figure."""

    app = _build_app_for_plotting(plot_x="Temps", plot_y="Cinematique bras")

    x_data, y_data, x_label, y_label, title, curve_labels = app._plot_data()

    np.testing.assert_allclose(x_data, np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(
        y_data,
        np.rad2deg(
            np.array(
                [
                    [-0.2, -3.0, 0.2, 3.0],
                    [-0.1, -1.5, 0.1, 1.5],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )
        ),
    )
    assert x_label == "Temps (s)"
    assert y_label == "Angles bras (deg)"
    assert title == "Cinematique bras en fonction de temps"
    assert curve_labels == (
        "Plan bras gauche",
        "Elevation bras gauche",
        "Plan bras droit",
        "Elevation bras droit",
    )


def test_plot_data_can_return_the_four_arm_velocity_curves() -> None:
    """The plot selector should expose the four arm DoF velocities as one 4-curve figure."""

    app = _build_app_for_plotting(plot_x="Temps", plot_y="Vitesses bras")

    x_data, y_data, x_label, y_label, title, curve_labels = app._plot_data()

    np.testing.assert_allclose(x_data, np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(
        y_data,
        np.rad2deg(
            np.array(
                [
                    [-0.4, -6.0, 0.4, 6.0],
                    [-0.2, -3.0, 0.2, 3.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )
        ),
    )
    assert x_label == "Temps (s)"
    assert y_label == "Vitesses bras (deg/s)"
    assert title == "Vitesses bras en fonction de temps"
    assert curve_labels == (
        "Plan bras gauche",
        "Elevation bras gauche",
        "Plan bras droit",
        "Elevation bras droit",
    )


def test_plot_data_can_use_twist_as_the_horizontal_axis() -> None:
    """The x-axis selector should expose the root twist angle in degrees."""

    app = _build_app_for_plotting(plot_x="Vrille", plot_y="Tilt")

    x_data, y_data, x_label, y_label, title, curve_labels = app._plot_data()

    np.testing.assert_allclose(x_data, np.rad2deg(np.array([0.3, 0.6, 1.1])))
    np.testing.assert_allclose(y_data, np.rad2deg(np.array([0.2, 0.5, 1.0])))
    assert x_label == "Vrille (deg)"
    assert y_label == "Tilt (deg)"
    assert title == "Tilt en fonction de vrille"
    assert curve_labels is None


def test_top_view_plot_data_returns_relative_arm_trajectories_and_current_frame() -> None:
    """The top-view mode should expose pelvis-relative arm trajectories."""

    app = _build_app_for_plotting()
    app._animation_frame_index = 2

    top_view, frame_index = app._top_view_plot_data()

    np.testing.assert_allclose(
        top_view["hand_left"], np.array([[-0.5, 0.3], [-0.5, 0.4], [-0.5, 0.5]])
    )
    np.testing.assert_allclose(
        top_view["hand_right"], np.array([[0.5, 0.3], [0.5, 0.4], [0.5, 0.5]])
    )
    assert frame_index == 2


def test_current_plot_frame_index_returns_last_drawn_frame_while_playing() -> None:
    """While the animation is playing, the highlighted 2D frame should match the displayed 3D frame."""

    app = _build_app_for_plotting()
    app._animation_playing = True
    app._animation_after_id = 123
    app._animation_frame_index = 2

    assert app._current_plot_frame_index() == 1


def test_top_view_plot_data_can_neutralize_root_orientation_for_arm_visualization() -> None:
    """The `q racine(0)=0` mode should zero out the six root coordinates for display."""

    app = _build_app_for_plotting(root_mode=ROOT_INITIAL_OPTIONS[0])
    app._visualization_data["result"].q = np.array(
        [
            [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 11.0, 12.0],
            [4.0, 5.0, 6.0, 0.4, 0.5, 0.6, 13.0, 14.0],
            [7.0, 8.0, 9.0, 0.9, 1.0, 1.1, 15.0, 16.0],
        ],
        dtype=float,
    )

    display_q = app._display_q_history(app._visualization_data["result"])

    np.testing.assert_allclose(display_q[:, :6], 0.0)
    np.testing.assert_allclose(display_q[:, 6:], np.array([[11.0, 12.0], [13.0, 14.0], [15.0, 16.0]]))


def test_scan_plot_datasets_returns_current_mode_then_other_cached_mode(tmp_path) -> None:
    """The embedded scan figure should prioritize the current optimization mode and overlay the other one."""

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app.optimization_mode_var = _FakeVar("Optimize DMS")
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
                        "scan_start_times": [0.16, 0.20],
                        "scan_final_twist_turns": [-0.30, -0.50],
                        "scan_objective_values": [-0.29, -0.49],
                    },
                    "optimize_dms": {
                        "signature": app._optimization_cache_signature_for_mode("Optimize DMS"),
                        "values": {
                            "right_arm_start": 0.28,
                            "left_plane_initial": 0.0,
                            "left_plane_final": 0.0,
                            "right_plane_initial": 0.0,
                            "right_plane_final": 0.0,
                        },
                        "scan_start_times": [0.16, 0.18, 0.28],
                        "scan_final_twist_turns": [-0.40, -0.55, -0.63],
                        "scan_objective_values": [-0.39, -0.54, -0.62],
                        "scan_success_mask": [True, True, True],
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    datasets = app._scan_plot_datasets()

    assert [dataset["mode"] for dataset in datasets] == ["Optimize DMS", "Optimize 2D"]
    assert datasets[0]["best_start_time"] == 0.28
    assert datasets[1]["best_start_time"] == 0.20


def test_refresh_plot_adds_arm_angle_bounds_when_plotting_arm_kinematics() -> None:
    """The arm-kinematics figure should display the validated angular bounds for all four DoFs."""

    app = _build_app_for_plotting(plot_x="Temps", plot_y="Cinematique bras")
    app._plot_axis = _FakeAxis()
    app._plot_canvas = _FakeCanvas()

    app._refresh_plot()

    assert len(app._plot_axis.plot_calls) == 4
    assert len(app._plot_axis.axhline_calls) == 8
    np.testing.assert_allclose(
        [call["y"] for call in app._plot_axis.axhline_calls],
        [-135.0, 0.0, -180.0, 0.0, 0.0, 135.0, 0.0, 180.0],
    )
    assert app._plot_canvas.draw_idle_calls == 1
