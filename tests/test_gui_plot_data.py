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
        self.axvline_calls: list[dict[str, object]] = []
        self.aspect_calls: list[tuple[object, ...]] = []

    def clear(self) -> None:
        """Mirror the matplotlib API."""

    def plot(self, x, y, **kwargs) -> None:
        """Record one plotted curve."""

        self.plot_calls.append({"x": np.asarray(x), "y": np.asarray(y), "kwargs": dict(kwargs)})

    def step(self, x, y, **kwargs) -> None:
        """Record one stepped curve."""

        self.plot_calls.append({"x": np.asarray(x), "y": np.asarray(y), "kwargs": dict(kwargs), "step": True})

    def axhline(self, y, **kwargs) -> None:
        """Record one horizontal bound."""

        self.axhline_calls.append({"y": float(y), "kwargs": dict(kwargs)})

    def axvline(self, x, **kwargs):
        """Record one vertical indicator line."""

        call = {"x": float(x), "kwargs": dict(kwargs), "xdata": [float(x), float(x)]}
        self.axvline_calls.append(call)

        class _Line:
            def __init__(self, payload) -> None:
                self.payload = payload

            def set_xdata(self, values) -> None:
                self.payload["xdata"] = list(values)

        return _Line(call)

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

    def set_aspect(self, *args, **_kwargs) -> None:
        """Record aspect-ratio changes."""

        self.aspect_calls.append(args)


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
                "qddot": np.array(
                    [
                        [0.0, 0.0, 0.0, 1.7, 1.8, 1.9, -0.8, -12.0, 0.8, 12.0],
                        [0.0, 0.0, 0.0, 2.0, 2.1, 2.2, -0.4, -6.0, 0.4, 6.0],
                        [0.0, 0.0, 0.0, 2.3, 2.4, 2.5, 0.0, 0.0, 0.0, 0.0],
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
            ),
            "angular_momentum_groups": np.array(
                [
                    [[1.0, 10.0, 100.0], [2.0, 20.0, 200.0], [3.0, 30.0, 300.0]],
                    [[4.0, 40.0, 110.0], [5.0, 50.0, 210.0], [6.0, 60.0, 310.0]],
                    [[7.0, 70.0, 120.0], [8.0, 80.0, 220.0], [9.0, 90.0, 320.0]],
                ],
                dtype=float,
            ),
            "shoulder_torques": np.array(
                [
                    [[1.0, 0.1, 0.01, 0.001], [2.0, 0.2, 0.02, 0.002], [3.0, 0.3, 0.03, 0.003], [4.0, 0.4, 0.04, 0.004]],
                    [[5.0, 0.5, 0.05, 0.005], [6.0, 0.6, 0.06, 0.006], [7.0, 0.7, 0.07, 0.007], [8.0, 0.8, 0.08, 0.008]],
                    [[9.0, 0.9, 0.09, 0.009], [10.0, 1.0, 0.10, 0.010], [11.0, 1.1, 0.11, 0.011], [12.0, 1.2, 0.12, 0.012]],
                ],
                dtype=float,
            ),
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


def test_plot_data_can_return_the_four_arm_acceleration_curves() -> None:
    """The plot selector should expose the four arm DoF accelerations as one 4-curve figure."""

    app = _build_app_for_plotting(plot_x="Temps", plot_y="Accelerations bras")

    x_data, y_data, x_label, y_label, title, curve_labels = app._plot_data()

    np.testing.assert_allclose(x_data, np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(
        y_data,
        np.rad2deg(
            np.array(
                [
                    [-0.8, -12.0, 0.8, 12.0],
                    [-0.4, -6.0, 0.4, 6.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )
        ),
    )
    assert x_label == "Temps (s)"
    assert y_label == "Accelerations bras (deg/s2)"
    assert title == "Accelerations bras en fonction de temps"
    assert curve_labels == (
        "Plan bras gauche",
        "Elevation bras gauche",
        "Plan bras droit",
        "Elevation bras droit",
    )


def test_plot_data_can_return_the_four_arm_jerk_curves() -> None:
    """The plot selector should expose the four arm DoF jerks as one 4-curve figure."""

    app = _build_app_for_plotting(plot_x="Temps", plot_y="Jerks bras")
    app._last_simulation = app._visualization_data["result"]
    app._current_values = lambda: {"right_arm_start": 0.2}
    app._values_with_current_fixed_parameters = lambda values: dict(values)
    app._motion_for_kinematic_candidate = lambda _candidate: type(
        "FakeMotion",
        (),
        {
            "left_arm_start": 0.2,
            "right_arm_start": 0.0,
            "left_plane": type("Trajectory", (), {"jerks": np.array([1.0, 2.0]), "step": 0.5, "control_duration": 1.0})(),
            "left_elevation": type("Trajectory", (), {"jerks": np.array([3.0, 4.0]), "step": 0.5, "control_duration": 1.0})(),
            "right_plane": type("Trajectory", (), {"jerks": np.array([5.0, 6.0]), "step": 0.5, "control_duration": 1.0})(),
            "right_elevation": type("Trajectory", (), {"jerks": np.array([7.0, 8.0]), "step": 0.5, "control_duration": 1.0})(),
        },
    )()

    x_data, y_data, x_label, y_label, title, curve_labels = app._plot_data()

    np.testing.assert_allclose(x_data, np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(
        y_data,
        np.rad2deg(
            np.array(
                [
                    [0.0, 0.0, 5.0, 7.0],
                    [1.0, 3.0, 6.0, 8.0],
                    [2.0, 4.0, 6.0, 8.0],
                ]
            )
        ),
    )
    assert x_label == "Temps (s)"
    assert y_label == "Jerks bras (deg/s3)"
    assert title == "Jerks bras en fonction de temps"
    assert curve_labels == (
        "Plan bras gauche",
        "Elevation bras gauche",
        "Plan bras droit",
        "Elevation bras droit",
    )


def test_plot_data_can_group_both_arm_deviations_on_one_figure() -> None:
    """The deviation plot should show both arms together with two curve labels."""

    app = _build_app_for_plotting(plot_x="Temps", plot_y="Deviations bras")

    x_data, y_data, x_label, y_label, title, curve_labels = app._plot_data()

    np.testing.assert_allclose(x_data, np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(
        y_data,
        np.rad2deg(
            np.array(
                [
                    [0.0, 0.0],
                    [0.1, -0.1],
                    [0.2, -0.2],
                ]
            )
        ),
    )
    assert x_label == "Temps (s)"
    assert y_label == "Deviation bras / BTP (deg)"
    assert title == "Deviations bras en fonction de temps"
    assert curve_labels == ("Bras gauche", "Bras droit")


def test_plot_data_can_return_twist_axis_angular_momentum_transfers() -> None:
    """The plot selector should expose the twist-axis angular momentum split by group."""

    app = _build_app_for_plotting(plot_x="Temps", plot_y="Moment cinetique vrille segments")

    x_data, y_data, x_label, y_label, title, curve_labels = app._plot_data()

    np.testing.assert_allclose(x_data, np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(
        y_data,
        np.array(
            [
                [100.0, 200.0, 300.0],
                [110.0, 210.0, 310.0],
                [120.0, 220.0, 320.0],
            ]
        ),
    )
    assert x_label == "Temps (s)"
    assert y_label == "H axe z global au CoM (kg.m2/s)"
    assert title == "Moment cinetique vrille segments en fonction de temps"
    assert curve_labels == ("Bras gauche", "Bras droit", "Reste du corps")


def test_plot_data_can_return_shoulder_torques_and_decomposition() -> None:
    """The plot selector should expose shoulder torques and their decomposition terms."""

    app = _build_app_for_plotting(plot_x="Temps", plot_y="Couples epaules detail")

    x_data, y_data, x_label, y_label, title, curve_labels = app._plot_data()

    np.testing.assert_allclose(x_data, np.array([0.0, 0.5, 1.0]))
    assert y_data.shape == (3, 16)
    np.testing.assert_allclose(y_data[0, :4], np.array([1.0, 0.1, 0.01, 0.001]))
    assert x_label == "Temps (s)"
    assert y_label == "Couples epaules detail (N.m)"
    assert title == "Couples epaules detail en fonction de temps"
    assert curve_labels[0] == "Plan bras gauche | Total"
    assert curve_labels[3] == "Plan bras gauche | N(q,qdot)-N(q,0)"
    assert curve_labels[-1] == "Elevation bras droit | N(q,qdot)-N(q,0)"


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
    app.optimization_mode_var = _FakeVar("Optimize 3D")
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
                    "optimize_3d": {
                        "signature": app._optimization_cache_signature_for_mode("Optimize 3D"),
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

    assert [dataset["mode"] for dataset in datasets] == ["Optimize 3D", "Optimize 2D"]
    assert datasets[0]["best_start_time"] == 0.28
    assert datasets[1]["best_start_time"] == 0.20


def test_scan_plot_datasets_can_include_optimize_3d_btp(tmp_path) -> None:
    """The embedded `vrilles selon t1` figure should include the 3D-BTP scan when available."""

    app = BestTiltingPlaneApp.__new__(BestTiltingPlaneApp)
    app.optimization_mode_var = _FakeVar("Optimize 2D")
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
                    "optimize_3d": {
                        "signature": app._optimization_cache_signature_for_mode("Optimize 3D"),
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
                    "optimize_3d_btp": {
                        "signature": app._optimization_cache_signature_for_mode("Optimize 3D BTP"),
                        "values": {
                            "right_arm_start": 0.30,
                            "left_plane_initial": 0.0,
                            "left_plane_final": 0.0,
                            "right_plane_initial": 0.0,
                            "right_plane_final": 0.0,
                        },
                        "scan_start_times": [0.20, 0.24, 0.30],
                        "scan_final_twist_turns": [-0.44, -0.59, -0.66],
                        "scan_objective_values": [-0.41, -0.56, -0.61],
                        "scan_success_mask": [True, True, True],
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    datasets = app._scan_plot_datasets()

    assert [dataset["mode"] for dataset in datasets] == ["Optimize 2D", "Optimize 3D", "Optimize 3D BTP"]
    assert datasets[2]["best_start_time"] == 0.30


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
        [-135.0, 20.0, -180.0, 0.0, -20.0, 135.0, 0.0, 180.0],
    )
    assert app._plot_canvas.draw_idle_calls == 1


def test_refresh_plot_can_filter_selected_arm_curves() -> None:
    """The embedded plot should draw only the arm curves selected by the user."""

    app = _build_app_for_plotting(plot_x="Temps", plot_y="Cinematique bras")
    app._plot_axis = _FakeAxis()
    app._plot_canvas = _FakeCanvas()
    app._curve_selection_by_plot = {"Cinematique bras": ("Plan bras gauche", "Plan bras droit")}

    app._refresh_plot()

    assert [call["kwargs"]["label"] for call in app._plot_axis.plot_calls] == [
        "Plan bras gauche",
        "Plan bras droit",
    ]


def test_refresh_plot_adds_jerk_bounds_when_plotting_arm_jerks() -> None:
    """The embedded jerk figure should display one jerk bound pair per arm DoF."""

    app = _build_app_for_plotting(plot_x="Temps", plot_y="Jerks bras")
    app._plot_axis = _FakeAxis()
    app._plot_canvas = _FakeCanvas()
    app._last_simulation = app._visualization_data["result"]
    app._current_values = lambda: {"right_arm_start": 0.2}
    app._values_with_current_fixed_parameters = lambda values: dict(values)
    app._motion_for_kinematic_candidate = lambda _candidate: type(
        "FakeMotion",
        (),
        {
            "left_arm_start": 0.2,
            "right_arm_start": 0.0,
            "left_plane": type("Trajectory", (), {"jerks": np.array([1.0, 2.0]), "step": 0.5, "control_duration": 1.0})(),
            "left_elevation": type("Trajectory", (), {"jerks": np.array([3.0, 4.0]), "step": 0.5, "control_duration": 1.0})(),
            "right_plane": type("Trajectory", (), {"jerks": np.array([5.0, 6.0]), "step": 0.5, "control_duration": 1.0})(),
            "right_elevation": type("Trajectory", (), {"jerks": np.array([7.0, 8.0]), "step": 0.5, "control_duration": 1.0})(),
        },
    )()

    app._refresh_plot()

    expected_bounds = sorted(
        round(value, 6)
        for value in np.rad2deg(np.array([-2.0, 2.0, -4.0, 4.0, -6.0, 6.0, -8.0, 8.0], dtype=float))
    )
    assert sorted(round(call["y"], 6) for call in app._plot_axis.axhline_calls) == expected_bounds
    assert app._plot_canvas.draw_idle_calls == 1


def test_refresh_plot_restores_automatic_axis_aspect_in_curve_mode() -> None:
    """Returning to curve mode should reset the 2D axis aspect after top-view plotting."""

    app = _build_app_for_plotting(plot_x="Temps", plot_y="Twist")
    app._plot_axis = _FakeAxis()
    app._plot_canvas = _FakeCanvas()

    app._refresh_plot()

    assert app._plot_axis.aspect_calls == [("auto",)]
