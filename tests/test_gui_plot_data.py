"""Tests for GUI plot-data selection."""

from __future__ import annotations

import numpy as np

from best_tilting_plane.gui.app import BestTiltingPlaneApp, ROOT_INITIAL_OPTIONS


class _FakeVar:
    """Tiny stand-in exposing the Tk variable `get` API used by the GUI."""

    def __init__(self, value: str) -> None:
        self._value = value

    def get(self) -> str:
        """Return the stored value."""

        return self._value


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
                        [0.0, 0.0, 0.0, 0.1, 0.2, 0.3],
                        [0.0, 0.0, 0.0, 0.4, 0.5, 0.6],
                        [0.0, 0.0, 0.0, 0.9, 1.0, 1.1],
                    ],
                    dtype=float,
                ),
            },
        )(),
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

    x_data, y_data, x_label, y_label, title = app._plot_data()

    np.testing.assert_allclose(x_data, np.rad2deg(np.array([0.0, 0.3, 0.8])))
    np.testing.assert_allclose(y_data, np.rad2deg(np.array([0.0, 0.3, 0.8])))
    assert x_label == "Somersault (deg)"
    assert y_label == "Twist (deg)"
    assert title == "Twist en fonction de somersault"


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
    """The `q racine(0)=0` mode should express the arms in the pelvis frame."""

    app = _build_app_for_plotting(root_mode=ROOT_INITIAL_OPTIONS[0])
    rotation_z_90 = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    app._visualization_data["frames"]["pelvis"]["axes"][1] = rotation_z_90

    top_view, _ = app._top_view_plot_data()

    np.testing.assert_allclose(top_view["hand_left"][0], np.array([-0.5, 0.3]))
    np.testing.assert_allclose(top_view["hand_left"][1], np.array([0.4, 0.5]))
