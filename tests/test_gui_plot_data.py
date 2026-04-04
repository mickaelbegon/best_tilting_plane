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
    app.root_initial_mode = _FakeVar(root_mode)
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
    }
    return app


def test_plot_data_returns_selected_angular_momentum_component() -> None:
    """The GUI plot selector should expose the chosen angular-momentum component."""

    app = _build_app_for_plotting(plot_x="Temps", plot_y="Moment cinetique y")

    x_data, y_data, x_label, y_label, title = app._plot_data()

    np.testing.assert_allclose(x_data, np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(y_data, np.array([2.0, 4.0, 12.0]))
    assert x_label == "Temps (s)"
    assert y_label == "H_y au CoM"
    assert title == "Moment cinetique y en fonction de temps"


def test_plot_data_returns_angular_momentum_norm_against_somersault() -> None:
    """The norm of the angular momentum should be plottable against somersault."""

    app = _build_app_for_plotting(plot_x="Somersault", plot_y="Norme moment cinetique")

    x_data, y_data, x_label, y_label, title = app._plot_data()

    np.testing.assert_allclose(x_data, np.rad2deg(np.array([0.0, 0.3, 0.8])))
    np.testing.assert_allclose(y_data, np.array([3.0, 5.0, 13.0]))
    assert x_label == "Somersault (deg)"
    assert y_label == "||H(CoM)||"
    assert title == "Norme moment cinetique en fonction de somersault"
