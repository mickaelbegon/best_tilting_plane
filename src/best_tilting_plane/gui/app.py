"""Interactive GUI for the predictive twisting simulation."""

from __future__ import annotations

import json
import queue
import tkinter as tk
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk
from types import SimpleNamespace

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from best_tilting_plane.gui.debounce import DebouncedRunner
from best_tilting_plane.modeling import (
    ARM_ELEVATION_SEQUENCE,
    ARM_PLANE_SEQUENCE,
    ARM_SEGMENTS_FOR_DEVIATION,
    ARM_SEGMENTS_FOR_VISUALIZATION,
    GLOBAL_AXIS_LABELS,
    LEFT_ARM_ELEVATION_BOUNDS_DEG,
    LEFT_ARM_PLANE_BOUNDS_DEG,
    RIGHT_ARM_ELEVATION_BOUNDS_DEG,
    RIGHT_ARM_PLANE_BOUNDS_DEG,
    ROOT_ROTATION_SEQUENCE,
)
from best_tilting_plane.optimization import (
    DirectMultipleShootingOptimizer,
    TwistStrategyOptimizer,
    show_dms_start_time_sweep_figure,
    show_right_arm_start_sweep_figure,
)
from best_tilting_plane.optimization.dms import (
    DEFAULT_DMS_BTP_DEVIATION_WEIGHT,
    DEFAULT_DMS_JERK_REGULARIZATION,
    DEFAULT_DMS_TWIST_RATE_LAGRANGE_WEIGHT,
    JERK_BOUND_SCALE,
    MULTISTART_START_COUNT,
    OBJECTIVE_MODE_TWIST,
    OBJECTIVE_MODE_TWIST_BTP,
    RIGHT_ARM_START_BOUNDS,
    show_dms_jerk_bounds_figure,
)
from best_tilting_plane.optimization.ipopt import DEFAULT_TWIST_RATE_LAGRANGE_WEIGHT
from best_tilting_plane.simulation import (
    AerialSimulationResult,
    build_piecewise_constant_jerk_arm_motion,
    PiecewiseConstantJerkArmMotion,
    PiecewiseConstantJerkTrajectory,
    PredictiveAerialTwistSimulator,
    SimulationConfiguration,
    TwistOptimizationVariables,
)
from best_tilting_plane.visualization import (
    SKELETON_CONNECTIONS,
    arm_btp_reference_trajectories,
    arm_deviation_from_frames,
    arm_top_view_trajectories,
    best_tilting_plane_corners,
    marker_trajectories,
    present_external_figure,
    segment_frame_trajectories,
    system_observables,
)


@dataclass(frozen=True)
class SliderDefinition:
    """Describe one GUI-controlled decision variable."""

    name: str
    label: str
    minimum: float
    maximum: float
    default: float
    resolution: float = 0.01


SLIDER_DEFINITIONS = (
    SliderDefinition("right_arm_start", "Start bras droit (s)", 0.0, 0.7, 0.10, resolution=0.02),
    SliderDefinition(
        "contact_twist_turns_per_second",
        "Vrille contact (tour/s)",
        -2.0,
        0.0,
        0.0,
        resolution=0.2,
    ),
)
GUI_FIXED_VALUES = {
    "left_plane_initial": 0.0,
    "left_plane_final": 0.0,
    "right_plane_initial": 0.0,
    "right_plane_final": 0.0,
}
GUI_VALUE_NAMES = tuple(definition.name for definition in SLIDER_DEFINITIONS) + tuple(GUI_FIXED_VALUES.keys())
PLOT_X_OPTIONS = ("Temps", "Somersault", "Vrille")
PLOT_MODE_OPTIONS = ("Courbe", "Bras hors BTP (dessus)")
ANIMATION_MODE_OPTIONS = ("Animation 3D", "Bras / BTP")
ANIMATION_REFERENCE_OPTIONS = ("Global", "Racine", "Kinogramme-Racine", "Best tilting plane")
OPTIMIZATION_MODE_OPTIONS = ("Optimize 2D", "Optimize 3D", "Optimize 3D BTP")
PLOT_Y_OPTIONS = (
    "Somersault",
    "Tilt",
    "Twist",
    "Cinematique bras",
    "Vitesses bras",
    "Deviations bras",
)
ROOT_INITIAL_OPTIONS = ("Avec q racine(0)=0", "Sans q racine(0)=0")
TOP_VIEW_LEFT_CHAIN = ("shoulder_left", "elbow_left", "wrist_left", "hand_left")
TOP_VIEW_RIGHT_CHAIN = ("shoulder_right", "elbow_right", "wrist_right", "hand_right")
KINEMATIC_EXPLORER_COLUMN_LABELS = ("Jerk", "Acceleration", "Velocity", "Position")
DEFAULT_CAMERA_ELEVATION_DEG = 20.0
DEFAULT_CAMERA_AZIMUTH_DEG = -60.0
BTP_CAMERA_ELEVATION_DEG = 22.0
BTP_CAMERA_AZIMUTH_DEG = -35.0
ROOT_VIEW_CAMERA_ELEVATION_DEG = 0.0
ROOT_VIEW_CAMERA_AZIMUTH_DEG = -90.0
KINOGRAM_SAMPLE_COUNT = 9
ALL_FRAME_SEGMENTS = tuple(
    dict.fromkeys(ARM_SEGMENTS_FOR_VISUALIZATION + ARM_SEGMENTS_FOR_DEVIATION)
)
ANIMATION_INTERVAL_MS = 35
STANDARD_RK4_STEP = 0.005
OPTIMIZATION_CACHE_VERSION = 6
DMS_SHOOTING_STEP = 0.02
DMS_ACTIVE_DURATION = 0.3
DMS_SCAN_START = 0.0
DMS_SCAN_END = 0.7
DMS_MULTISTART_OFFSET_FROM_2D = 0.2
DMS_JERK_REGULARIZATION = DEFAULT_DMS_JERK_REGULARIZATION
DMS_BTP_DEVIATION_WEIGHT = DEFAULT_DMS_BTP_DEVIATION_WEIGHT
DMS_TWIST_RATE_LAGRANGE_WEIGHT = DEFAULT_DMS_TWIST_RATE_LAGRANGE_WEIGHT
TWIST_RATE_LAGRANGE_WEIGHT = DEFAULT_TWIST_RATE_LAGRANGE_WEIGHT
SCAN_SELECTION_MAX_NORMALIZED_DISTANCE = 0.08
THREE_D_BTP_DISCONTINUITY_THRESHOLD = 0.35
ARM_KINEMATICS_LABELS = (
    "Plan bras gauche",
    "Elevation bras gauche",
    "Plan bras droit",
    "Elevation bras droit",
)
ARM_KINEMATICS_BOUNDS_DEG = (
    LEFT_ARM_PLANE_BOUNDS_DEG,
    LEFT_ARM_ELEVATION_BOUNDS_DEG,
    RIGHT_ARM_PLANE_BOUNDS_DEG,
    RIGHT_ARM_ELEVATION_BOUNDS_DEG,
)
SCAN_PLOT_STYLE_BY_MODE = {
    "Optimize 2D": {"color": "tab:blue", "marker": "o", "label": "Optimize 2D"},
    "Optimize 3D": {"color": "tab:orange", "marker": "s", "label": "Optimize 3D"},
    "Optimize 3D BTP": {"color": "tab:green", "marker": "^", "label": "Optimize 3D BTP"},
    "Optimize DMS": {"color": "tab:orange", "marker": "s", "label": "Optimize 3D"},
}


def _is_three_d_optimization_mode(mode: str) -> bool:
    """Return whether one optimization mode uses the 3D DMS backend."""

    return mode in {"Optimize DMS", "Optimize 3D", "Optimize 3D BTP"}


def _three_d_objective_mode(mode: str) -> str:
    """Map one GUI optimization mode to the underlying DMS objective."""

    if mode == "Optimize 3D BTP":
        return OBJECTIVE_MODE_TWIST_BTP
    return OBJECTIVE_MODE_TWIST


def _optimization_mode_label(mode: str) -> str:
    """Return one short user-facing optimization label."""

    return "DMS" if mode == "Optimize DMS" else mode


def _dms_result_is_better(candidate, reference) -> bool:
    """Return whether one DMS result should be preferred over another."""

    if reference is None:
        return True
    candidate_success = bool(getattr(candidate, "success", False))
    reference_success = bool(getattr(reference, "success", False))
    if candidate_success != reference_success:
        return candidate_success
    return float(getattr(candidate, "objective", float("inf"))) < float(
        getattr(reference, "objective", float("inf"))
    )


def _legacy_cache_keys_for_mode(mode: str) -> tuple[str, ...]:
    """Return compatible legacy cache keys for one optimization mode."""

    if mode == "Optimize 3D":
        return ("optimize_dms",)
    return ()


def _variables_from_gui(values: dict[str, float]) -> TwistOptimizationVariables:
    """Convert GUI values into the optimization-variable structure."""

    return TwistOptimizationVariables(
        right_arm_start=float(values["right_arm_start"]),
        left_plane_initial=np.deg2rad(float(values.get("left_plane_initial", GUI_FIXED_VALUES["left_plane_initial"]))),
        left_plane_final=np.deg2rad(float(values.get("left_plane_final", GUI_FIXED_VALUES["left_plane_final"]))),
        right_plane_initial=np.deg2rad(
            float(values.get("right_plane_initial", GUI_FIXED_VALUES["right_plane_initial"]))
        ),
        right_plane_final=np.deg2rad(float(values.get("right_plane_final", GUI_FIXED_VALUES["right_plane_final"]))),
        contact_twist_rate=2.0 * np.pi * float(values.get("contact_twist_turns_per_second", 0.0)),
    )


def _gui_values_from_variables(variables: TwistOptimizationVariables) -> dict[str, float]:
    """Convert optimization variables to GUI-ready scalar values."""

    return {
        "right_arm_start": float(variables.right_arm_start),
        "left_plane_initial": float(np.rad2deg(variables.left_plane_initial)),
        "left_plane_final": float(np.rad2deg(variables.left_plane_final)),
        "right_plane_initial": float(np.rad2deg(variables.right_plane_initial)),
        "right_plane_final": float(np.rad2deg(variables.right_plane_final)),
    }


class BestTiltingPlaneApp:
    """Tkinter GUI with integrated controls, 3D animation, and configurable plots."""

    def __init__(self, root: tk.Tk) -> None:
        """Create the main window and its controls."""

        self.root = root
        self.root.title("Best Tilting Plane")
        self.root.geometry("1500x900")

        self._entries: dict[str, tk.StringVar] = {}
        self._scales: dict[str, tk.Scale] = {}
        self._last_simulation = None
        self._last_model_path: Path | None = None
        self._visualization_data: dict[str, object] | None = None
        self._animation_after_id: str | None = None
        self._animation_frame_index = 0
        self._animation_playing = False
        self._time_slider_updating = False
        self._auto_simulation_suspended = False
        self._is_closing = False
        self._selected_scan_solutions: list[tuple[str, int]] = []
        self._run_optimization_in_background = True
        self._optimization_thread: threading.Thread | None = None
        self._optimization_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._optimization_poll_after_id: str | None = None
        self._auto_runner = DebouncedRunner(self.root, self._run_simulation, delay_ms=250)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        controls = ttk.Frame(root, padding=12)
        controls.grid(row=0, column=0, sticky="nsw")
        display = ttk.Frame(root, padding=12)
        display.grid(row=0, column=1, sticky="nsew")
        display.columnconfigure(0, weight=1)
        display.rowconfigure(0, weight=3)
        display.rowconfigure(1, weight=0)
        display.rowconfigure(2, weight=3)

        for row, definition in enumerate(SLIDER_DEFINITIONS):
            ttk.Label(controls, text=definition.label).grid(
                row=row, column=0, sticky="w", padx=(0, 8), pady=4
            )
            scale = tk.Scale(
                controls,
                orient=tk.HORIZONTAL,
                from_=definition.minimum,
                to=definition.maximum,
                resolution=definition.resolution,
                length=280,
            )
            scale.set(definition.default)
            scale.grid(row=row, column=1, sticky="ew", pady=4)
            entry_var = tk.StringVar(value=f"{definition.default:.2f}")
            entry = ttk.Entry(controls, textvariable=entry_var, width=10)
            entry.grid(row=row, column=2, sticky="e", pady=4)
            controls.columnconfigure(1, weight=1)

            scale.configure(
                command=lambda value, name=definition.name, var=entry_var: self._on_slider_change(
                    name, var, value
                )
            )
            entry.bind(
                "<Return>",
                lambda _event, name=definition.name, var=entry_var: self._sync_scale_from_entry(
                    name, var
                ),
            )
            entry.bind(
                "<FocusOut>",
                lambda _event, name=definition.name, var=entry_var: self._sync_scale_from_entry(
                    name, var
                ),
            )

            self._entries[definition.name] = entry_var
            self._scales[definition.name] = scale

        scan_row = len(SLIDER_DEFINITIONS)
        self._scan_figure = Figure(figsize=(4.8, 2.6), tight_layout=True)
        self._scan_axis = self._scan_figure.add_subplot(111)
        self._scan_canvas = FigureCanvasTkAgg(self._scan_figure, master=controls)
        self._scan_canvas.get_tk_widget().grid(
            row=scan_row,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(10, 12),
        )
        self._scan_canvas.mpl_connect("button_press_event", self._on_scan_plot_click)

        self.show_btp = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls,
            text="Afficher le best tilting plane",
            variable=self.show_btp,
            command=self._refresh_animation_scene,
        ).grid(row=scan_row + 1, column=0, columnspan=3, sticky="w", pady=(4, 4))

        ttk.Label(controls, text="Repere animation").grid(
            row=scan_row + 2, column=0, sticky="w", pady=(8, 4)
        )
        self.root_initial_mode = tk.StringVar(value=ROOT_INITIAL_OPTIONS[1])
        self.animation_mode_var = tk.StringVar(value=ANIMATION_MODE_OPTIONS[0])
        self.animation_reference_var = tk.StringVar(value=ANIMATION_REFERENCE_OPTIONS[0])
        animation_reference_box = ttk.Combobox(
            controls,
            textvariable=self.animation_reference_var,
            values=ANIMATION_REFERENCE_OPTIONS,
            state="readonly",
            width=24,
        )
        animation_reference_box.grid(
            row=scan_row + 2, column=1, columnspan=2, sticky="ew", pady=(8, 4)
        )
        animation_reference_box.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._on_animation_reference_change(),
        )
        self._apply_animation_reference(self.animation_reference_var.get())

        ttk.Label(controls, text="Mode figure").grid(
            row=scan_row + 3, column=0, sticky="w", pady=4
        )
        self.plot_mode_var = tk.StringVar(value=PLOT_MODE_OPTIONS[0])
        plot_mode_box = ttk.Combobox(
            controls,
            textvariable=self.plot_mode_var,
            values=PLOT_MODE_OPTIONS,
            state="readonly",
            width=24,
        )
        plot_mode_box.grid(
            row=scan_row + 3, column=1, columnspan=2, sticky="ew", pady=4
        )
        plot_mode_box.bind("<<ComboboxSelected>>", lambda _event: self._refresh_plot())

        ttk.Label(controls, text="Figure x").grid(
            row=scan_row + 4, column=0, sticky="w", pady=4
        )
        self.plot_x_var = tk.StringVar(value=PLOT_X_OPTIONS[0])
        plot_x_box = ttk.Combobox(
            controls,
            textvariable=self.plot_x_var,
            values=PLOT_X_OPTIONS,
            state="readonly",
            width=18,
        )
        plot_x_box.grid(
            row=scan_row + 4, column=1, columnspan=2, sticky="ew", pady=4
        )
        plot_x_box.bind("<<ComboboxSelected>>", lambda _event: self._refresh_plot())

        ttk.Label(controls, text="Figure y").grid(
            row=scan_row + 5, column=0, sticky="w", pady=4
        )
        self.plot_y_var = tk.StringVar(value="Twist")
        plot_y_box = ttk.Combobox(
            controls,
            textvariable=self.plot_y_var,
            values=PLOT_Y_OPTIONS,
            state="readonly",
            width=18,
        )
        plot_y_box.grid(
            row=scan_row + 5, column=1, columnspan=2, sticky="ew", pady=4
        )
        plot_y_box.bind("<<ComboboxSelected>>", lambda _event: self._refresh_plot())

        ttk.Button(controls, text="Simulate", command=self._run_simulation).grid(
            row=scan_row + 6, column=0, sticky="w", pady=(10, 0)
        )
        ttk.Button(
            controls,
            text="Explorer cinematique",
            command=self._show_kinematic_explorer,
        ).grid(row=scan_row + 6, column=1, columnspan=2, sticky="ew", pady=(10, 0))
        self.ignore_optimization_cache_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            controls,
            text="Ignorer le cache optimum",
            variable=self.ignore_optimization_cache_var,
        ).grid(row=scan_row + 7, column=0, columnspan=3, sticky="w", pady=(10, 0))
        self.optimization_mode_var = tk.StringVar(value=OPTIMIZATION_MODE_OPTIONS[0])
        optimization_mode_box = ttk.Combobox(
            controls,
            textvariable=self.optimization_mode_var,
            values=OPTIMIZATION_MODE_OPTIONS,
            state="readonly",
            width=18,
        )
        optimization_mode_box.grid(
            row=scan_row + 8, column=0, columnspan=2, sticky="ew", pady=(10, 0), padx=(0, 8)
        )
        ttk.Button(controls, text="Optimize", command=self._optimize_strategy).grid(
            row=scan_row + 8, column=2, sticky="w", pady=(10, 0)
        )

        self.result_var = tk.StringVar(value="Aucune simulation lancée.")
        ttk.Label(controls, textvariable=self.result_var, wraplength=360, justify="left").grid(
            row=scan_row + 9, column=0, columnspan=3, sticky="w", pady=(10, 0)
        )
        self.root.report_callback_exception = self._report_callback_exception
        self.sequence_var = tk.StringVar(
            value=(
                f"BioMod root: translations xyz, rotations {ROOT_ROTATION_SEQUENCE} = "
                f"(somersault, tilt, twist) | Bras: plan {ARM_PLANE_SEQUENCE[0]}, "
                f"elevation {ARM_ELEVATION_SEQUENCE[0]} | "
                f"Plan G {LEFT_ARM_PLANE_BOUNDS_DEG}, D {RIGHT_ARM_PLANE_BOUNDS_DEG} deg | "
                f"Elevation G {LEFT_ARM_ELEVATION_BOUNDS_DEG}, D {RIGHT_ARM_ELEVATION_BOUNDS_DEG} deg"
            )
        )
        ttk.Label(controls, textvariable=self.sequence_var, wraplength=360, justify="left").grid(
            row=scan_row + 10, column=0, columnspan=3, sticky="w", pady=(8, 0)
        )

        self._animation_figure = Figure(figsize=(8.0, 5.0), tight_layout=True)
        self._animation_axis = self._animation_figure.add_subplot(111, projection="3d")
        self._animation_canvas = FigureCanvasTkAgg(self._animation_figure, master=display)
        self._animation_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        playback_controls = ttk.Frame(display)
        playback_controls.grid(row=1, column=0, sticky="ew", pady=(8, 8))
        playback_controls.columnconfigure(1, weight=1)
        self.play_pause_label = tk.StringVar(value="Pause")
        ttk.Button(
            playback_controls,
            textvariable=self.play_pause_label,
            command=self._toggle_animation_playback,
            width=8,
        ).grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.time_value_var = tk.StringVar(value="0.00 s")
        self.time_slider_var = tk.DoubleVar(value=0.0)
        ttk.Label(playback_controls, text="Temps animation").grid(row=0, column=1, sticky="w")
        self.time_slider = ttk.Scale(
            playback_controls,
            orient=tk.HORIZONTAL,
            from_=0.0,
            to=1.0,
            variable=self.time_slider_var,
            command=self._on_time_slider_change,
        )
        self.time_slider.grid(row=1, column=0, columnspan=2, sticky="ew", padx=(0, 8))
        ttk.Label(playback_controls, textvariable=self.time_value_var, width=10).grid(
            row=1, column=2, sticky="e"
        )

        self._plot_figure = Figure(figsize=(8.0, 3.8), tight_layout=True)
        self._plot_axis = self._plot_figure.add_subplot(111)
        self._plot_canvas = FigureCanvasTkAgg(self._plot_figure, master=display)
        self._plot_canvas.get_tk_widget().grid(row=2, column=0, sticky="nsew")

        self._line_artists: tuple[object, ...] = ()
        self._secondary_line_artists: tuple[object, ...] = ()
        self._kinogram_line_artists: tuple[tuple[object, ...], ...] = ()
        self._frame_artists: dict[str, tuple[object, object, object]] = {}
        self._btp_chain_artists: dict[str, object] = {}
        self._btp_path_artists: dict[str, object] = {}
        self._btp_marker_artists: dict[str, object] = {}
        self._secondary_visualization_data: dict[str, object] | None = None
        self._angular_momentum_artist = None
        self._plane_artist: Poly3DCollection | None = None

        self._run_simulation()

    def _on_slider_change(self, _name: str, variable: tk.StringVar, value: str) -> None:
        """Update the entry field and trigger a debounced simulation."""

        variable.set(f"{float(value):.2f}")
        if not self._auto_simulation_suspended:
            self.result_var.set("Simulation automatique...")
            self._auto_runner.schedule()

    def _sync_scale_from_entry(self, name: str, variable: tk.StringVar) -> None:
        """Update the slider when the entry content changes."""

        try:
            self._scales[name].set(float(variable.get()))
        except ValueError:
            variable.set(f"{float(self._scales[name].get()):.2f}")

    def _current_values(self) -> dict[str, float]:
        """Return the current GUI values as a plain dictionary."""

        values = dict(GUI_FIXED_VALUES)
        values.update({name: float(variable.get()) for name, variable in self._entries.items()})
        return values

    def _set_values(self, values: dict[str, float]) -> None:
        """Write a new set of values back to the sliders and entry boxes."""

        self._auto_simulation_suspended = True
        for name, value in values.items():
            if name not in self._scales or name not in self._entries:
                continue
            self._scales[name].set(float(value))
            self._entries[name].set(f"{float(value):.2f}")
        self._auto_simulation_suspended = False

    def _model_path(self) -> Path:
        """Return the generated-model path used by the simulator and optimizer."""

        project_root = Path(__file__).resolve().parents[3]
        return project_root / "generated" / "reduced_aerial_model.bioMod"

    def _standard_optimization_configuration(self) -> SimulationConfiguration:
        """Return the fixed configuration shared by simulation and optimization in the GUI."""

        return SimulationConfiguration(
            integrator="rk4",
            rk4_step=STANDARD_RK4_STEP,
            contact_twist_rate=2.0 * np.pi * self._current_contact_twist_turns_per_second(),
        )

    def _current_contact_twist_turns_per_second(self) -> float:
        """Return the discrete contact twist rate selected in the GUI."""

        entry = getattr(self, "_entries", {}).get("contact_twist_turns_per_second")
        if entry is not None:
            try:
                return float(entry.get())
            except (TypeError, ValueError):
                return 0.0
        current_values = getattr(self, "_current_values", None)
        if callable(current_values):
            try:
                return float(current_values().get("contact_twist_turns_per_second", 0.0))
            except (AttributeError, TypeError, ValueError):
                return 0.0
        return 0.0

    def _optimization_cache_path(self) -> Path:
        """Return the JSON cache path used to store optimal strategies."""

        return self._model_path().with_name("optimization_cache.json")

    def _optimization_cache_key(self) -> str:
        """Return the cache key associated with the current optimization mode."""

        return self._optimization_cache_key_for_mode(self.optimization_mode_var.get())

    def _optimization_cache_key_for_mode(self, mode: str) -> str:
        """Return the cache key associated with one optimization mode."""

        base_key = mode.lower().replace(" ", "_")
        contact_twist_turns_per_second = round(self._current_contact_twist_turns_per_second(), 10)
        if np.isclose(contact_twist_turns_per_second, 0.0):
            return base_key
        sign = "m" if contact_twist_turns_per_second < 0.0 else "p"
        magnitude = f"{abs(contact_twist_turns_per_second):.1f}".replace(".", "p")
        return f"{base_key}__contact_twist_{sign}{magnitude}"

    def _should_ignore_optimization_cache(self) -> bool:
        """Return whether the current optimization should bypass cached results."""

        variable = getattr(self, "ignore_optimization_cache_var", None)
        if variable is None:
            return False
        return bool(variable.get())

    def _optimization_cache_signature(self) -> dict[str, float | int | str]:
        """Describe the numerical setup that must match for a cached optimum to be reused."""

        return self._optimization_cache_signature_for_mode(self.optimization_mode_var.get())

    def _optimization_cache_signature_for_mode(
        self,
        mode: str,
    ) -> dict[str, float | int | str]:
        """Describe the numerical setup that must match for one cached mode to be reused."""

        configuration = self._standard_optimization_configuration()
        signature = {
            "version": OPTIMIZATION_CACHE_VERSION,
            "mode": self._optimization_cache_key_for_mode(mode),
            "final_time": float(configuration.final_time),
            "steps": int(configuration.steps),
            "somersault_rate": float(configuration.somersault_rate),
            "contact_twist_rate": float(configuration.contact_twist_rate),
            "integrator": configuration.integrator,
            "rk4_step": float(configuration.rk4_step) if configuration.rk4_step is not None else None,
        }
        if _is_three_d_optimization_mode(mode):
            signature["dms_shooting_step"] = DMS_SHOOTING_STEP
            signature["dms_active_duration"] = DMS_ACTIVE_DURATION
            signature["dms_scan_start"] = DMS_SCAN_START
            signature["dms_scan_end"] = DMS_SCAN_END
            signature["dms_jerk_regularization"] = DMS_JERK_REGULARIZATION
            signature["dms_btp_deviation_weight"] = DMS_BTP_DEVIATION_WEIGHT
            signature["dms_twist_rate_lagrange_weight"] = DMS_TWIST_RATE_LAGRANGE_WEIGHT
            signature["dms_objective_mode"] = _three_d_objective_mode(mode)
            signature["dms_start_bounds"] = [float(value) for value in RIGHT_ARM_START_BOUNDS]
            signature["dms_multistart_offset_from_2d"] = float(DMS_MULTISTART_OFFSET_FROM_2D)
            signature["dms_multistart_start_count"] = int(MULTISTART_START_COUNT)
            signature["dms_jerk_bound_scale"] = float(JERK_BOUND_SCALE)
        else:
            signature["twist_rate_lagrange_weight"] = TWIST_RATE_LAGRANGE_WEIGHT
        return signature

    def _read_optimization_cache_file(self) -> dict[str, object]:
        """Return the JSON cache content, or an empty structure if it does not exist."""

        cache_path = self._optimization_cache_path()
        if not cache_path.exists():
            return {"records": {}, "progress_records": {}}
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"records": {}, "progress_records": {}}
        if not isinstance(data, dict):
            return {"records": {}, "progress_records": {}}
        records = data.get("records")
        if not isinstance(records, dict):
            records = {}
        progress_records = data.get("progress_records")
        if not isinstance(progress_records, dict):
            progress_records = {}
        return {"records": records, "progress_records": progress_records}

    def _cache_signatures_match(self, record_signature: object, *, mode: str) -> bool:
        """Return whether one stored cache signature is compatible with the current mode."""

        if not isinstance(record_signature, dict):
            return False
        expected = dict(self._optimization_cache_signature_for_mode(mode))
        if record_signature == expected:
            return True

        if mode != "Optimize 3D":
            return False

        normalized = dict(record_signature)
        if normalized.get("mode") == "optimize_dms":
            normalized["mode"] = "optimize_3d"
        normalized.setdefault("dms_objective_mode", OBJECTIVE_MODE_TWIST)
        normalized.setdefault("dms_btp_deviation_weight", DMS_BTP_DEVIATION_WEIGHT)
        normalized.setdefault("dms_twist_rate_lagrange_weight", DMS_TWIST_RATE_LAGRANGE_WEIGHT)
        normalized.setdefault("contact_twist_rate", 0.0)
        normalized["version"] = expected["version"]
        return normalized == expected

    def _matching_cache_record(
        self,
        records: dict[str, object],
        *,
        mode: str,
    ) -> dict[str, object] | None:
        """Return the first cache record compatible with one optimization mode."""

        record_keys = (self._optimization_cache_key_for_mode(mode), *_legacy_cache_keys_for_mode(mode))
        for record_key in record_keys:
            record = records.get(record_key)
            if not isinstance(record, dict):
                continue
            if self._cache_signatures_match(record.get("signature"), mode=mode):
                return record
        return None

    @staticmethod
    def _normalized_cached_gui_values(values: object) -> dict[str, float] | None:
        """Return cached GUI values upgraded to the current slider set."""

        if not isinstance(values, dict):
            return None
        normalized = dict(values)
        normalized.setdefault("contact_twist_turns_per_second", 0.0)
        expected_names = set(GUI_VALUE_NAMES)
        if set(normalized) != expected_names:
            return None
        try:
            return {name: float(normalized[name]) for name in expected_names}
        except (TypeError, ValueError):
            return None

    def _values_with_current_fixed_parameters(self, values: dict[str, float]) -> dict[str, float]:
        """Augment one optimized-value payload with the current non-optimized GUI parameters."""

        merged = dict(values)
        merged.setdefault("contact_twist_turns_per_second", self._current_contact_twist_turns_per_second())
        return merged

    def _load_cached_optimized_values(self, *, mode: str | None = None) -> dict[str, float] | None:
        """Return cached optimized GUI values when the stored signature matches the current setup."""

        cache = self._read_optimization_cache_file()
        target_mode = self.optimization_mode_var.get() if mode is None else mode
        record = self._matching_cache_record(cache["records"], mode=target_mode)
        if record is None:
            return None
        return self._normalized_cached_gui_values(record.get("values"))

    def _load_cached_optimized_scan_data(
        self,
    ) -> dict[str, list[float] | list[bool]] | None:
        """Return optional scan data stored with a cached 1D optimization result."""

        return self._load_cached_scan_bundle_for_mode("Optimize 2D")

    def _load_cached_scan_bundle_for_mode(
        self,
        mode: str,
    ) -> dict[str, list[float] | list[bool] | float | str] | None:
        """Return cached scan data for one mode when the stored signature matches."""

        cache = self._read_optimization_cache_file()
        record = self._matching_cache_record(cache["records"], mode=mode)
        if record is None:
            return None
        values = self._normalized_cached_gui_values(record.get("values"))
        if values is None:
            return None
        try:
            best_start_time = float(values["right_arm_start"])
        except (KeyError, TypeError, ValueError):
            return None
        scan_start_times = record.get("scan_start_times")
        scan_final_twist_turns = record.get("scan_final_twist_turns")
        scan_objective_values = record.get("scan_objective_values")
        scan_success_mask = record.get("scan_success_mask")
        scan_candidate_solutions = record.get("scan_candidate_solutions")
        if (
            not isinstance(scan_start_times, list)
            or not isinstance(scan_final_twist_turns, list)
            or not isinstance(scan_objective_values, list)
            or len(scan_start_times) != len(scan_final_twist_turns)
            or len(scan_start_times) != len(scan_objective_values)
            or len(scan_start_times) == 0
        ):
            return None
        try:
            success_mask = (
                [bool(value) for value in scan_success_mask]
                if isinstance(scan_success_mask, list) and len(scan_success_mask) == len(scan_start_times)
                else [True] * len(scan_start_times)
            )
            candidate_solutions = None
            if isinstance(scan_candidate_solutions, list) and len(scan_candidate_solutions) == len(scan_start_times):
                normalized_candidates = [
                    self._normalized_scan_candidate_record(candidate) for candidate in scan_candidate_solutions
                ]
                if all(candidate is not None for candidate in normalized_candidates):
                    candidate_solutions = normalized_candidates
            return {
                "start_times": [float(value) for value in scan_start_times],
                "final_twist_turns": [float(value) for value in scan_final_twist_turns],
                "objective_values": [float(value) for value in scan_objective_values],
                "success_mask": success_mask,
                "best_start_time": best_start_time,
                "mode": mode,
                "candidate_solutions": candidate_solutions,
            }
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _scan_data_from_lists(
        *,
        start_times: list[float] | np.ndarray,
        final_twist_turns: list[float] | np.ndarray,
        objective_values: list[float] | np.ndarray,
        success_mask: list[bool] | np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Normalize scan data lists into numpy arrays."""

        normalized = {
            "start_times": np.asarray(start_times, dtype=float),
            "final_twist_turns": np.asarray(final_twist_turns, dtype=float),
            "objective_values": np.asarray(objective_values, dtype=float),
        }
        if success_mask is not None:
            normalized["success_mask"] = np.asarray(success_mask, dtype=bool)
        return normalized

    def _evaluate_two_d_sweep(
        self,
        *,
        initial_guess: TwistOptimizationVariables,
    ) -> tuple[object, object, dict[str, float]]:
        """Run the current 2D discrete sweep and return its best result in GUI-friendly form."""

        del initial_guess
        optimizer = TwistStrategyOptimizer.from_builder(
            self._model_path(),
            configuration=self._standard_optimization_configuration(),
        )
        if hasattr(optimizer, "sweep_right_arm_start_only"):
            sweep = optimizer.sweep_right_arm_start_only(
                step=DMS_SHOOTING_STEP,
            )
            result = sweep.best_result
        else:
            fallback_start = float(self._current_values()["right_arm_start"])
            result = optimizer.optimize_right_arm_start_only(
                fallback_start,
                max_iter=25,
                print_level=5,
                print_time=True,
            )
            objective_value = float(getattr(result, "objective", result.final_twist_turns))
            sweep = SimpleNamespace(
                start_times=np.array([result.variables.right_arm_start], dtype=float),
                final_twist_turns=np.array([result.final_twist_turns], dtype=float),
                objective_values=np.array([objective_value], dtype=float),
                best_result=result,
            )
        return sweep, result, _gui_values_from_variables(result.variables)

    def _dms_multistart_reference_start_time(
        self,
        *,
        initial_guess: TwistOptimizationVariables,
        candidate_start_times: np.ndarray,
        use_cache: bool,
        report,
    ) -> float | None:
        """Return the DMS node that should receive the dedicated multistart refinement."""

        if candidate_start_times.size == 0:
            return None

        cached_two_d_values = self._load_cached_optimized_values(mode="Optimize 2D") if use_cache else None
        if cached_two_d_values is not None:
            best_two_d_start = float(cached_two_d_values["right_arm_start"])
        else:
            report("Preparation multistart 3D: balayage 2D de reference...")
            two_d_sweep, two_d_result, two_d_values = self._evaluate_two_d_sweep(initial_guess=initial_guess)
            best_two_d_start = float(two_d_result.variables.right_arm_start)
            if use_cache:
                self._store_cached_optimized_values(
                    two_d_values,
                    final_twist_turns=two_d_result.final_twist_turns,
                    solver_status=two_d_result.solver_status,
                    scan_start_times=np.asarray(two_d_sweep.start_times, dtype=float),
                    scan_final_twist_turns=np.asarray(two_d_sweep.final_twist_turns, dtype=float),
                    scan_objective_values=np.asarray(two_d_sweep.objective_values, dtype=float),
                    mode="Optimize 2D",
                )

        requested_start = best_two_d_start - DMS_MULTISTART_OFFSET_FROM_2D
        snapped_index = int(np.argmin(np.abs(candidate_start_times - requested_start)))
        return float(candidate_start_times[snapped_index])

    def _rebuild_cached_dms_motion(
        self,
        optimized_values: dict[str, float],
        *,
        left_plane_jerk: list[float],
        right_plane_jerk: list[float],
    ) -> PiecewiseConstantJerkArmMotion:
        """Rebuild one cached DMS prescribed motion from stored GUI values and jerk sequences."""

        configuration = self._standard_optimization_configuration()
        right_arm_start = float(optimized_values["right_arm_start"])
        left_plane = PiecewiseConstantJerkTrajectory(
            q0=float(np.deg2rad(optimized_values["left_plane_initial"])),
            qdot0=0.0,
            qddot0=0.0,
            step=DMS_SHOOTING_STEP,
            jerks=np.asarray(left_plane_jerk, dtype=float),
            active_start=0.0,
            active_end=DMS_ACTIVE_DURATION,
            total_duration=configuration.final_time,
        )
        right_plane = PiecewiseConstantJerkTrajectory(
            q0=float(np.deg2rad(optimized_values["right_plane_initial"])),
            qdot0=0.0,
            qddot0=0.0,
            step=DMS_SHOOTING_STEP,
            jerks=np.asarray(right_plane_jerk, dtype=float),
            active_start=0.0,
            active_end=DMS_ACTIVE_DURATION,
            total_duration=configuration.final_time - right_arm_start,
        )
        return PiecewiseConstantJerkArmMotion(
            left_plane=left_plane,
            right_plane=right_plane,
            right_arm_start=right_arm_start,
        )

    def _cached_simulation_result_from_record(self, record: dict[str, object]) -> AerialSimulationResult | None:
        """Return one cached simulation result rebuilt from stored `q` and `qdot` histories."""

        q_history = record.get("q")
        qdot_history = record.get("qdot")
        if not isinstance(q_history, list) or not isinstance(qdot_history, list):
            return None
        try:
            q_array = np.asarray(q_history, dtype=float)
            qdot_array = np.asarray(qdot_history, dtype=float)
        except (TypeError, ValueError):
            return None
        if (
            q_array.ndim != 2
            or qdot_array.ndim != 2
            or q_array.shape != qdot_array.shape
            or q_array.shape[0] == 0
        ):
            return None

        configuration = self._standard_optimization_configuration()
        return AerialSimulationResult(
            time=np.linspace(0.0, float(configuration.final_time), q_array.shape[0]),
            q=q_array,
            qdot=qdot_array,
            qddot=np.zeros_like(q_array),
            integrator_method=str(configuration.integrator),
            rk4_step=configuration.rk4_step,
            integration_seconds=None,
        )

    @staticmethod
    def _store_simulation_result_in_record(
        record: dict[str, object],
        simulation_result: AerialSimulationResult | None,
    ) -> dict[str, object]:
        """Augment one cache record with explicit `q` and `qdot` histories when available."""

        if simulation_result is None:
            return record
        record["q"] = np.asarray(simulation_result.q, dtype=float).tolist()
        record["qdot"] = np.asarray(simulation_result.qdot, dtype=float).tolist()
        return record

    def _scan_candidate_record(
        self,
        *,
        optimized_values: dict[str, float],
        simulation_result: AerialSimulationResult | None,
        mode: str,
        final_twist_turns: float,
        objective: float,
        solver_status: str,
        success: bool,
        left_plane_jerk: np.ndarray | None = None,
        right_plane_jerk: np.ndarray | None = None,
    ) -> dict[str, object]:
        """Build one cacheable scan-candidate record."""

        record: dict[str, object] = {
            "mode": str(mode),
            "values": {
                name: float(value)
                for name, value in self._values_with_current_fixed_parameters(optimized_values).items()
            },
            "final_twist_turns": float(final_twist_turns),
            "objective": float(objective),
            "solver_status": str(solver_status),
            "success": bool(success),
        }
        if left_plane_jerk is not None:
            record["left_plane_jerk"] = np.asarray(left_plane_jerk, dtype=float).tolist()
        if right_plane_jerk is not None:
            record["right_plane_jerk"] = np.asarray(right_plane_jerk, dtype=float).tolist()
        return self._store_simulation_result_in_record(record, simulation_result)

    def _normalized_scan_candidate_record(
        self,
        record: object,
    ) -> dict[str, object] | None:
        """Normalize one cached scan-candidate record for interactive replay."""

        if not isinstance(record, dict):
            return None
        values = self._normalized_cached_gui_values(record.get("values"))
        if values is None:
            return None
        try:
            normalized: dict[str, object] = {
                "mode": str(record.get("mode", "")),
                "values": dict(values),
                "final_twist_turns": float(record.get("final_twist_turns", float("nan"))),
                "objective": float(record.get("objective", float("nan"))),
                "solver_status": str(record.get("solver_status", "cache")),
                "success": bool(record.get("success", True)),
                "simulation": self._cached_simulation_result_from_record(record),
            }
        except (TypeError, ValueError):
            return None
        left_plane_jerk = record.get("left_plane_jerk")
        right_plane_jerk = record.get("right_plane_jerk")
        if isinstance(left_plane_jerk, list) and isinstance(right_plane_jerk, list):
            try:
                normalized["left_plane_jerk"] = np.asarray(left_plane_jerk, dtype=float)
                normalized["right_plane_jerk"] = np.asarray(right_plane_jerk, dtype=float)
            except (TypeError, ValueError):
                return None
        return normalized

    def _show_dms_sweep_figure(
        self,
        *,
        start_times: list[float] | np.ndarray,
        final_twist_turns: list[float] | np.ndarray,
        objective_values: list[float] | np.ndarray,
        success_mask: list[bool] | np.ndarray,
        best_start_time: float,
    ) -> None:
        """Open the discrete 3D sweep figure in an external matplotlib window."""

        if len(start_times) == 0:
            return
        mode = str(self.optimization_mode_var.get())
        primary_scan = {
            "mode": mode,
            "start_times": np.asarray(start_times, dtype=float),
            "final_twist_turns": np.asarray(final_twist_turns, dtype=float),
            "objective_values": np.asarray(objective_values, dtype=float),
            "success_mask": np.asarray(success_mask, dtype=bool),
            "best_start_time": float(best_start_time),
        }
        comparison_scans = [
            bundle
            for bundle in (
                self._load_cached_scan_bundle_for_mode("Optimize 2D"),
                self._load_cached_scan_bundle_for_mode("Optimize 3D"),
                self._load_cached_scan_bundle_for_mode("Optimize 3D BTP"),
            )
            if bundle is not None and bundle.get("mode") != mode
        ]
        if len(comparison_scans) == 1:
            self._show_scan_comparison_figure(primary_scan=primary_scan, comparison_scan=comparison_scans[0])
        else:
            self._show_scan_comparison_figure(primary_scan=primary_scan, comparison_scans=comparison_scans)

    def _schedule_external_callback(self, callback) -> None:
        """Run one external-window callback after Tk regains control."""

        def safe_callback() -> None:
            if getattr(self, "_is_closing", False):
                return
            try:
                callback()
            except Exception as error:
                traceback.print_exc()
                self.result_var.set(f"Erreur figure externe: {error}")

        self.root.after(1, safe_callback)

    def _schedule_scan_figure(
        self,
        *,
        start_times: list[float] | np.ndarray,
        final_twist_turns: list[float] | np.ndarray,
        objective_values: list[float] | np.ndarray,
        best_start_time: float,
        success_mask: list[bool] | np.ndarray | None = None,
    ) -> None:
        """Open the scan figure after the current Tk callback returns."""

        start_times_copy = np.asarray(start_times, dtype=float).copy()
        final_twist_turns_copy = np.asarray(final_twist_turns, dtype=float).copy()
        objective_values_copy = np.asarray(objective_values, dtype=float).copy()
        success_mask_copy = None if success_mask is None else np.asarray(success_mask, dtype=bool).copy()
        self._schedule_external_callback(
            lambda: self._show_scan_figure(
                start_times=start_times_copy,
                final_twist_turns=final_twist_turns_copy,
                objective_values=objective_values_copy,
                success_mask=success_mask_copy,
                best_start_time=float(best_start_time),
            )
        )

    def _show_embedded_scan_plot(self) -> None:
        """Refresh the dedicated embedded `t1` scan figure."""

        if hasattr(self, "_scan_axis"):
            self._refresh_scan_plot()

    def _schedule_dms_jerk_diagnostic_figure(
        self,
        *,
        optimizer: DirectMultipleShootingOptimizer,
        result,
    ) -> None:
        """Open the final DMS jerk diagnostic after the current Tk callback returns."""

        if (
            not hasattr(optimizer, "_global_jerk_bounds")
            or not hasattr(optimizer, "interval_count")
            or not hasattr(optimizer, "node_times")
        ):
            return
        right_start_node_index = getattr(result, "right_arm_start_node_index", None)
        if right_start_node_index is None:
            right_start_node_index = int(round(float(result.variables.right_arm_start) / DMS_SHOOTING_STEP))
        left_lower_bounds, left_upper_bounds, right_lower_bounds, right_upper_bounds = optimizer._global_jerk_bounds(
            right_start_node_index=int(right_start_node_index)
        )
        left_jerk = np.zeros(optimizer.interval_count, dtype=float)
        right_jerk = np.zeros(optimizer.interval_count, dtype=float)
        left_jerk[: len(result.left_plane_jerk)] = np.asarray(result.left_plane_jerk, dtype=float)
        right_start = int(right_start_node_index)
        right_stop = right_start + len(result.right_plane_jerk)
        right_jerk[right_start:right_stop] = np.asarray(result.right_plane_jerk, dtype=float)
        node_times = np.asarray(optimizer.node_times, dtype=float).copy()
        self._schedule_external_callback(
            lambda: show_dms_jerk_bounds_figure(
                node_times=node_times,
                left_jerk=left_jerk.copy(),
                right_jerk=right_jerk.copy(),
                left_lower_bounds=np.asarray(left_lower_bounds, dtype=float).copy(),
                left_upper_bounds=np.asarray(left_upper_bounds, dtype=float).copy(),
                right_lower_bounds=np.asarray(right_lower_bounds, dtype=float).copy(),
                right_upper_bounds=np.asarray(right_upper_bounds, dtype=float).copy(),
                right_arm_start=float(result.variables.right_arm_start),
            )
        )

    def _show_1d_sweep_figure(
        self,
        *,
        start_times: list[float] | np.ndarray,
        final_twist_turns: list[float] | np.ndarray,
        objective_values: list[float] | np.ndarray,
        best_start_time: float,
    ) -> None:
        """Open the discrete 1D sweep figure in an external matplotlib window."""

        if len(start_times) == 0:
            return
        primary_scan = {
            "mode": "Optimize 2D",
            "start_times": np.asarray(start_times, dtype=float),
            "final_twist_turns": np.asarray(final_twist_turns, dtype=float),
            "objective_values": np.asarray(objective_values, dtype=float),
            "success_mask": np.ones(len(start_times), dtype=bool),
            "best_start_time": float(best_start_time),
        }
        comparison_scans = [
            bundle
            for bundle in (
                self._load_cached_scan_bundle_for_mode("Optimize 3D"),
                self._load_cached_scan_bundle_for_mode("Optimize 3D BTP"),
            )
            if bundle is not None
        ]
        if len(comparison_scans) == 1:
            self._show_scan_comparison_figure(primary_scan=primary_scan, comparison_scan=comparison_scans[0])
        else:
            self._show_scan_comparison_figure(primary_scan=primary_scan, comparison_scans=comparison_scans)

    @staticmethod
    def _show_scan_comparison_figure(
        *,
        primary_scan: dict[str, object],
        comparison_scan: dict[str, object] | None = None,
        comparison_scans: list[dict[str, object]] | None = None,
    ) -> None:
        """Open one external figure comparing the available optimization scans on the same axes."""

        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(1, 1, figsize=(8.0, 4.8), tight_layout=True)
        datasets = [primary_scan]
        if comparison_scan is not None and comparison_scan.get("mode") != primary_scan.get("mode"):
            datasets.append(comparison_scan)
        for comparison_scan in comparison_scans or []:
            if comparison_scan.get("mode") != primary_scan.get("mode"):
                datasets.append(comparison_scan)

        for dataset in datasets:
            mode = str(dataset["mode"])
            style = SCAN_PLOT_STYLE_BY_MODE[mode]
            times = np.asarray(dataset["start_times"], dtype=float)
            twists = np.asarray(dataset["final_twist_turns"], dtype=float)
            success_mask = np.asarray(dataset["success_mask"], dtype=bool)
            best_start_time = float(dataset["best_start_time"])
            best_index = int(np.argmin(np.abs(times - best_start_time)))

            axis.plot(
                times,
                twists,
                color=style["color"],
                linewidth=1.8,
                marker=style["marker"],
                markersize=4.0,
                label=style["label"],
            )
            if np.any(~success_mask):
                axis.scatter(
                    times[~success_mask],
                    twists[~success_mask],
                    color=style["color"],
                    marker="x",
                    s=36,
                )
            axis.scatter(
                [times[best_index]],
                [twists[best_index]],
                color=style["color"],
                edgecolors="black",
                linewidths=0.8,
                s=68,
                zorder=3,
            )

        axis.set_ylabel("Vrille finale (tours)")
        axis.set_xlabel("Debut bras droit t1 (s)")
        axis.set_title("Comparaison des scans d'optimisation")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
        BestTiltingPlaneApp._present_external_figure(figure)

    def _show_scan_figure(
        self,
        *,
        start_times: list[float] | np.ndarray,
        final_twist_turns: list[float] | np.ndarray,
        objective_values: list[float] | np.ndarray,
        best_start_time: float,
        success_mask: list[bool] | np.ndarray | None = None,
    ) -> None:
        """Open the figure associated with the current optimization mode scan."""

        if _is_three_d_optimization_mode(self.optimization_mode_var.get()):
            if success_mask is None:
                raise ValueError("The 3D scan figure requires a success mask.")
            self._show_dms_sweep_figure(
                start_times=start_times,
                final_twist_turns=final_twist_turns,
                objective_values=objective_values,
                success_mask=success_mask,
                best_start_time=best_start_time,
            )
            return
        self._show_1d_sweep_figure(
            start_times=start_times,
            final_twist_turns=final_twist_turns,
            objective_values=objective_values,
            best_start_time=best_start_time,
        )

    def _load_cached_dms_solution(
        self,
    ) -> tuple[dict[str, float], PiecewiseConstantJerkArmMotion, float, str, dict[str, list[float] | list[bool]] | None] | None:
        """Return one cached DMS solution when the stored signature matches the current setup."""

        cache = self._read_optimization_cache_file()
        record = self._matching_cache_record(cache["records"], mode=self.optimization_mode_var.get())
        if record is None:
            return None
        if bool(record.get("in_progress", False)):
            return None
        values = self._normalized_cached_gui_values(record.get("values"))
        left_plane_jerk = record.get("left_plane_jerk")
        right_plane_jerk = record.get("right_plane_jerk")
        if values is None or not isinstance(left_plane_jerk, list) or not isinstance(right_plane_jerk, list):
            return None
        try:
            optimized_values = dict(values)
            left_jerk_values = [float(value) for value in left_plane_jerk]
            right_jerk_values = [float(value) for value in right_plane_jerk]
            final_twist_turns = float(record.get("final_twist_turns", float("nan")))
            solver_status = str(record.get("solver_status", "cache"))
        except (TypeError, ValueError):
            return None
        if len(left_jerk_values) == 0 or len(right_jerk_values) == 0:
            return None
        scan_data = None
        scan_start_times = record.get("scan_start_times")
        scan_final_twist_turns = record.get("scan_final_twist_turns")
        scan_objective_values = record.get("scan_objective_values")
        scan_success_mask = record.get("scan_success_mask")
        if (
            isinstance(scan_start_times, list)
            and isinstance(scan_final_twist_turns, list)
            and isinstance(scan_objective_values, list)
            and isinstance(scan_success_mask, list)
            and len(scan_start_times) == len(scan_final_twist_turns) == len(scan_objective_values) == len(scan_success_mask)
            and len(scan_start_times) > 0
        ):
            try:
                scan_data = {
                    "start_times": [float(value) for value in scan_start_times],
                    "final_twist_turns": [float(value) for value in scan_final_twist_turns],
                    "objective_values": [float(value) for value in scan_objective_values],
                    "success_mask": [bool(value) for value in scan_success_mask],
                }
            except (TypeError, ValueError):
                scan_data = None
        motion = self._rebuild_cached_dms_motion(
            optimized_values,
            left_plane_jerk=left_jerk_values,
            right_plane_jerk=right_jerk_values,
        )
        cached_simulation = self._cached_simulation_result_from_record(record)
        if cached_simulation is not None:
            setattr(motion, "_cached_simulation_result", cached_simulation)
        return optimized_values, motion, final_twist_turns, solver_status, scan_data

    def _load_cached_dms_progress(self) -> dict[str, object] | None:
        """Return one resumable DMS checkpoint when the stored signature matches the current setup."""

        cache = self._read_optimization_cache_file()
        record = self._matching_cache_record(cache["progress_records"], mode=self.optimization_mode_var.get())
        if record is None:
            record = self._matching_cache_record(cache["records"], mode=self.optimization_mode_var.get())
        if record is None:
            return None
        if not bool(record.get("in_progress", False)):
            return None

        scan_start_times = record.get("scan_start_times")
        scan_final_twist_turns = record.get("scan_final_twist_turns")
        scan_objective_values = record.get("scan_objective_values")
        scan_success_mask = record.get("scan_success_mask")
        scan_candidate_solutions = record.get("scan_candidate_solutions")
        values = self._normalized_cached_gui_values(record.get("values"))
        left_plane_jerk = record.get("left_plane_jerk")
        right_plane_jerk = record.get("right_plane_jerk")
        if (
            not isinstance(scan_start_times, list)
            or not isinstance(scan_final_twist_turns, list)
            or not isinstance(scan_objective_values, list)
            or not isinstance(scan_success_mask, list)
            or len(scan_start_times) != len(scan_final_twist_turns)
            or len(scan_start_times) != len(scan_objective_values)
            or len(scan_start_times) != len(scan_success_mask)
            or len(scan_start_times) == 0
            or values is None
            or not isinstance(left_plane_jerk, list)
            or not isinstance(right_plane_jerk, list)
        ):
            return None

        try:
            optimized_values = dict(values)
            progress = {
                "start_times": [float(value) for value in scan_start_times],
                "final_twist_turns": [float(value) for value in scan_final_twist_turns],
                "objective_values": [float(value) for value in scan_objective_values],
                "success_mask": [bool(value) for value in scan_success_mask],
                "optimized_values": optimized_values,
                "left_plane_jerk": [float(value) for value in left_plane_jerk],
                "right_plane_jerk": [float(value) for value in right_plane_jerk],
                "final_twist_turns_best": float(record.get("final_twist_turns", float("nan"))),
                "solver_status_best": str(record.get("solver_status", "cache")),
                "last_completed_index": int(record.get("last_completed_index", len(scan_start_times) - 1)),
                "cached_simulation": self._cached_simulation_result_from_record(record),
                "scan_candidate_solutions": (
                    None
                    if not isinstance(scan_candidate_solutions, list)
                    else [
                        candidate
                        for candidate in (
                            self._normalized_scan_candidate_record(candidate_record)
                            for candidate_record in scan_candidate_solutions
                        )
                        if candidate is not None
                    ]
                ),
            }
        except (TypeError, ValueError):
            return None

        for key in ("last_warm_start_primal", "last_warm_start_lam_x", "last_warm_start_lam_g"):
            array_values = record.get(key)
            if array_values is None:
                progress[key] = None
            elif isinstance(array_values, list):
                try:
                    progress[key] = np.asarray([float(value) for value in array_values], dtype=float)
                except (TypeError, ValueError):
                    return None
            else:
                return None
        return progress

    def _store_cached_optimized_values(
        self,
        optimized_values: dict[str, float],
        *,
        final_twist_turns: float,
        solver_status: str,
        scan_start_times: np.ndarray | None = None,
        scan_final_twist_turns: np.ndarray | None = None,
        scan_objective_values: np.ndarray | None = None,
        scan_candidate_solutions: list[dict[str, object]] | None = None,
        mode: str | None = None,
    ) -> None:
        """Persist optimized GUI values for reuse in later GUI sessions."""

        cache_path = self._optimization_cache_path()
        cache = self._read_optimization_cache_file()
        target_mode = self.optimization_mode_var.get() if mode is None else mode
        stored_values = self._values_with_current_fixed_parameters(optimized_values)
        record = {
            "signature": self._optimization_cache_signature_for_mode(target_mode),
            "values": {name: float(value) for name, value in stored_values.items()},
            "final_twist_turns": float(final_twist_turns),
            "solver_status": str(solver_status),
        }
        if (
            scan_start_times is not None
            and scan_final_twist_turns is not None
            and scan_objective_values is not None
        ):
            record["scan_start_times"] = np.asarray(scan_start_times, dtype=float).tolist()
            record["scan_final_twist_turns"] = np.asarray(scan_final_twist_turns, dtype=float).tolist()
            record["scan_objective_values"] = np.asarray(scan_objective_values, dtype=float).tolist()
        if scan_candidate_solutions is not None:
            record["scan_candidate_solutions"] = scan_candidate_solutions
        cache_key = self._optimization_cache_key_for_mode(target_mode)
        cache["records"][cache_key] = record
        cache["progress_records"].pop(cache_key, None)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")

    def _store_cached_dms_solution(
        self,
        optimized_values: dict[str, float],
        *,
        left_plane_jerk: np.ndarray,
        right_plane_jerk: np.ndarray,
        simulation_result: AerialSimulationResult | None,
        scan_start_times: np.ndarray,
        scan_final_twist_turns: np.ndarray,
        scan_objective_values: np.ndarray,
        scan_success_mask: np.ndarray,
        scan_candidate_solutions: list[dict[str, object]] | None = None,
        final_twist_turns: float,
        solver_status: str,
    ) -> None:
        """Persist one DMS optimum for reuse in later GUI sessions."""

        cache_path = self._optimization_cache_path()
        cache = self._read_optimization_cache_file()
        stored_values = self._values_with_current_fixed_parameters(optimized_values)
        cache["records"][self._optimization_cache_key()] = self._store_simulation_result_in_record({
            "signature": self._optimization_cache_signature(),
            "in_progress": False,
            "values": {name: float(value) for name, value in stored_values.items()},
            "left_plane_jerk": np.asarray(left_plane_jerk, dtype=float).tolist(),
            "right_plane_jerk": np.asarray(right_plane_jerk, dtype=float).tolist(),
            "scan_start_times": np.asarray(scan_start_times, dtype=float).tolist(),
            "scan_final_twist_turns": np.asarray(scan_final_twist_turns, dtype=float).tolist(),
            "scan_objective_values": np.asarray(scan_objective_values, dtype=float).tolist(),
            "scan_success_mask": np.asarray(scan_success_mask, dtype=bool).tolist(),
            "final_twist_turns": float(final_twist_turns),
            "solver_status": str(solver_status),
        }, simulation_result)
        if scan_candidate_solutions is not None:
            cache["records"][self._optimization_cache_key()]["scan_candidate_solutions"] = scan_candidate_solutions
        cache["progress_records"].pop(self._optimization_cache_key(), None)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")

    def _store_cached_dms_progress(
        self,
        optimized_values: dict[str, float],
        *,
        left_plane_jerk: np.ndarray,
        right_plane_jerk: np.ndarray,
        simulation_result: AerialSimulationResult | None = None,
        scan_start_times: np.ndarray,
        scan_final_twist_turns: np.ndarray,
        scan_objective_values: np.ndarray,
        scan_success_mask: np.ndarray,
        scan_candidate_solutions: list[dict[str, object]] | None = None,
        last_completed_index: int,
        last_warm_start_primal: np.ndarray | None,
        last_warm_start_lam_x: np.ndarray | None,
        last_warm_start_lam_g: np.ndarray | None,
        final_twist_turns: float,
        solver_status: str,
    ) -> None:
        """Persist one resumable DMS checkpoint after a completed fixed-start OCP."""

        cache_path = self._optimization_cache_path()
        cache = self._read_optimization_cache_file()
        stored_values = self._values_with_current_fixed_parameters(optimized_values)
        cache["progress_records"][self._optimization_cache_key()] = self._store_simulation_result_in_record({
            "signature": self._optimization_cache_signature(),
            "in_progress": True,
            "values": {name: float(value) for name, value in stored_values.items()},
            "left_plane_jerk": np.asarray(left_plane_jerk, dtype=float).tolist(),
            "right_plane_jerk": np.asarray(right_plane_jerk, dtype=float).tolist(),
            "scan_start_times": np.asarray(scan_start_times, dtype=float).tolist(),
            "scan_final_twist_turns": np.asarray(scan_final_twist_turns, dtype=float).tolist(),
            "scan_objective_values": np.asarray(scan_objective_values, dtype=float).tolist(),
            "scan_success_mask": np.asarray(scan_success_mask, dtype=bool).tolist(),
            "last_completed_index": int(last_completed_index),
            "last_warm_start_primal": (
                None if last_warm_start_primal is None else np.asarray(last_warm_start_primal, dtype=float).tolist()
            ),
            "last_warm_start_lam_x": (
                None if last_warm_start_lam_x is None else np.asarray(last_warm_start_lam_x, dtype=float).tolist()
            ),
            "last_warm_start_lam_g": (
                None if last_warm_start_lam_g is None else np.asarray(last_warm_start_lam_g, dtype=float).tolist()
            ),
            "final_twist_turns": float(final_twist_turns),
            "solver_status": str(solver_status),
        }, simulation_result)
        if scan_candidate_solutions is not None:
            cache["progress_records"][self._optimization_cache_key()]["scan_candidate_solutions"] = scan_candidate_solutions
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")

    def _run_simulation(self) -> None:
        """Simulate the current strategy and refresh the embedded displays."""

        variables = _variables_from_gui(self._current_values())
        simulator = PredictiveAerialTwistSimulator.from_builder(
            self._model_path(),
            variables,
            configuration=self._standard_optimization_configuration(),
        )
        self._update_from_simulation(simulator.simulate(), model_path=Path(simulator.model_path))

    def _run_simulation_with_motion(self, prescribed_motion) -> None:
        """Simulate one explicit prescribed motion and refresh the displays."""

        simulator = PredictiveAerialTwistSimulator(
            self._model_path(),
            prescribed_motion,
            configuration=self._standard_optimization_configuration(),
        )
        self._update_from_simulation(simulator.simulate(), model_path=Path(self._model_path()))

    def _update_from_simulation(self, result, *, model_path: Path) -> None:
        """Refresh the GUI from a simulation result."""

        self._last_simulation = result
        self._last_model_path = model_path
        self._refresh_visualization_data()

        integrator_label = result.integrator_method.upper()
        if result.rk4_step is not None:
            integrator_label += f" (dt={result.rk4_step:.4f} s)"
        self.result_var.set(
            f"Vrilles finales: {result.final_twist_turns:.2f} tours | "
            f"Integrateur: {integrator_label}"
        )

        self._animation_frame_index = 0
        self._configure_time_slider()
        self._prepare_animation_scene()
        self._refresh_scan_plot()
        self._refresh_plot()
        self._start_animation_loop()

    def _display_q_history(self, result) -> np.ndarray:
        """Return the generalized coordinates used for display."""

        q_history = np.asarray(result.q, dtype=float).copy()
        if self._animation_reference() in {
            ANIMATION_REFERENCE_OPTIONS[1],
            ANIMATION_REFERENCE_OPTIONS[2],
        }:
            q_history[:, :6] = 0.0
        return q_history

    def _display_qdot_history(self, result) -> np.ndarray:
        """Return the generalized velocities used for display."""

        qdot_history = np.asarray(result.qdot, dtype=float).copy()
        if self._animation_reference() in {
            ANIMATION_REFERENCE_OPTIONS[1],
            ANIMATION_REFERENCE_OPTIONS[2],
        }:
            qdot_history[:, :6] = 0.0
        return qdot_history

    def _animation_reference(self) -> str:
        """Return the currently selected animation reference frame."""

        if hasattr(self, "animation_reference_var"):
            return self.animation_reference_var.get()
        if hasattr(self, "animation_mode_var") and self.animation_mode_var.get() == ANIMATION_MODE_OPTIONS[1]:
            return ANIMATION_REFERENCE_OPTIONS[3]
        if hasattr(self, "root_initial_mode") and self.root_initial_mode.get() == ROOT_INITIAL_OPTIONS[0]:
            return ANIMATION_REFERENCE_OPTIONS[1]
        return ANIMATION_REFERENCE_OPTIONS[0]

    def _apply_animation_reference(self, reference: str) -> None:
        """Map the user-facing animation reference to the internal display settings."""

        if reference == ANIMATION_REFERENCE_OPTIONS[3]:
            self.root_initial_mode.set(ROOT_INITIAL_OPTIONS[1])
            self.animation_mode_var.set(ANIMATION_MODE_OPTIONS[1])
        elif reference in {ANIMATION_REFERENCE_OPTIONS[1], ANIMATION_REFERENCE_OPTIONS[2]}:
            self.root_initial_mode.set(ROOT_INITIAL_OPTIONS[0])
            self.animation_mode_var.set(ANIMATION_MODE_OPTIONS[0])
        else:
            self.root_initial_mode.set(ROOT_INITIAL_OPTIONS[1])
            self.animation_mode_var.set(ANIMATION_MODE_OPTIONS[0])

    def _animation_mode(self) -> str:
        """Return the currently selected animation mode."""

        if hasattr(self, "animation_mode_var"):
            return self.animation_mode_var.get()
        return ANIMATION_MODE_OPTIONS[0]

    def _refresh_visualization_data(self) -> None:
        """Rebuild the cached visualization data from the latest simulation."""

        if self._last_simulation is None or self._last_model_path is None:
            return

        self._visualization_data = self._visualization_payload_for_result(self._last_simulation)
        self._secondary_visualization_data = self._secondary_animation_visualization_data()

    def _visualization_payload_for_result(self, result) -> dict[str, object]:
        """Build the cached visualization payload associated with one simulation result."""

        display_q = self._display_q_history(result)
        display_qdot = self._display_qdot_history(result)
        trajectories = marker_trajectories(self._last_model_path, display_q)
        frame_trajectories = segment_frame_trajectories(
            self._last_model_path,
            display_q,
            ALL_FRAME_SEGMENTS,
        )
        observables = system_observables(self._last_model_path, display_q, display_qdot)
        deviations = arm_deviation_from_frames(frame_trajectories, display_q[:, 3])
        btp_trajectories = arm_btp_reference_trajectories(trajectories, display_q[:, 3])
        return {
            "result": result,
            "display_q": display_q,
            "display_qdot": display_qdot,
            "trajectories": trajectories,
            "btp_trajectories": btp_trajectories,
            "frames": frame_trajectories,
            "observables": observables,
            "deviations": deviations,
        }

    def _secondary_animation_visualization_data(self) -> dict[str, object] | None:
        """Return the comparison solution that should be overlaid in the animation."""

        selected_candidates = self._selected_scan_candidate_records()
        if len(selected_candidates) < 2 or self._last_model_path is None:
            return None

        current_result = None if self._visualization_data is None else self._visualization_data.get("result")
        secondary_candidate = next(
            (
                candidate
                for candidate in selected_candidates
                if candidate.get("simulation") is not None and candidate.get("simulation") is not current_result
            ),
            None,
        )
        if secondary_candidate is None:
            secondary_candidate = selected_candidates[-1]
        secondary_result = secondary_candidate.get("simulation")
        if secondary_result is None:
            return None
        return self._visualization_payload_for_result(secondary_result)

    def _on_animation_reference_change(self) -> None:
        """Refresh the displays when toggling the animation reference frame."""

        self._apply_animation_reference(self.animation_reference_var.get())
        if self._visualization_data is None:
            return
        self._refresh_visualization_data()
        self._prepare_animation_scene()
        self._refresh_plot()
        self._sync_time_slider_to_frame(self._current_plot_frame_index())

    def _prepare_animation_scene(self) -> None:
        """Prepare the 3D axis and artists for the current simulation result."""

        if self._visualization_data is None:
            return

        if self._animation_mode() == ANIMATION_MODE_OPTIONS[1]:
            self._prepare_btp_animation_scene()
            return

        self._prepare_standard_animation_scene()

    def _prepare_standard_animation_scene(self) -> None:
        """Prepare the default global 3D animation scene."""

        trajectories = self._visualization_data["trajectories"]
        secondary_trajectories = (
            None
            if self._secondary_visualization_data is None
            else self._secondary_visualization_data["trajectories"]
        )
        self._animation_axis.clear()
        self._animation_axis.set_xlabel("Mediolat.")
        self._animation_axis.set_ylabel("Ant.-post.")
        self._animation_axis.set_zlabel("Longitudinal")
        self._animation_axis.set_box_aspect((1.0, 1.0, 1.0))
        self._apply_camera_view()

        all_points = np.concatenate(
            [
                marker_history
                for trajectory_set in ([trajectories] if secondary_trajectories is None else [trajectories, secondary_trajectories])
                for marker_history in trajectory_set.values()
            ],
            axis=0,
        )
        span = np.max(all_points, axis=0) - np.min(all_points, axis=0)
        center = np.mean(all_points, axis=0)
        radius = 0.6 * np.max(span)
        self._animation_axis.set_xlim(center[0] - radius, center[0] + radius)
        self._animation_axis.set_ylim(center[1] - radius, center[1] + radius)
        self._animation_axis.set_zlim(center[2] - radius, center[2] + radius)

        self._line_artists = tuple(
            self._animation_axis.plot([], [], [], color="black", linewidth=2.0)[0]
            for _ in SKELETON_CONNECTIONS
        )
        self._secondary_line_artists = tuple(
            self._animation_axis.plot(
                [],
                [],
                [],
                color="0.8",
                linewidth=1.8,
                linestyle="--",
            )[0]
            for _ in SKELETON_CONNECTIONS
        ) if secondary_trajectories is not None else ()
        self._kinogram_line_artists = ()
        if self._animation_reference() == ANIMATION_REFERENCE_OPTIONS[2]:
            kinogram_indices = self._kinogram_sample_indices(
                next(iter(trajectories.values())).shape[0]
            )
            cmap = colormaps["viridis"]
            kinogram_groups: list[tuple[object, ...]] = []
            for color_index, frame_index in enumerate(kinogram_indices):
                color = cmap(0.2 + 0.7 * color_index / max(len(kinogram_indices) - 1, 1))
                artists = tuple(
                    self._animation_axis.plot(
                        [],
                        [],
                        [],
                        color=color,
                        linewidth=1.6,
                        alpha=0.45,
                    )[0]
                    for _ in SKELETON_CONNECTIONS
                )
                for artist, (start_name, end_name) in zip(artists, SKELETON_CONNECTIONS):
                    segment = np.vstack((trajectories[start_name][frame_index], trajectories[end_name][frame_index]))
                    artist.set_data(segment[:, 0], segment[:, 1])
                    artist.set_3d_properties(segment[:, 2])
                kinogram_groups.append(artists)
            self._kinogram_line_artists = tuple(kinogram_groups)
        self._frame_artists = {
            segment_name: tuple(
                self._animation_axis.plot([], [], [], color=color, linewidth=2.0)[0]
                for color in ("tab:red", "tab:green", "tab:blue")
            )
            for segment_name in ARM_SEGMENTS_FOR_VISUALIZATION
        }
        self._angular_momentum_artist = self._animation_axis.plot(
            [], [], [], color="tab:orange", linewidth=3.0
        )[0]
        self._plane_artist = None
        if self.show_btp.get():
            self._plane_artist = Poly3DCollection(
                [np.zeros((4, 3))], alpha=0.18, facecolor="tab:orange", edgecolor="none"
            )
            self._animation_axis.add_collection3d(self._plane_artist)

        self._draw_animation_frame(self._animation_frame_index)

    def _prepare_btp_animation_scene(self) -> None:
        """Prepare the full-model animation expressed in the best-tilting-plane frame."""

        projected = self._visualization_data["btp_trajectories"]
        secondary_projected = (
            None
            if self._secondary_visualization_data is None
            else self._secondary_visualization_data["btp_trajectories"]
        )
        self._animation_axis.clear()
        self._animation_axis.set_xlabel("Axe somersault (m)")
        self._animation_axis.set_ylabel("Axe twist BTP (m)")
        self._animation_axis.set_zlabel("Hors BTP (m)")
        self._animation_axis.set_box_aspect((1.0, 1.0, 0.7))
        self._animation_axis.view_init(
            elev=BTP_CAMERA_ELEVATION_DEG,
            azim=BTP_CAMERA_AZIMUTH_DEG,
        )

        all_points = np.concatenate(
            [
                marker_history
                for trajectory_set in ([projected] if secondary_projected is None else [projected, secondary_projected])
                for marker_history in trajectory_set.values()
            ],
            axis=0,
        )
        span = np.max(all_points, axis=0) - np.min(all_points, axis=0)
        center = np.mean(all_points, axis=0)
        radius = max(0.25, 0.6 * float(np.max(span)))
        self._animation_axis.set_xlim(center[0] - radius, center[0] + radius)
        self._animation_axis.set_ylim(center[1] - radius, center[1] + radius)
        self._animation_axis.set_zlim(center[2] - 0.8 * radius, center[2] + 0.8 * radius)

        self._line_artists = tuple(
            self._animation_axis.plot([], [], [], color="black", linewidth=2.0)[0]
            for _ in SKELETON_CONNECTIONS
        )
        self._secondary_line_artists = tuple(
            self._animation_axis.plot(
                [],
                [],
                [],
                color="0.8",
                linewidth=1.8,
                linestyle="--",
            )[0]
            for _ in SKELETON_CONNECTIONS
        ) if secondary_projected is not None else ()
        self._btp_chain_artists = {
            "left": self._animation_axis.plot([], [], [], color="tab:red", linewidth=2.8, marker="o")[0],
            "right": self._animation_axis.plot([], [], [], color="tab:blue", linewidth=2.8, marker="o")[0],
        }
        self._btp_path_artists = {
            "left": self._animation_axis.plot([], [], [], color="tab:red", linewidth=1.8, alpha=0.35)[0],
            "right": self._animation_axis.plot([], [], [], color="tab:blue", linewidth=1.8, alpha=0.35)[0],
        }
        self._btp_marker_artists = {
            "left": self._animation_axis.plot([], [], [], color="tab:red", marker="o", linestyle="None")[0],
            "right": self._animation_axis.plot([], [], [], color="tab:blue", marker="o", linestyle="None")[0],
        }
        self._frame_artists = {}
        self._angular_momentum_artist = None
        self._plane_artist = None
        if self.show_btp.get():
            self._plane_artist = Poly3DCollection(
                [
                    np.array(
                        [
                            [-radius, -radius, 0.0],
                            [radius, -radius, 0.0],
                            [radius, radius, 0.0],
                            [-radius, radius, 0.0],
                        ]
                    )
                ],
                alpha=0.14,
                facecolor="tab:orange",
                edgecolor="none",
            )
            self._animation_axis.add_collection3d(self._plane_artist)
        self._draw_animation_frame(self._animation_frame_index)

    def _start_animation_loop(self) -> None:
        """Start or restart the embedded animation loop."""

        self._stop_animation_loop()
        self._animation_playing = True
        self.play_pause_label.set("Pause")
        self._animate_next_frame()

    def _apply_camera_view(self) -> None:
        """Apply the default or root-aligned camera depending on the selected reference."""

        if self._animation_reference() in {
            ANIMATION_REFERENCE_OPTIONS[1],
            ANIMATION_REFERENCE_OPTIONS[2],
        }:
            self._animation_axis.view_init(
                elev=ROOT_VIEW_CAMERA_ELEVATION_DEG,
                azim=ROOT_VIEW_CAMERA_AZIMUTH_DEG,
            )
            return
        self._animation_axis.view_init(
            elev=DEFAULT_CAMERA_ELEVATION_DEG,
            azim=DEFAULT_CAMERA_AZIMUTH_DEG,
        )

    @staticmethod
    def _kinogram_sample_indices(frame_count: int) -> np.ndarray:
        """Return evenly spaced frame indices used to draw a root-frame kinogram."""

        if frame_count <= 0:
            return np.array([], dtype=int)
        sample_count = min(KINOGRAM_SAMPLE_COUNT, frame_count)
        return np.unique(np.linspace(0, frame_count - 1, sample_count, dtype=int))

    def _stop_animation_loop(self) -> None:
        """Cancel the currently scheduled animation callback, if any."""

        if self._animation_after_id is not None:
            try:
                self.root.after_cancel(self._animation_after_id)
            except tk.TclError:
                pass
            self._animation_after_id = None
        self._animation_playing = False
        self.play_pause_label.set("Play")

    @staticmethod
    def _cancel_canvas_idle_draw(canvas) -> None:
        """Cancel one pending matplotlib Tk idle draw callback, if present."""

        if canvas is None:
            return
        idle_draw_id = getattr(canvas, "_idle_draw_id", None)
        if idle_draw_id is None:
            return
        tk_canvas = getattr(canvas, "_tkcanvas", None)
        if tk_canvas is None:
            widget_getter = getattr(canvas, "get_tk_widget", None)
            if callable(widget_getter):
                tk_canvas = widget_getter()
        try:
            if tk_canvas is not None:
                tk_canvas.after_cancel(idle_draw_id)
        except tk.TclError:
            pass
        finally:
            try:
                canvas._idle_draw_id = None
            except Exception:
                pass

    def _on_close(self) -> None:
        """Close the GUI after cancelling any pending Tk callbacks."""

        self._is_closing = True
        self._stop_animation_loop()
        self._cancel_canvas_idle_draw(getattr(self, "_animation_canvas", None))
        self._cancel_canvas_idle_draw(getattr(self, "_scan_canvas", None))
        self._cancel_canvas_idle_draw(getattr(self, "_plot_canvas", None))
        optimization_poll_after_id = getattr(self, "_optimization_poll_after_id", None)
        if optimization_poll_after_id is not None:
            try:
                self.root.after_cancel(optimization_poll_after_id)
            except tk.TclError:
                pass
            self._optimization_poll_after_id = None
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _toggle_animation_playback(self) -> None:
        """Toggle the embedded animation between play and pause."""

        if self._visualization_data is None:
            return
        if self._animation_playing:
            self._stop_animation_loop()
            return
        self._start_animation_loop()

    def _animate_next_frame(self) -> None:
        """Advance the embedded animation by one frame."""

        if self._is_closing:
            return
        if self._visualization_data is None:
            return

        result = self._visualization_data["result"]
        current_index = self._animation_frame_index
        self._draw_animation_frame(current_index)
        self._sync_time_slider_to_frame(current_index)
        if self._plot_requires_frame_sync():
            self._refresh_plot()
        self._animation_frame_index = (current_index + 1) % result.time.size
        try:
            self._animation_after_id = self.root.after(ANIMATION_INTERVAL_MS, self._animate_next_frame)
        except tk.TclError:
            self._animation_after_id = None
            self._animation_playing = False

    def _configure_time_slider(self) -> None:
        """Match the time slider range and display to the current simulation."""

        if self._visualization_data is None:
            return

        times = np.asarray(self._visualization_data["result"].time, dtype=float)
        self.time_slider.configure(from_=float(times[0]), to=float(times[-1]))
        self._sync_time_slider_to_frame(self._animation_frame_index)

    def _frame_index_from_time(self, time_value: float) -> int:
        """Return the closest frame index to the requested animation time."""

        if self._visualization_data is None:
            raise RuntimeError("No simulation available for animation.")

        times = np.asarray(self._visualization_data["result"].time, dtype=float)
        clipped_time = float(np.clip(time_value, times[0], times[-1]))
        insertion_index = int(np.searchsorted(times, clipped_time, side="left"))
        if insertion_index <= 0:
            return 0
        if insertion_index >= times.size:
            return int(times.size - 1)
        previous_time = times[insertion_index - 1]
        next_time = times[insertion_index]
        if abs(clipped_time - previous_time) <= abs(next_time - clipped_time):
            return insertion_index - 1
        return insertion_index

    def _sync_time_slider_to_frame(self, frame_index: int) -> None:
        """Update the time slider and label from the current animation frame."""

        if self._visualization_data is None:
            return

        times = np.asarray(self._visualization_data["result"].time, dtype=float)
        bounded_index = int(np.clip(frame_index, 0, times.size - 1))
        time_value = float(times[bounded_index])
        self._time_slider_updating = True
        self.time_slider_var.set(time_value)
        self._time_slider_updating = False
        self.time_value_var.set(f"{time_value:.2f} s")

    def _on_time_slider_change(self, value: str) -> None:
        """Jump to the selected time and pause the animation for manual inspection."""

        if self._visualization_data is None or self._time_slider_updating:
            return

        self._stop_animation_loop()
        frame_index = self._frame_index_from_time(float(value))
        self._animation_frame_index = frame_index
        self._draw_animation_frame(frame_index)
        self._sync_time_slider_to_frame(frame_index)
        if self._plot_requires_frame_sync():
            self._refresh_plot()

    def _draw_animation_frame(self, frame_index: int) -> None:
        """Draw one frame of the embedded 3D animation."""

        if self._visualization_data is None:
            return

        if self._animation_mode() == ANIMATION_MODE_OPTIONS[1]:
            self._draw_btp_animation_frame(frame_index)
            return

        if self._angular_momentum_artist is None:
            return

        result = self._visualization_data["result"]
        display_q = self._visualization_data["display_q"]
        trajectories = self._visualization_data["trajectories"]
        secondary_trajectories = (
            None
            if self._secondary_visualization_data is None
            else self._secondary_visualization_data["trajectories"]
        )
        frame_trajectories = self._visualization_data["frames"]
        observables = self._visualization_data["observables"]

        for artist, (start_name, end_name) in zip(self._line_artists, SKELETON_CONNECTIONS):
            segment = np.vstack(
                (trajectories[start_name][frame_index], trajectories[end_name][frame_index])
            )
            artist.set_data(segment[:, 0], segment[:, 1])
            artist.set_3d_properties(segment[:, 2])

        if secondary_trajectories is not None:
            secondary_frame_count = next(iter(secondary_trajectories.values())).shape[0]
            secondary_frame_index = int(np.clip(frame_index, 0, secondary_frame_count - 1))
            for artist, (start_name, end_name) in zip(self._secondary_line_artists, SKELETON_CONNECTIONS):
                segment = np.vstack(
                    (
                        secondary_trajectories[start_name][secondary_frame_index],
                        secondary_trajectories[end_name][secondary_frame_index],
                    )
                )
                artist.set_data(segment[:, 0], segment[:, 1])
                artist.set_3d_properties(segment[:, 2])

        for segment_name, artists in self._frame_artists.items():
            origin = frame_trajectories[segment_name]["origin"][frame_index]
            axes_matrix = frame_trajectories[segment_name]["axes"][frame_index]
            for axis_index, artist in enumerate(artists):
                endpoint = origin + 0.15 * axes_matrix[:, axis_index]
                points = np.vstack((origin, endpoint))
                artist.set_data(points[:, 0], points[:, 1])
                artist.set_3d_properties(points[:, 2])

        com = observables["center_of_mass"][frame_index]
        angular_momentum = observables["angular_momentum"][frame_index]
        momentum_points = np.vstack((com, com + 0.08 * angular_momentum))
        self._angular_momentum_artist.set_data(momentum_points[:, 0], momentum_points[:, 1])
        self._angular_momentum_artist.set_3d_properties(momentum_points[:, 2])

        if self._plane_artist is not None:
            corners = best_tilting_plane_corners(
                trajectories["pelvis_origin"][frame_index],
                somersault_angle=display_q[frame_index, 3],
            )
            self._plane_artist.set_verts([corners])

        self._animation_axis.set_title(
            "Animation 3D | "
            f"t = {result.time[frame_index]:.2f} s | "
            f"vrilles = {display_q[frame_index, 5] / (2*np.pi):.2f} | "
            f"H(CoM) = {np.array2string(angular_momentum, precision=2)} | "
            f"axes: {GLOBAL_AXIS_LABELS}"
        )
        self._animation_canvas.draw_idle()

    def _draw_btp_animation_frame(self, frame_index: int) -> None:
        """Draw one frame of the full model expressed in the best-tilting-plane frame."""

        result = self._visualization_data["result"]
        projected = self._visualization_data["btp_trajectories"]
        secondary_projected = (
            None
            if self._secondary_visualization_data is None
            else self._secondary_visualization_data["btp_trajectories"]
        )
        deviations = self._visualization_data["deviations"]

        for artist, (start_name, end_name) in zip(self._line_artists, SKELETON_CONNECTIONS):
            segment = np.vstack((projected[start_name][frame_index], projected[end_name][frame_index]))
            artist.set_data(segment[:, 0], segment[:, 1])
            artist.set_3d_properties(segment[:, 2])

        if secondary_projected is not None:
            secondary_frame_count = next(iter(secondary_projected.values())).shape[0]
            secondary_frame_index = int(np.clip(frame_index, 0, secondary_frame_count - 1))
            for artist, (start_name, end_name) in zip(self._secondary_line_artists, SKELETON_CONNECTIONS):
                segment = np.vstack(
                    (
                        secondary_projected[start_name][secondary_frame_index],
                        secondary_projected[end_name][secondary_frame_index],
                    )
                )
                artist.set_data(segment[:, 0], segment[:, 1])
                artist.set_3d_properties(segment[:, 2])

        left_chain = np.vstack([projected[marker_name][frame_index] for marker_name in TOP_VIEW_LEFT_CHAIN])
        right_chain = np.vstack(
            [projected[marker_name][frame_index] for marker_name in TOP_VIEW_RIGHT_CHAIN]
        )
        left_path = projected["hand_left"][: frame_index + 1]
        right_path = projected["hand_right"][: frame_index + 1]

        self._btp_chain_artists["left"].set_data(left_chain[:, 0], left_chain[:, 1])
        self._btp_chain_artists["left"].set_3d_properties(left_chain[:, 2])
        self._btp_chain_artists["right"].set_data(right_chain[:, 0], right_chain[:, 1])
        self._btp_chain_artists["right"].set_3d_properties(right_chain[:, 2])

        self._btp_path_artists["left"].set_data(left_path[:, 0], left_path[:, 1])
        self._btp_path_artists["left"].set_3d_properties(left_path[:, 2])
        self._btp_path_artists["right"].set_data(right_path[:, 0], right_path[:, 1])
        self._btp_path_artists["right"].set_3d_properties(right_path[:, 2])

        self._btp_marker_artists["left"].set_data([left_chain[-1, 0]], [left_chain[-1, 1]])
        self._btp_marker_artists["left"].set_3d_properties([left_chain[-1, 2]])
        self._btp_marker_artists["right"].set_data([right_chain[-1, 0]], [right_chain[-1, 1]])
        self._btp_marker_artists["right"].set_3d_properties([right_chain[-1, 2]])

        self._animation_axis.set_title(
            "Animation modele / BTP | "
            f"t = {result.time[frame_index]:.2f} s | "
            f"dev. G = {np.rad2deg(deviations['left'][frame_index]):.1f} deg | "
            f"dev. D = {np.rad2deg(deviations['right'][frame_index]):.1f} deg"
        )
        self._animation_canvas.draw_idle()

    def _root_series(self, result, root_index: int) -> np.ndarray:
        """Return one root-angle series from the currently displayed configurations."""

        if self._visualization_data is not None and "display_q" in self._visualization_data:
            return np.asarray(self._visualization_data["display_q"][:, 3 + root_index], dtype=float)
        return np.asarray(result.q[:, 3 + root_index], dtype=float)

    def _arm_coordinate_series(self) -> np.ndarray:
        """Return the four displayed arm coordinates currently shown in the GUI."""

        if self._visualization_data is None:
            raise RuntimeError("No simulation available for plotting.")
        if "display_q" in self._visualization_data:
            return np.asarray(self._visualization_data["display_q"][:, 6:10], dtype=float)
        return np.asarray(self._visualization_data["result"].q[:, 6:10], dtype=float)

    def _arm_velocity_series(self) -> np.ndarray:
        """Return the four displayed arm velocities currently shown in the GUI."""

        if self._visualization_data is None:
            raise RuntimeError("No simulation available for plotting.")
        if "display_qdot" in self._visualization_data:
            return np.asarray(self._visualization_data["display_qdot"][:, 6:10], dtype=float)
        return np.asarray(self._visualization_data["result"].qdot[:, 6:10], dtype=float)

    def _selected_scan_candidate_records(self) -> list[dict[str, object]]:
        """Return the currently selected scan candidates, in selection order."""

        if not hasattr(self, "_selected_scan_solutions"):
            return []
        datasets_by_mode = {
            str(dataset["mode"]): dataset for dataset in self._scan_plot_datasets()
        }
        selected_candidates: list[dict[str, object]] = []
        for mode, index in self._selected_scan_solutions:
            dataset = datasets_by_mode.get(mode)
            if dataset is None:
                continue
            candidates = dataset.get("candidate_solutions")
            if not isinstance(candidates, list) or not (0 <= index < len(candidates)):
                continue
            candidate = candidates[index]
            if isinstance(candidate, dict):
                selected_candidates.append(candidate)
        return selected_candidates

    def _plot_data_for_result(
        self,
        result: AerialSimulationResult,
    ) -> tuple[np.ndarray, np.ndarray, str, str, str, tuple[str, ...] | None]:
        """Return plot-ready data for one specific simulation result."""

        display_q = self._display_q_history(result)
        display_qdot = self._display_qdot_history(result)

        if self.plot_x_var.get() == "Somersault":
            x_data = np.rad2deg(display_q[:, 3])
            x_label = "Somersault (deg)"
        elif self.plot_x_var.get() == "Vrille":
            x_data = np.rad2deg(display_q[:, 5])
            x_label = "Vrille (deg)"
        else:
            x_data = np.asarray(result.time, dtype=float)
            x_label = "Temps (s)"

        y_choice = self.plot_y_var.get()
        if y_choice == "Somersault":
            y_data = np.rad2deg(display_q[:, 3])
            y_label = "Somersault (deg)"
        elif y_choice == "Tilt":
            y_data = np.rad2deg(display_q[:, 4])
            y_label = "Tilt (deg)"
        elif y_choice == "Twist":
            y_data = np.rad2deg(display_q[:, 5])
            y_label = "Twist (deg)"
        elif y_choice == "Cinematique bras":
            y_data = np.rad2deg(display_q[:, 6:10])
            y_label = "Angles bras (deg)"
            curve_labels = ARM_KINEMATICS_LABELS
        elif y_choice == "Vitesses bras":
            y_data = np.rad2deg(display_qdot[:, 6:10])
            y_label = "Vitesses bras (deg/s)"
            curve_labels = ARM_KINEMATICS_LABELS
        elif y_choice == "Deviations bras":
            model_path = self._last_model_path or Path(self._model_path())
            frame_trajectories = segment_frame_trajectories(model_path, display_q, ALL_FRAME_SEGMENTS)
            deviations = arm_deviation_from_frames(frame_trajectories, display_q[:, 3])
            y_data = np.rad2deg(np.column_stack((deviations["left"], deviations["right"])))
            y_label = "Deviation bras / BTP (deg)"
            curve_labels = ("Bras gauche", "Bras droit")
        else:
            y_data = np.asarray(result.time, dtype=float)
            y_label = "Temps (s)"
            curve_labels = None
        if y_choice not in {"Cinematique bras", "Vitesses bras", "Deviations bras"}:
            curve_labels = None

        title = f"{y_choice} en fonction de {self.plot_x_var.get().lower()}"
        return x_data, y_data, x_label, y_label, title, curve_labels

    def _scan_plot_datasets(self) -> list[dict[str, object]]:
        """Return the scan datasets available for the embedded bottom-left figure."""

        if not hasattr(self, "optimization_mode_var"):
            return []

        current_mode = str(self.optimization_mode_var.get())
        datasets: list[dict[str, object]] = []
        seen_modes: set[str] = set()
        ordered_modes = [current_mode] + [
            mode for mode in OPTIMIZATION_MODE_OPTIONS if mode != current_mode
        ]
        for mode in ordered_modes:
            if current_mode == "Optimize DMS" and mode == "Optimize 3D":
                continue
            if mode in seen_modes:
                continue
            seen_modes.add(mode)
            bundle = self._load_cached_scan_bundle_for_mode(mode)
            if bundle is not None:
                datasets.append(bundle)
        return datasets

    def _apply_scan_candidate_solution(self, candidate: dict[str, object]) -> None:
        """Replay one scan candidate selected from the dedicated `t1` figure."""

        values = dict(candidate["values"])
        mode = str(candidate.get("mode", self.optimization_mode_var.get()))
        self.optimization_mode_var.set(mode)

        prescribed_motion = None
        simulation = candidate.get("simulation")
        if isinstance(simulation, AerialSimulationResult):
            prescribed_motion = SimpleNamespace(_cached_simulation_result=simulation)
        elif "left_plane_jerk" in candidate and "right_plane_jerk" in candidate:
            prescribed_motion = self._rebuild_cached_dms_motion(
                values,
                left_plane_jerk=np.asarray(candidate["left_plane_jerk"], dtype=float),
                right_plane_jerk=np.asarray(candidate["right_plane_jerk"], dtype=float),
            )
            rebuilt_simulation = candidate.get("simulation")
            if isinstance(rebuilt_simulation, AerialSimulationResult):
                setattr(prescribed_motion, "_cached_simulation_result", rebuilt_simulation)

        self._apply_optimized_values(
            values,
            prescribed_motion=prescribed_motion,
            status_suffix=(
                f"solution selectionnee: {float(candidate['final_twist_turns']):.2f} tours "
                f"({candidate['solver_status']})"
            ),
        )

    def _nearest_scan_candidate(
        self,
        x_value: float,
        y_value: float,
    ) -> tuple[str, int, dict[str, object]] | None:
        """Return the nearest clickable scan candidate, if one is close enough."""

        datasets = self._scan_plot_datasets()
        best_match = None
        best_distance = float("inf")
        for dataset in datasets:
            candidates = dataset.get("candidate_solutions")
            if not isinstance(candidates, list):
                continue
            x_series = np.asarray(dataset["start_times"], dtype=float)
            y_series = np.asarray(dataset["final_twist_turns"], dtype=float)
            if x_series.size == 0 or y_series.size == 0:
                continue
            x_scale = max(float(np.ptp(x_series)), 1e-9)
            y_scale = max(float(np.ptp(y_series)), 1e-9)
            for index, candidate in enumerate(candidates):
                if candidate is None:
                    continue
                normalized_distance = float(
                    np.hypot((x_value - x_series[index]) / x_scale, (y_value - y_series[index]) / y_scale)
                )
                if normalized_distance < best_distance:
                    best_distance = normalized_distance
                    best_match = (str(dataset["mode"]), index, candidate)
        if best_match is None or best_distance > SCAN_SELECTION_MAX_NORMALIZED_DISTANCE:
            return None
        return best_match

    def _on_scan_plot_click(self, event) -> None:
        """Replay the clicked scan solution from the dedicated bottom-left figure."""

        if getattr(event, "inaxes", None) is not self._scan_axis:
            return
        if event.xdata is None or event.ydata is None:
            return
        nearest = self._nearest_scan_candidate(float(event.xdata), float(event.ydata))
        if nearest is None:
            return
        mode, index, candidate = nearest
        selection = (mode, index)
        if not hasattr(self, "_selected_scan_solutions"):
            self._selected_scan_solutions = []
        if selection in self._selected_scan_solutions:
            self._selected_scan_solutions.remove(selection)
        else:
            self._selected_scan_solutions.append(selection)
            self._selected_scan_solutions = self._selected_scan_solutions[-2:]
        self._refresh_scan_plot()
        self._refresh_plot()
        self._apply_scan_candidate_solution(candidate)

    def _refresh_scan_plot(self) -> None:
        """Draw the embedded scan figure showing the final twist count as a function of `t1`."""

        self._scan_axis.clear()
        datasets = self._scan_plot_datasets()
        self._scan_axis.set_xlabel("Debut bras droit t1 (s)")
        self._scan_axis.set_ylabel("Vrilles finales (tours)")
        self._scan_axis.set_title("Vrilles finales en fonction de t1")
        self._scan_axis.grid(True, alpha=0.3)
        if not datasets:
            self._scan_axis.text(
                0.5,
                0.5,
                "Aucun scan disponible.\nLancez Optimize.",
                ha="center",
                va="center",
                transform=self._scan_axis.transAxes,
            )
            self._scan_canvas.draw_idle()
            return

        for dataset in datasets:
            mode = str(dataset["mode"])
            style = SCAN_PLOT_STYLE_BY_MODE[mode]
            start_times = np.asarray(dataset["start_times"], dtype=float)
            final_twist_turns = np.asarray(dataset["final_twist_turns"], dtype=float)
            success_mask = np.asarray(dataset["success_mask"], dtype=bool)
            best_start_time = float(dataset["best_start_time"])
            best_index = int(np.argmin(np.abs(start_times - best_start_time)))
            self._scan_axis.plot(
                start_times,
                final_twist_turns,
                color=style["color"],
                linewidth=1.8,
                marker=style["marker"],
                markersize=4.0,
                label=style["label"],
            )
            if np.any(~success_mask):
                self._scan_axis.scatter(
                    start_times[~success_mask],
                    final_twist_turns[~success_mask],
                    color=style["color"],
                    marker="x",
                    s=36,
                )
            self._scan_axis.scatter(
                [start_times[best_index]],
                [final_twist_turns[best_index]],
                color=style["color"],
                edgecolors="black",
                linewidths=0.8,
                s=64,
                zorder=3,
            )
            candidates = dataset.get("candidate_solutions")
            if isinstance(candidates, list):
                for selection_rank, (selected_mode, selected_index) in enumerate(
                    getattr(self, "_selected_scan_solutions", []),
                    start=1,
                ):
                    if selected_mode != mode or not (0 <= selected_index < len(candidates)):
                        continue
                    self._scan_axis.scatter(
                        [start_times[selected_index]],
                        [final_twist_turns[selected_index]],
                        facecolors="none",
                        edgecolors="black",
                        linewidths=2.0,
                        s=180,
                        zorder=4,
                    )
                    self._scan_axis.text(
                        start_times[selected_index],
                        final_twist_turns[selected_index],
                        str(selection_rank),
                        ha="center",
                        va="center",
                        fontsize=9,
                        weight="bold",
                        zorder=5,
                    )
        self._scan_axis.legend(loc="best")
        self._scan_canvas.draw_idle()

    def _add_arm_kinematic_bounds_to_plot(self, colors: tuple[str, ...]) -> None:
        """Overlay the validated arm-angle bounds on the current kinematics plot."""

        for color, (lower_bound, upper_bound) in zip(colors, ARM_KINEMATICS_BOUNDS_DEG, strict=False):
            self._plot_axis.axhline(
                lower_bound,
                color=color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.35,
            )
            self._plot_axis.axhline(
                upper_bound,
                color=color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.35,
            )

    def _plot_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, str, str, str, tuple[str, ...] | None]:
        """Return the currently selected x/y data and corresponding labels."""

        if self._visualization_data is None:
            raise RuntimeError("No simulation available for plotting.")

        result = self._visualization_data["result"]
        deviations = self._visualization_data["deviations"]

        if self.plot_x_var.get() == "Somersault":
            x_data = np.rad2deg(self._root_series(result, 0))
            x_label = "Somersault (deg)"
        elif self.plot_x_var.get() == "Vrille":
            x_data = np.rad2deg(self._root_series(result, 2))
            x_label = "Vrille (deg)"
        else:
            x_data = np.asarray(result.time, dtype=float)
            x_label = "Temps (s)"

        y_choice = self.plot_y_var.get()
        if y_choice == "Somersault":
            y_data = np.rad2deg(self._root_series(result, 0))
            y_label = "Somersault (deg)"
        elif y_choice == "Tilt":
            y_data = np.rad2deg(self._root_series(result, 1))
            y_label = "Tilt (deg)"
        elif y_choice == "Twist":
            y_data = np.rad2deg(self._root_series(result, 2))
            y_label = "Twist (deg)"
        elif y_choice == "Cinematique bras":
            y_data = np.rad2deg(self._arm_coordinate_series())
            y_label = "Angles bras (deg)"
            curve_labels = ARM_KINEMATICS_LABELS
        elif y_choice == "Vitesses bras":
            y_data = np.rad2deg(self._arm_velocity_series())
            y_label = "Vitesses bras (deg/s)"
            curve_labels = ARM_KINEMATICS_LABELS
        elif y_choice == "Deviations bras":
            y_data = np.rad2deg(np.column_stack((deviations["left"], deviations["right"])))
            y_label = "Deviation bras / BTP (deg)"
            curve_labels = ("Bras gauche", "Bras droit")
        else:
            y_data = np.rad2deg(self._root_series(result, 2))
            y_label = "Twist (deg)"
        if y_choice not in {"Cinematique bras", "Vitesses bras", "Deviations bras"}:
            curve_labels = None

        title = f"{y_choice} en fonction de {self.plot_x_var.get().lower()}"
        return x_data, y_data, x_label, y_label, title, curve_labels

    def _plot_requires_frame_sync(self) -> bool:
        """Return whether the current 2D figure depends on the animation frame."""

        return self.plot_mode_var.get() == PLOT_MODE_OPTIONS[1]

    def _current_plot_frame_index(self) -> int:
        """Return the frame currently displayed to the user."""

        if self._visualization_data is None:
            raise RuntimeError("No simulation available for plotting.")

        frame_count = np.asarray(self._visualization_data["result"].time, dtype=float).size
        if self._animation_playing and self._animation_after_id is not None:
            return (self._animation_frame_index - 1) % frame_count
        return int(np.clip(self._animation_frame_index, 0, frame_count - 1))

    def _top_view_plot_data(self) -> tuple[dict[str, np.ndarray], int]:
        """Return the top-view arm trajectories and the highlighted frame index."""

        if self._visualization_data is None:
            raise RuntimeError("No simulation available for plotting.")

        top_view = arm_top_view_trajectories(self._visualization_data["trajectories"])
        return top_view, self._current_plot_frame_index()

    def _refresh_top_view_plot(self) -> None:
        """Draw the dedicated top-view visualization of the arm motion."""

        if self._visualization_data is None:
            return

        top_view, frame_index = self._top_view_plot_data()
        current_time = float(self._visualization_data["result"].time[frame_index])
        self._plot_axis.clear()

        all_points = np.concatenate(list(top_view.values()), axis=0)
        minimum = np.min(all_points, axis=0)
        maximum = np.max(all_points, axis=0)
        center = 0.5 * (minimum + maximum)
        radius = max(0.25, 0.6 * float(np.max(maximum - minimum)))

        self._plot_axis.axhline(0.0, color="0.7", linestyle="--", linewidth=1.0)
        self._plot_axis.plot(
            top_view["hand_left"][:, 0],
            top_view["hand_left"][:, 1],
            color="tab:red",
            alpha=0.45,
            linewidth=1.8,
            label="Main gauche",
        )
        self._plot_axis.plot(
            top_view["hand_right"][:, 0],
            top_view["hand_right"][:, 1],
            color="tab:blue",
            alpha=0.45,
            linewidth=1.8,
            label="Main droite",
        )

        left_chain = np.vstack(
            [top_view[marker_name][frame_index] for marker_name in TOP_VIEW_LEFT_CHAIN]
        )
        right_chain = np.vstack(
            [top_view[marker_name][frame_index] for marker_name in TOP_VIEW_RIGHT_CHAIN]
        )
        self._plot_axis.plot(
            left_chain[:, 0],
            left_chain[:, 1],
            color="tab:red",
            linewidth=2.5,
            marker="o",
        )
        self._plot_axis.plot(
            right_chain[:, 0],
            right_chain[:, 1],
            color="tab:blue",
            linewidth=2.5,
            marker="o",
        )
        self._plot_axis.scatter(
            [left_chain[-1, 0], right_chain[-1, 0]],
            [left_chain[-1, 1], right_chain[-1, 1]],
            color=["tab:red", "tab:blue"],
            s=45,
            zorder=3,
        )
        self._plot_axis.set_xlim(center[0] - radius, center[0] + radius)
        self._plot_axis.set_ylim(center[1] - radius, center[1] + radius)
        self._plot_axis.set_aspect("equal", adjustable="box")
        if self._animation_reference() == ANIMATION_REFERENCE_OPTIONS[1]:
            x_label = "x mediolateral avec q(root)=0 (m)"
            y_label = "y anteroposterior avec q(root)=0 (m)"
            title_suffix = " | q(root)=0"
        else:
            x_label = "x mediolateral relatif pelvis (m)"
            y_label = "y anteroposterior relatif pelvis (m)"
            title_suffix = ""
        self._plot_axis.set_xlabel(x_label)
        self._plot_axis.set_ylabel(y_label)
        self._plot_axis.set_title(f"Bras hors BTP (dessus) | t = {current_time:.2f} s{title_suffix}")
        self._plot_axis.grid(True, alpha=0.3)
        self._plot_axis.legend(loc="upper right")
        self._plot_canvas.draw_idle()

    def _refresh_plot(self) -> None:
        """Refresh the embedded 2D plot using the latest simulation result."""

        if self._visualization_data is None:
            return

        if self.plot_mode_var.get() == PLOT_MODE_OPTIONS[1]:
            self._refresh_top_view_plot()
            return

        x_data, y_data, x_label, y_label, title, curve_labels = self._plot_data()
        selected_candidates = self._selected_scan_candidate_records()
        self._plot_axis.clear()
        aspect_setter = getattr(self._plot_axis, "set_aspect", None)
        if callable(aspect_setter):
            aspect_setter("auto")
        if selected_candidates:
            styles = ("-", "--")
            widths = (2.2, 2.0)
            alphas = (1.0, 0.9)
            colors = ("tab:red", "tab:orange", "tab:blue", "tab:green")
            for selection_index, candidate in enumerate(selected_candidates):
                simulation = candidate.get("simulation")
                if not isinstance(simulation, AerialSimulationResult):
                    continue
                candidate_x, candidate_y, x_label, y_label, title, curve_labels = self._plot_data_for_result(simulation)
                linestyle = styles[min(selection_index, len(styles) - 1)]
                linewidth = widths[min(selection_index, len(widths) - 1)]
                alpha = alphas[min(selection_index, len(alphas) - 1)]
                label_suffix = (
                    f"t1={float(candidate['values']['right_arm_start']):.2f} s"
                )
                if np.asarray(candidate_y).ndim == 2:
                    for curve_index, curve_label in enumerate(curve_labels or ()):
                        self._plot_axis.plot(
                            candidate_x,
                            candidate_y[:, curve_index],
                            color=colors[curve_index % len(colors)],
                            linewidth=linewidth,
                            linestyle=linestyle,
                            alpha=alpha,
                            label=f"{curve_label} | {label_suffix}",
                        )
                    if self.plot_y_var.get() == "Cinematique bras":
                        self._add_arm_kinematic_bounds_to_plot(colors)
                else:
                    self._plot_axis.plot(
                        candidate_x,
                        candidate_y,
                        color="tab:blue" if selection_index == 0 else "tab:orange",
                        linewidth=linewidth,
                        linestyle=linestyle,
                        alpha=alpha,
                        label=label_suffix,
                    )
            self._plot_axis.legend(loc="best")
        elif np.asarray(y_data).ndim == 2:
            colors = ("tab:red", "tab:orange", "tab:blue", "tab:green")
            for curve_index, curve_label in enumerate(curve_labels or ()):
                self._plot_axis.plot(
                    x_data,
                    y_data[:, curve_index],
                    color=colors[curve_index % len(colors)],
                    linewidth=2.0,
                    label=curve_label,
                )
            if self.plot_y_var.get() == "Cinematique bras":
                self._add_arm_kinematic_bounds_to_plot(colors)
            self._plot_axis.legend(loc="best")
        else:
            self._plot_axis.plot(x_data, y_data, color="tab:blue", linewidth=2.0)
        self._plot_axis.set_xlabel(x_label)
        self._plot_axis.set_ylabel(y_label)
        self._plot_axis.set_title(title)
        self._plot_axis.grid(True, alpha=0.3)
        self._plot_canvas.draw_idle()

    def _refresh_animation_scene(self) -> None:
        """Refresh the 3D scene without rerunning the simulation."""

        if self._visualization_data is None:
            return
        self._prepare_animation_scene()
        self._sync_time_slider_to_frame(self._animation_frame_index)

    def _apply_optimized_values(
        self,
        optimized_values: dict[str, float],
        *,
        prescribed_motion=None,
        status_suffix: str | None = None,
    ) -> None:
        """Push optimized parameters to the GUI, rerun the simulation, and restart the animation."""

        self._set_values(optimized_values)
        self.root.update_idletasks()
        if prescribed_motion is None:
            self._run_simulation()
        elif hasattr(prescribed_motion, "_cached_simulation_result"):
            self._update_from_simulation(
                getattr(prescribed_motion, "_cached_simulation_result"),
                model_path=Path(self._model_path()),
            )
        else:
            self._run_simulation_with_motion(prescribed_motion)
        if status_suffix is not None:
            self.result_var.set(f"{self.result_var.get()} | {status_suffix}")

    @staticmethod
    def _present_external_figure(figure) -> None:
        """Force one external matplotlib figure to appear promptly without blocking Tk."""
        present_external_figure(figure)

    def _kinematic_explorer_candidates(self) -> list[dict[str, object]]:
        """Return the selected scan candidates, or the currently displayed solution if none is selected."""

        selected_candidates = self._selected_scan_candidate_records()
        if selected_candidates:
            return selected_candidates[:2]
        if self._last_simulation is None:
            return []
        return [
            {
                "mode": "Courant",
                "values": self._values_with_current_fixed_parameters(self._current_values()),
                "simulation": self._last_simulation,
            }
        ]

    def _motion_for_kinematic_candidate(self, candidate: dict[str, object]) -> PiecewiseConstantJerkArmMotion:
        """Return the jerk-driven arm motion associated with one explorer candidate."""

        values = self._values_with_current_fixed_parameters(dict(candidate["values"]))
        if "left_plane_jerk" in candidate and "right_plane_jerk" in candidate:
            return self._rebuild_cached_dms_motion(
                values,
                left_plane_jerk=np.asarray(candidate["left_plane_jerk"], dtype=float),
                right_plane_jerk=np.asarray(candidate["right_plane_jerk"], dtype=float),
            )
        return build_piecewise_constant_jerk_arm_motion(
            _variables_from_gui(values),
            total_time=self._standard_optimization_configuration().final_time,
            step=0.02,
        )

    def _kinematic_explorer_payloads(self) -> list[dict[str, object]]:
        """Build plot-ready kinematic payloads for the selected solutions."""

        payloads: list[dict[str, object]] = []
        for selection_index, candidate in enumerate(self._kinematic_explorer_candidates(), start=1):
            motion = self._motion_for_kinematic_candidate(candidate)
            simulation = candidate.get("simulation")
            if isinstance(simulation, AerialSimulationResult):
                sample_times = np.asarray(simulation.time, dtype=float)
            elif self._last_simulation is not None:
                sample_times = np.asarray(self._last_simulation.time, dtype=float)
            else:
                final_time = float(self._standard_optimization_configuration().final_time)
                sample_times = np.arange(0.0, final_time + 0.0025, 0.005, dtype=float)

            trajectories = (
                ("Plan bras gauche", motion.left_plane, 0.0),
                ("Elevation bras gauche", motion.left_elevation, 0.0),
                ("Plan bras droit", motion.right_plane, float(motion.right_arm_start)),
                ("Elevation bras droit", motion.right_elevation, float(motion.right_arm_start)),
            )
            dof_payloads: list[dict[str, object]] = []
            for name, trajectory, offset in trajectories:
                global_starts = float(offset) + np.arange(trajectory.jerks.size, dtype=float) * float(trajectory.step)
                global_step = float(trajectory.step)
                q = np.asarray(trajectory.position(sample_times - float(offset)), dtype=float)
                qdot = np.asarray(trajectory.velocity(sample_times - float(offset)), dtype=float)
                qddot = np.asarray(trajectory.acceleration(sample_times - float(offset)), dtype=float)
                bound = float(np.max(np.abs(np.asarray(trajectory.jerks, dtype=float)))) if trajectory.jerks.size else 0.0
                dof_payloads.append(
                    {
                        "name": name,
                        "jerk_times": global_starts,
                        "jerk_values": np.asarray(trajectory.jerks, dtype=float),
                        "jerk_step": global_step,
                        "jerk_bound": bound,
                        "time": sample_times,
                        "q": q,
                        "qdot": qdot,
                        "qddot": qddot,
                    }
                )
            mode_label = str(candidate.get("mode", f"Selection {selection_index}"))
            start_time = float(candidate["values"]["right_arm_start"])
            payloads.append(
                {
                    "label": f"{mode_label} | t1={start_time:.2f} s",
                    "dofs": dof_payloads,
                }
            )
        return payloads

    def _show_kinematic_explorer_figure(self, payloads: list[dict[str, object]]) -> None:
        """Open one external multi-panel figure comparing jerk-driven arm kinematics."""

        import matplotlib.pyplot as plt

        if not payloads:
            raise ValueError("Aucune solution disponible pour l'exploration cinematique.")

        figure, axes = plt.subplots(
            4,
            4,
            sharex="col",
            figsize=(16.0, 11.0),
            tight_layout=True,
        )
        solution_colors = ("tab:blue", "tab:orange")
        solution_styles = ("-", "--")

        for column_index, column_label in enumerate(KINEMATIC_EXPLORER_COLUMN_LABELS):
            axes[0, column_index].set_title(column_label)

        for row_index, dof_name in enumerate(ARM_KINEMATICS_LABELS):
            axes[row_index, 0].set_ylabel(f"{dof_name}\njerk")
            axes[row_index, 1].set_ylabel("qddot")
            axes[row_index, 2].set_ylabel("qdot")
            axes[row_index, 3].set_ylabel("q")

        legend_handles = []
        legend_labels = []
        for solution_index, payload in enumerate(payloads):
            color = solution_colors[min(solution_index, len(solution_colors) - 1)]
            linestyle = solution_styles[min(solution_index, len(solution_styles) - 1)]
            for row_index, dof_payload in enumerate(payload["dofs"]):
                jerk_axis, qddot_axis, qdot_axis, q_axis = axes[row_index]
                jerk_times = np.asarray(dof_payload["jerk_times"], dtype=float)
                jerk_values = np.asarray(dof_payload["jerk_values"], dtype=float)
                if jerk_values.size:
                    jerk_plot_times = np.append(jerk_times, jerk_times[-1] + float(dof_payload["jerk_step"]))
                    jerk_plot_values = np.append(jerk_values, jerk_values[-1])
                    jerk_axis.step(
                        jerk_plot_times,
                        jerk_plot_values,
                        where="post",
                        color=color,
                        linestyle=linestyle,
                        linewidth=1.8,
                    )
                    jerk_bound = float(dof_payload["jerk_bound"])
                    jerk_axis.axhline(jerk_bound, color=color, linestyle=":", linewidth=1.0, alpha=0.35)
                    jerk_axis.axhline(-jerk_bound, color=color, linestyle=":", linewidth=1.0, alpha=0.35)
                qddot_axis.plot(
                    dof_payload["time"],
                    dof_payload["qddot"],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                )
                qdot_axis.plot(
                    dof_payload["time"],
                    dof_payload["qdot"],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                )
                line = q_axis.plot(
                    dof_payload["time"],
                    dof_payload["q"],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                )[0]
                if row_index == 0:
                    legend_handles.append(line)
                    legend_labels.append(str(payload["label"]))

        for axis_row in axes:
            for axis in axis_row:
                axis.grid(True, alpha=0.3)
        for axis in axes[-1, :]:
            axis.set_xlabel("Temps (s)")
        figure.suptitle("Exploration cinematique des bras", fontsize=14)
        if legend_handles:
            figure.legend(legend_handles, legend_labels, loc="upper center", ncol=max(1, len(legend_handles)))
        self._present_external_figure(figure)

    def _show_kinematic_explorer(self) -> None:
        """Open an external kinematic explorer for the selected or current solutions."""

        payloads = self._kinematic_explorer_payloads()
        if not payloads:
            self.result_var.set("Aucune solution disponible pour explorer la cinematique.")
            return
        self._show_kinematic_explorer_figure(payloads)

    def _optimization_progress(self, message: str) -> None:
        """Update the optimization status line from the Tk thread."""

        self.result_var.set(str(message))
        self.root.update_idletasks()

    def _start_background_optimization(
        self,
        *,
        current_values: dict[str, float],
        mode: str,
        use_cache: bool,
    ) -> None:
        """Launch one optimization worker while keeping the Tk loop responsive."""

        if self._optimization_thread is not None and self._optimization_thread.is_alive():
            self.result_var.set("Une optimisation est deja en cours.")
            return

        self._optimization_queue = queue.Queue()

        def worker() -> None:
            try:
                outcome = self._compute_optimization_outcome(
                    current_values=current_values,
                    mode=mode,
                    use_cache=use_cache,
                    progress_callback=lambda message: self._optimization_queue.put(("progress", str(message))),
                )
            except Exception as error:  # pragma: no cover - exercised through GUI callback reporting
                self._optimization_queue.put(("error", (error, traceback.format_exc())))
                return
            self._optimization_queue.put(("done", outcome))

        self._optimization_thread = threading.Thread(
            target=worker,
            name="best-tilting-plane-optimize",
            daemon=True,
        )
        self._optimization_thread.start()
        self._poll_background_optimization()

    def _poll_background_optimization(self) -> None:
        """Drain worker messages and reschedule the polling while optimization is running."""

        while True:
            try:
                message_type, payload = self._optimization_queue.get_nowait()
            except queue.Empty:
                break

            if message_type == "progress":
                self._optimization_progress(str(payload))
            elif message_type == "error":
                error, formatted_traceback = payload
                print(formatted_traceback, end="")
                self.result_var.set(f"Erreur optimisation: {error}")
                self._optimization_thread = None
            elif message_type == "done":
                self._optimization_thread = None
                self._handle_optimization_outcome(payload)

        if self._optimization_thread is not None and self._optimization_thread.is_alive():
            self._optimization_poll_after_id = self.root.after(100, self._poll_background_optimization)
        else:
            self._optimization_poll_after_id = None

    def _handle_optimization_outcome(self, outcome: dict[str, object]) -> None:
        """Apply one completed optimization result back onto the Tk widgets."""

        scan_figure = outcome.get("scan_figure")
        if isinstance(scan_figure, dict):
            self._schedule_scan_figure(**scan_figure)
        if bool(outcome.get("show_embedded_scan", False)):
            self._show_embedded_scan_plot()

        jerk_diagnostic = outcome.get("jerk_diagnostic")
        if isinstance(jerk_diagnostic, dict):
            self._schedule_dms_jerk_diagnostic_figure(**jerk_diagnostic)

        apply_kwargs = {"status_suffix": outcome.get("status_suffix")}
        if outcome.get("prescribed_motion") is not None:
            apply_kwargs["prescribed_motion"] = outcome.get("prescribed_motion")
        self._apply_optimized_values(
            outcome["optimized_values"],
            **apply_kwargs,
        )

    def _compute_optimization_outcome(
        self,
        *,
        current_values: dict[str, float],
        mode: str,
        use_cache: bool,
        progress_callback=None,
    ) -> dict[str, object]:
        """Compute one optimization result without touching Tk widgets directly."""

        initial_guess = _variables_from_gui(current_values)
        mode_label = _optimization_mode_label(mode)

        def report(message: str) -> None:
            if progress_callback is not None:
                progress_callback(message)

        if _is_three_d_optimization_mode(mode) and use_cache:
            cached_dms_solution = self._load_cached_dms_solution()
            if cached_dms_solution is not None:
                cached_values, cached_motion, cached_final_twist_turns, cached_solver_status, cached_scan_data = (
                    cached_dms_solution
                )
                return {
                    "optimized_values": cached_values,
                    "prescribed_motion": cached_motion,
                    "status_suffix": (
                        f"optimum {mode_label} charge depuis le cache: "
                        f"{cached_final_twist_turns:.2f} tours ({cached_solver_status})"
                    ),
                    "scan_figure": (
                        None
                        if cached_scan_data is None
                        else {
                            "start_times": cached_scan_data["start_times"],
                            "final_twist_turns": cached_scan_data["final_twist_turns"],
                            "objective_values": cached_scan_data["objective_values"],
                            "success_mask": cached_scan_data["success_mask"],
                            "best_start_time": cached_values["right_arm_start"],
                        }
                    ),
                }
        elif not _is_three_d_optimization_mode(mode) and use_cache:
            cached_values = self._load_cached_optimized_values()
            if cached_values is not None:
                cached_scan_data = self._load_cached_optimized_scan_data()
                return {
                    "optimized_values": cached_values,
                    "status_suffix": "optimum charge depuis le cache",
                    "scan_figure": (
                        None
                        if cached_scan_data is None
                        else {
                            "start_times": cached_scan_data["start_times"],
                            "final_twist_turns": cached_scan_data["final_twist_turns"],
                            "objective_values": cached_scan_data["objective_values"],
                            "best_start_time": cached_values["right_arm_start"],
                        }
                    ),
                    "show_embedded_scan": cached_scan_data is not None,
                }

        if _is_three_d_optimization_mode(mode):
            optimizer = DirectMultipleShootingOptimizer.from_builder(
                self._model_path(),
                configuration=self._standard_optimization_configuration(),
                shooting_step=DMS_SHOOTING_STEP,
                jerk_regularization=DMS_JERK_REGULARIZATION,
                objective_mode=_three_d_objective_mode(mode),
                btp_deviation_weight=DMS_BTP_DEVIATION_WEIGHT,
                twist_rate_lagrange_weight=DMS_TWIST_RATE_LAGRANGE_WEIGHT,
            )
            candidate_start_times = np.asarray(optimizer.candidate_start_times(), dtype=float)
            cached_progress = None if not use_cache else self._load_cached_dms_progress()
            start_index = 0
            scan_start_times: list[float] = []
            scan_final_twist_turns: list[float] = []
            scan_objective_values: list[float] = []
            scan_success_mask: list[bool] = []
            scan_candidate_solutions: list[dict[str, object]] = []
            scan_candidate_results: list[object | None] = []
            previous_result = None
            best_result = None
            best_warm_start_result = None
            multistart_reference_time = self._dms_multistart_reference_start_time(
                initial_guess=initial_guess,
                candidate_start_times=candidate_start_times,
                use_cache=use_cache,
                report=report,
            )

            if cached_progress is not None:
                scan_start_times = list(cached_progress["start_times"])
                scan_final_twist_turns = list(cached_progress["final_twist_turns"])
                scan_objective_values = list(cached_progress["objective_values"])
                scan_success_mask = list(cached_progress["success_mask"])
                scan_candidate_results = [None] * len(scan_start_times)
                scan_candidate_solutions = (
                    []
                    if cached_progress.get("scan_candidate_solutions") is None
                    else [
                        self._scan_candidate_record(
                            optimized_values=dict(candidate["values"]),
                            simulation_result=candidate.get("simulation"),
                            mode=mode,
                            final_twist_turns=float(candidate["final_twist_turns"]),
                            objective=float(candidate["objective"]),
                            solver_status=str(candidate["solver_status"]),
                            success=bool(candidate["success"]),
                            left_plane_jerk=(
                                None
                                if "left_plane_jerk" not in candidate
                                else np.asarray(candidate["left_plane_jerk"], dtype=float)
                            ),
                            right_plane_jerk=(
                                None
                                if "right_plane_jerk" not in candidate
                                else np.asarray(candidate["right_plane_jerk"], dtype=float)
                            ),
                        )
                        for candidate in cached_progress["scan_candidate_solutions"]
                    ]
                )
                start_index = len(scan_start_times)
                best_motion = self._rebuild_cached_dms_motion(
                    cached_progress["optimized_values"],
                    left_plane_jerk=cached_progress["left_plane_jerk"],
                    right_plane_jerk=cached_progress["right_plane_jerk"],
                )
                cached_best_simulation = cached_progress.get("cached_simulation")
                if cached_best_simulation is not None:
                    setattr(best_motion, "_cached_simulation_result", cached_best_simulation)
                if any(scan_success_mask):
                    best_objective = min(
                        objective
                        for objective, success in zip(scan_objective_values, scan_success_mask, strict=False)
                        if success
                    )
                    best_success = True
                else:
                    best_objective = min(scan_objective_values)
                    best_success = False
                best_result = SimpleNamespace(
                    variables=_variables_from_gui(cached_progress["optimized_values"]),
                    right_arm_start_node_index=int(
                        round(float(cached_progress["optimized_values"]["right_arm_start"]) / DMS_SHOOTING_STEP)
                    ),
                    prescribed_motion=best_motion,
                    simulation=cached_best_simulation,
                    left_plane_jerk=np.asarray(cached_progress["left_plane_jerk"], dtype=float),
                    right_plane_jerk=np.asarray(cached_progress["right_plane_jerk"], dtype=float),
                    final_twist_turns=float(cached_progress["final_twist_turns_best"]),
                    solver_status=str(cached_progress["solver_status_best"]),
                    objective=float(best_objective),
                    success=best_success,
                )
                if cached_progress["last_warm_start_primal"] is not None:
                    previous_result = SimpleNamespace(
                        warm_start_primal=np.asarray(cached_progress["last_warm_start_primal"], dtype=float),
                        warm_start_lam_x=(
                            None
                            if cached_progress["last_warm_start_lam_x"] is None
                            else np.asarray(cached_progress["last_warm_start_lam_x"], dtype=float)
                        ),
                        warm_start_lam_g=(
                            None
                            if cached_progress["last_warm_start_lam_g"] is None
                            else np.asarray(cached_progress["last_warm_start_lam_g"], dtype=float)
                        ),
                    )
                    best_warm_start_result = previous_result

            for index in range(start_index, len(candidate_start_times)):
                current_start_time = float(candidate_start_times[index])
                report(
                    f"{mode_label} en cours... t1={current_start_time:.2f} s "
                    f"({index + 1}/{len(candidate_start_times)})"
                )
                warm_start_seed = best_warm_start_result if best_warm_start_result is not None else previous_result
                if (
                    multistart_reference_time is not None
                    and np.isclose(current_start_time, multistart_reference_time)
                    and hasattr(optimizer, "solve_fixed_start_multistart")
                ):
                    current_result = optimizer.solve_fixed_start_multistart(
                        initial_guess,
                        right_arm_start=current_start_time,
                        start_count=MULTISTART_START_COUNT,
                        previous_result=warm_start_seed,
                        max_iter=50,
                        print_level=5,
                        print_time=True,
                        show_jerk_diagnostics=False,
                    )
                else:
                    current_result = optimizer.solve_fixed_start(
                        initial_guess,
                        right_arm_start=current_start_time,
                        previous_result=warm_start_seed,
                        max_iter=50,
                        print_level=5,
                        print_time=True,
                        show_jerk_diagnostics=False,
                    )
                scan_start_times.append(current_start_time)
                scan_final_twist_turns.append(current_result.final_twist_turns)
                scan_objective_values.append(current_result.objective)
                scan_success_mask.append(bool(current_result.success))
                scan_candidate_solutions.append(
                    self._scan_candidate_record(
                        optimized_values=_gui_values_from_variables(current_result.variables),
                        simulation_result=getattr(current_result, "simulation", None),
                        mode=mode,
                        final_twist_turns=current_result.final_twist_turns,
                        objective=current_result.objective,
                        solver_status=current_result.solver_status,
                        success=bool(current_result.success),
                        left_plane_jerk=np.asarray(current_result.left_plane_jerk, dtype=float),
                        right_plane_jerk=np.asarray(current_result.right_plane_jerk, dtype=float),
                    )
                )
                scan_candidate_results.append(current_result)
                previous_result = current_result

                if _dms_result_is_better(current_result, best_result):
                    best_result = current_result
                if (
                    getattr(current_result, "warm_start_primal", None) is not None
                    and _dms_result_is_better(current_result, best_warm_start_result)
                ):
                    best_warm_start_result = current_result

                optimized_values_checkpoint = _gui_values_from_variables(best_result.variables)
                self._store_cached_dms_progress(
                    optimized_values_checkpoint,
                    left_plane_jerk=np.asarray(best_result.left_plane_jerk, dtype=float),
                    right_plane_jerk=np.asarray(best_result.right_plane_jerk, dtype=float),
                    simulation_result=getattr(best_result, "simulation", None),
                    scan_start_times=np.asarray(scan_start_times, dtype=float),
                    scan_final_twist_turns=np.asarray(scan_final_twist_turns, dtype=float),
                    scan_objective_values=np.asarray(scan_objective_values, dtype=float),
                    scan_success_mask=np.asarray(scan_success_mask, dtype=bool),
                    scan_candidate_solutions=list(scan_candidate_solutions),
                    last_completed_index=index,
                    last_warm_start_primal=(
                        None
                        if getattr(best_warm_start_result, "warm_start_primal", None) is None
                        else np.asarray(best_warm_start_result.warm_start_primal, dtype=float)
                    ),
                    last_warm_start_lam_x=(
                        None
                        if getattr(best_warm_start_result, "warm_start_lam_x", None) is None
                        else np.asarray(best_warm_start_result.warm_start_lam_x, dtype=float)
                    ),
                    last_warm_start_lam_g=(
                        None
                        if getattr(best_warm_start_result, "warm_start_lam_g", None) is None
                        else np.asarray(best_warm_start_result.warm_start_lam_g, dtype=float)
                    ),
                    final_twist_turns=float(best_result.final_twist_turns),
                    solver_status=str(best_result.solver_status),
                )

            if mode == "Optimize 3D BTP" and len(scan_start_times) >= 3:
                for index in range(1, len(scan_start_times) - 1):
                    current_twist = float(scan_final_twist_turns[index])
                    neighbor_mean = 0.5 * (
                        float(scan_final_twist_turns[index - 1]) + float(scan_final_twist_turns[index + 1])
                    )
                    if abs(current_twist - neighbor_mean) < THREE_D_BTP_DISCONTINUITY_THRESHOLD:
                        continue
                    left_candidate = scan_candidate_solutions[index - 1]
                    right_candidate = scan_candidate_solutions[index + 1]
                    better_neighbor = (
                        scan_candidate_results[index - 1]
                        if scan_candidate_results[index - 1] is not None
                        and (
                            scan_candidate_results[index + 1] is None
                            or _dms_result_is_better(scan_candidate_results[index - 1], scan_candidate_results[index + 1])
                        )
                        else scan_candidate_results[index + 1]
                    )
                    if better_neighbor is None:
                        continue
                    try:
                        rerun = optimizer.solve_fixed_start(
                            initial_guess,
                            right_arm_start=float(scan_start_times[index]),
                            previous_result=better_neighbor,
                            max_iter=50,
                            print_level=5,
                            print_time=True,
                            show_jerk_diagnostics=False,
                        )
                    except Exception:
                        continue
                    if float(rerun.objective) >= float(scan_objective_values[index]):
                        continue
                    scan_final_twist_turns[index] = float(rerun.final_twist_turns)
                    scan_objective_values[index] = float(rerun.objective)
                    scan_success_mask[index] = bool(rerun.success)
                    scan_candidate_results[index] = rerun
                    scan_candidate_solutions[index] = self._scan_candidate_record(
                        optimized_values=_gui_values_from_variables(rerun.variables),
                        simulation_result=getattr(rerun, "simulation", None),
                        mode=mode,
                        final_twist_turns=rerun.final_twist_turns,
                        objective=rerun.objective,
                        solver_status=rerun.solver_status,
                        success=bool(rerun.success),
                        left_plane_jerk=np.asarray(rerun.left_plane_jerk, dtype=float),
                        right_plane_jerk=np.asarray(rerun.right_plane_jerk, dtype=float),
                    )
                    if _dms_result_is_better(rerun, best_result):
                        best_result = rerun
                    if (
                        getattr(rerun, "warm_start_primal", None) is not None
                        and _dms_result_is_better(rerun, best_warm_start_result)
                    ):
                        best_warm_start_result = rerun

            if best_result is None:
                raise RuntimeError("No DMS candidate was evaluated.")
            result = best_result
            if getattr(result, "simulation", None) is not None:
                setattr(result.prescribed_motion, "_cached_simulation_result", result.simulation)
                plane_q = np.asarray(result.simulation.q[:, [6, 8]], dtype=float)
                print(
                    "3D arm-plane amplitudes: "
                    f"left={np.rad2deg(np.max(np.abs(plane_q[:, 0]))):.2f} deg, "
                    f"right={np.rad2deg(np.max(np.abs(plane_q[:, 1]))):.2f} deg"
                )
            optimized_values = _gui_values_from_variables(result.variables)
            scan_data = self._scan_data_from_lists(
                start_times=scan_start_times,
                final_twist_turns=scan_final_twist_turns,
                objective_values=scan_objective_values,
                success_mask=scan_success_mask,
            )
            self._store_cached_dms_solution(
                optimized_values,
                left_plane_jerk=result.left_plane_jerk,
                right_plane_jerk=result.right_plane_jerk,
                simulation_result=getattr(result, "simulation", None),
                scan_start_times=scan_data["start_times"],
                scan_final_twist_turns=scan_data["final_twist_turns"],
                scan_objective_values=scan_data["objective_values"],
                scan_success_mask=scan_data["success_mask"],
                scan_candidate_solutions=list(scan_candidate_solutions),
                final_twist_turns=result.final_twist_turns,
                solver_status=result.solver_status,
            )
            return {
                "optimized_values": optimized_values,
                "prescribed_motion": result.prescribed_motion,
                "status_suffix": f"optimum {mode_label}: {result.final_twist_turns:.2f} tours ({result.solver_status})",
                "scan_figure": {
                    "start_times": scan_data["start_times"],
                    "final_twist_turns": scan_data["final_twist_turns"],
                    "objective_values": scan_data["objective_values"],
                    "success_mask": scan_data["success_mask"],
                    "best_start_time": result.variables.right_arm_start,
                },
                "jerk_diagnostic": {"optimizer": optimizer, "result": result},
            }

        sweep, result, optimized_values = self._evaluate_two_d_sweep(initial_guess=initial_guess)
        scan_candidate_solutions = [
            self._scan_candidate_record(
                optimized_values=_gui_values_from_variables(candidate_result.variables),
                simulation_result=getattr(candidate_result, "simulation", None),
                mode="Optimize 2D",
                final_twist_turns=candidate_result.final_twist_turns,
                objective=candidate_result.objective,
                solver_status=candidate_result.solver_status,
                success=bool(candidate_result.success),
            )
            for candidate_result in getattr(sweep, "candidate_results", ())
        ]
        self._store_cached_optimized_values(
            optimized_values,
            final_twist_turns=result.final_twist_turns,
            solver_status=result.solver_status,
            scan_start_times=sweep.start_times,
            scan_final_twist_turns=sweep.final_twist_turns,
            scan_objective_values=sweep.objective_values,
            scan_candidate_solutions=(scan_candidate_solutions or None),
        )
        return {
            "optimized_values": optimized_values,
            "status_suffix": f"optimum balayage 1D: {result.final_twist_turns:.2f} tours ({result.solver_status})",
            "scan_figure": {
                "start_times": sweep.start_times,
                "final_twist_turns": sweep.final_twist_turns,
                "objective_values": sweep.objective_values,
                "best_start_time": result.variables.right_arm_start,
            },
            "show_embedded_scan": True,
        }

    def _report_callback_exception(self, exception_class, exception, trace) -> None:
        """Report one Tkinter callback exception without leaving the GUI silent."""

        traceback.print_exception(exception_class, exception, trace)
        self.result_var.set(f"Erreur GUI: {exception}")

    def _optimize_strategy(self) -> None:
        """Optimize the current strategy with IPOPT, then update the GUI and rerun the simulation."""

        try:
            current_values = self._current_values()
            mode = self.optimization_mode_var.get()
            use_cache = not self._should_ignore_optimization_cache()
            self._auto_runner.cancel()
            self._optimization_progress("Optimisation en cours... voir les iterations IPOPT dans le terminal.")
            if getattr(self, "_run_optimization_in_background", False):
                self._start_background_optimization(
                    current_values=current_values,
                    mode=mode,
                    use_cache=use_cache,
                )
                return
            outcome = self._compute_optimization_outcome(
                current_values=current_values,
                mode=mode,
                use_cache=use_cache,
                progress_callback=self._optimization_progress,
            )
            self._handle_optimization_outcome(outcome)
        except Exception as error:
            traceback.print_exc()
            self.result_var.set(f"Erreur optimisation: {error}")


def launch_gui() -> None:
    """Launch the Tkinter GUI."""

    root = tk.Tk()
    BestTiltingPlaneApp(root)
    root.mainloop()
