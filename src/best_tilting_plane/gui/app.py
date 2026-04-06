"""Interactive GUI for the predictive twisting simulation."""

from __future__ import annotations

import json
import tkinter as tk
import traceback
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk
from types import SimpleNamespace

import numpy as np
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
    DEFAULT_DMS_JERK_REGULARIZATION,
    show_dms_jerk_bounds_figure,
)
from best_tilting_plane.simulation import (
    PiecewiseConstantJerkArmMotion,
    PiecewiseConstantJerkTrajectory,
    PredictiveAerialTwistSimulator,
    SimulationConfiguration,
    TwistOptimizationVariables,
    show_first_arm_piecewise_constant_comparison,
)
from best_tilting_plane.visualization import (
    SKELETON_CONNECTIONS,
    arm_btp_reference_trajectories,
    arm_deviation_from_frames,
    arm_top_view_trajectories,
    best_tilting_plane_corners,
    marker_trajectories,
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
)
GUI_FIXED_VALUES = {
    "left_plane_initial": 0.0,
    "left_plane_final": 0.0,
    "right_plane_initial": 0.0,
    "right_plane_final": 0.0,
}
GUI_VALUE_NAMES = ("right_arm_start", *GUI_FIXED_VALUES.keys())
PLOT_X_OPTIONS = ("Temps", "Somersault", "Vrille")
PLOT_MODE_OPTIONS = ("Courbe", "Bras hors BTP (dessus)")
ANIMATION_MODE_OPTIONS = ("Animation 3D", "Bras / BTP")
ANIMATION_REFERENCE_OPTIONS = ("Global", "Racine", "Best tilting plane")
OPTIMIZATION_MODE_OPTIONS = ("Optimize 2D", "Optimize DMS")
PLOT_Y_OPTIONS = (
    "Somersault",
    "Tilt",
    "Twist",
    "Cinematique bras",
    "Vitesses bras",
    "Deviation bras gauche",
    "Deviation bras droit",
    "Vrilles selon t1",
)
ROOT_INITIAL_OPTIONS = ("Avec q racine(0)=0", "Sans q racine(0)=0")
TOP_VIEW_LEFT_CHAIN = ("shoulder_left", "elbow_left", "wrist_left", "hand_left")
TOP_VIEW_RIGHT_CHAIN = ("shoulder_right", "elbow_right", "wrist_right", "hand_right")
DEFAULT_CAMERA_ELEVATION_DEG = 20.0
DEFAULT_CAMERA_AZIMUTH_DEG = -60.0
BTP_CAMERA_ELEVATION_DEG = 22.0
BTP_CAMERA_AZIMUTH_DEG = -35.0
ROOT_VIEW_CAMERA_ELEVATION_DEG = 0.0
ROOT_VIEW_CAMERA_AZIMUTH_DEG = -90.0
ALL_FRAME_SEGMENTS = tuple(
    dict.fromkeys(ARM_SEGMENTS_FOR_VISUALIZATION + ARM_SEGMENTS_FOR_DEVIATION)
)
ANIMATION_INTERVAL_MS = 35
STANDARD_RK4_STEP = 0.005
OPTIMIZATION_CACHE_VERSION = 2
DMS_SHOOTING_STEP = 0.02
DMS_ACTIVE_DURATION = 0.3
DMS_SCAN_START = 0.16
DMS_SCAN_END = 0.36
DMS_JERK_REGULARIZATION = DEFAULT_DMS_JERK_REGULARIZATION
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
    "Optimize DMS": {"color": "tab:orange", "marker": "s", "label": "Optimize DMS"},
}


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
        display.rowconfigure(2, weight=2)

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

        self.show_btp = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls,
            text="Afficher le best tilting plane",
            variable=self.show_btp,
            command=self._refresh_animation_scene,
        ).grid(row=len(SLIDER_DEFINITIONS), column=0, columnspan=3, sticky="w", pady=(8, 4))

        ttk.Label(controls, text="Repere animation").grid(
            row=len(SLIDER_DEFINITIONS) + 1, column=0, sticky="w", pady=(8, 4)
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
            row=len(SLIDER_DEFINITIONS) + 1, column=1, columnspan=2, sticky="ew", pady=(8, 4)
        )
        animation_reference_box.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._on_animation_reference_change(),
        )
        self._apply_animation_reference(self.animation_reference_var.get())

        ttk.Label(controls, text="Mode figure").grid(
            row=len(SLIDER_DEFINITIONS) + 2, column=0, sticky="w", pady=4
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
            row=len(SLIDER_DEFINITIONS) + 2, column=1, columnspan=2, sticky="ew", pady=4
        )
        plot_mode_box.bind("<<ComboboxSelected>>", lambda _event: self._refresh_plot())

        ttk.Label(controls, text="Figure x").grid(
            row=len(SLIDER_DEFINITIONS) + 3, column=0, sticky="w", pady=4
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
            row=len(SLIDER_DEFINITIONS) + 3, column=1, columnspan=2, sticky="ew", pady=4
        )
        plot_x_box.bind("<<ComboboxSelected>>", lambda _event: self._refresh_plot())

        ttk.Label(controls, text="Figure y").grid(
            row=len(SLIDER_DEFINITIONS) + 4, column=0, sticky="w", pady=4
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
            row=len(SLIDER_DEFINITIONS) + 4, column=1, columnspan=2, sticky="ew", pady=4
        )
        plot_y_box.bind("<<ComboboxSelected>>", lambda _event: self._refresh_plot())

        ttk.Button(controls, text="Simulate", command=self._run_simulation).grid(
            row=len(SLIDER_DEFINITIONS) + 5, column=0, sticky="w", pady=(10, 0)
        )
        ttk.Button(
            controls,
            text="Comparer jerk bras 1",
            command=self._show_first_arm_jerk_comparison,
        ).grid(row=len(SLIDER_DEFINITIONS) + 5, column=1, columnspan=2, sticky="ew", pady=(10, 0))
        self.ignore_optimization_cache_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            controls,
            text="Ignorer le cache optimum",
            variable=self.ignore_optimization_cache_var,
        ).grid(row=len(SLIDER_DEFINITIONS) + 6, column=0, columnspan=3, sticky="w", pady=(10, 0))
        self.optimization_mode_var = tk.StringVar(value=OPTIMIZATION_MODE_OPTIONS[0])
        optimization_mode_box = ttk.Combobox(
            controls,
            textvariable=self.optimization_mode_var,
            values=OPTIMIZATION_MODE_OPTIONS,
            state="readonly",
            width=18,
        )
        optimization_mode_box.grid(
            row=len(SLIDER_DEFINITIONS) + 7, column=0, columnspan=2, sticky="ew", pady=(10, 0), padx=(0, 8)
        )
        ttk.Button(controls, text="Optimize", command=self._optimize_strategy).grid(
            row=len(SLIDER_DEFINITIONS) + 7, column=2, sticky="w", pady=(10, 0)
        )

        self.result_var = tk.StringVar(value="Aucune simulation lancée.")
        ttk.Label(controls, textvariable=self.result_var, wraplength=360, justify="left").grid(
            row=len(SLIDER_DEFINITIONS) + 8, column=0, columnspan=3, sticky="w", pady=(10, 0)
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
            row=len(SLIDER_DEFINITIONS) + 9, column=0, columnspan=3, sticky="w", pady=(8, 0)
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

        self._plot_figure = Figure(figsize=(8.0, 3.5), tight_layout=True)
        self._plot_axis = self._plot_figure.add_subplot(111)
        self._plot_canvas = FigureCanvasTkAgg(self._plot_figure, master=display)
        self._plot_canvas.get_tk_widget().grid(row=2, column=0, sticky="nsew")

        self._line_artists: tuple[object, ...] = ()
        self._frame_artists: dict[str, tuple[object, object, object]] = {}
        self._btp_chain_artists: dict[str, object] = {}
        self._btp_path_artists: dict[str, object] = {}
        self._btp_marker_artists: dict[str, object] = {}
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

        return SimulationConfiguration(integrator="rk4", rk4_step=STANDARD_RK4_STEP)

    def _optimization_cache_path(self) -> Path:
        """Return the JSON cache path used to store optimal strategies."""

        return self._model_path().with_name("optimization_cache.json")

    def _optimization_cache_key(self) -> str:
        """Return the cache key associated with the current optimization mode."""

        return self._optimization_cache_key_for_mode(self.optimization_mode_var.get())

    @staticmethod
    def _optimization_cache_key_for_mode(mode: str) -> str:
        """Return the cache key associated with one optimization mode."""

        return mode.lower().replace(" ", "_")

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
            "integrator": configuration.integrator,
            "rk4_step": float(configuration.rk4_step) if configuration.rk4_step is not None else None,
        }
        if mode == "Optimize DMS":
            signature["dms_shooting_step"] = DMS_SHOOTING_STEP
            signature["dms_active_duration"] = DMS_ACTIVE_DURATION
            signature["dms_scan_start"] = DMS_SCAN_START
            signature["dms_scan_end"] = DMS_SCAN_END
            signature["dms_jerk_regularization"] = DMS_JERK_REGULARIZATION
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

    def _load_cached_optimized_values(self) -> dict[str, float] | None:
        """Return cached optimized GUI values when the stored signature matches the current setup."""

        cache = self._read_optimization_cache_file()
        record = cache["records"].get(self._optimization_cache_key())
        if not isinstance(record, dict):
            return None
        if record.get("signature") != self._optimization_cache_signature():
            return None
        values = record.get("values")
        if not isinstance(values, dict):
            return None
        expected_names = set(GUI_VALUE_NAMES)
        if set(values) != expected_names:
            return None
        try:
            return {name: float(values[name]) for name in expected_names}
        except (TypeError, ValueError):
            return None

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
        record = cache["records"].get(self._optimization_cache_key_for_mode(mode))
        if not isinstance(record, dict):
            return None
        if record.get("signature") != self._optimization_cache_signature_for_mode(mode):
            return None
        values = record.get("values")
        if not isinstance(values, dict):
            return None
        try:
            best_start_time = float(values["right_arm_start"])
        except (KeyError, TypeError, ValueError):
            return None
        scan_start_times = record.get("scan_start_times")
        scan_final_twist_turns = record.get("scan_final_twist_turns")
        scan_objective_values = record.get("scan_objective_values")
        scan_success_mask = record.get("scan_success_mask")
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
            return {
                "start_times": [float(value) for value in scan_start_times],
                "final_twist_turns": [float(value) for value in scan_final_twist_turns],
                "objective_values": [float(value) for value in scan_objective_values],
                "success_mask": success_mask,
                "best_start_time": best_start_time,
                "mode": mode,
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

    def _show_dms_sweep_figure(
        self,
        *,
        start_times: list[float] | np.ndarray,
        final_twist_turns: list[float] | np.ndarray,
        objective_values: list[float] | np.ndarray,
        success_mask: list[bool] | np.ndarray,
        best_start_time: float,
    ) -> None:
        """Open the discrete DMS sweep figure in an external matplotlib window."""

        if len(start_times) == 0:
            return
        primary_scan = {
            "mode": "Optimize DMS",
            "start_times": np.asarray(start_times, dtype=float),
            "final_twist_turns": np.asarray(final_twist_turns, dtype=float),
            "objective_values": np.asarray(objective_values, dtype=float),
            "success_mask": np.asarray(success_mask, dtype=bool),
            "best_start_time": float(best_start_time),
        }
        comparison_scan = self._load_cached_scan_bundle_for_mode("Optimize 2D")
        self._show_scan_comparison_figure(primary_scan=primary_scan, comparison_scan=comparison_scan)

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

        after_idle = getattr(self.root, "after_idle", None)
        if callable(after_idle):
            after_idle(safe_callback)
        else:
            self.root.after(0, safe_callback)

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
        """Switch the embedded figure to the `vrilles selon t1` view and refresh it."""

        if not hasattr(self, "plot_y_var"):
            return
        self.plot_y_var.set("Vrilles selon t1")
        self._refresh_plot()

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
        left_lower_bounds, left_upper_bounds, right_lower_bounds, right_upper_bounds = optimizer._global_jerk_bounds(
            right_start_node_index=int(result.right_arm_start_node_index)
        )
        left_jerk = np.zeros(optimizer.interval_count, dtype=float)
        right_jerk = np.zeros(optimizer.interval_count, dtype=float)
        left_jerk[: len(result.left_plane_jerk)] = np.asarray(result.left_plane_jerk, dtype=float)
        right_start = int(result.right_arm_start_node_index)
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
        comparison_scan = self._load_cached_scan_bundle_for_mode("Optimize DMS")
        self._show_scan_comparison_figure(primary_scan=primary_scan, comparison_scan=comparison_scan)

    @staticmethod
    def _show_scan_comparison_figure(
        *,
        primary_scan: dict[str, object],
        comparison_scan: dict[str, object] | None = None,
    ) -> None:
        """Open one external figure comparing the 2D and DMS scans on the same axes."""

        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(1, 1, figsize=(8.0, 4.8), tight_layout=True)
        datasets = [primary_scan]
        if comparison_scan is not None and comparison_scan.get("mode") != primary_scan.get("mode"):
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
        axis.set_title("Comparaison des scans 2D et DMS")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
        figure.canvas.draw_idle()
        if "agg" not in plt.get_backend().lower():
            plt.show(block=False)

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

        if self.optimization_mode_var.get() == "Optimize DMS":
            if success_mask is None:
                raise ValueError("The DMS scan figure requires a success mask.")
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
        record = cache["records"].get(self._optimization_cache_key())
        if not isinstance(record, dict):
            return None
        if record.get("signature") != self._optimization_cache_signature():
            return None
        if bool(record.get("in_progress", False)):
            return None
        values = record.get("values")
        left_plane_jerk = record.get("left_plane_jerk")
        right_plane_jerk = record.get("right_plane_jerk")
        if not isinstance(values, dict) or not isinstance(left_plane_jerk, list) or not isinstance(right_plane_jerk, list):
            return None
        expected_names = set(GUI_VALUE_NAMES)
        if set(values) != expected_names:
            return None
        try:
            optimized_values = {name: float(values[name]) for name in expected_names}
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
        return optimized_values, motion, final_twist_turns, solver_status, scan_data

    def _load_cached_dms_progress(self) -> dict[str, object] | None:
        """Return one resumable DMS checkpoint when the stored signature matches the current setup."""

        cache = self._read_optimization_cache_file()
        record = cache["progress_records"].get(self._optimization_cache_key())
        if record is None:
            record = cache["records"].get(self._optimization_cache_key())
        if not isinstance(record, dict):
            return None
        if record.get("signature") != self._optimization_cache_signature():
            return None
        if not bool(record.get("in_progress", False)):
            return None

        scan_start_times = record.get("scan_start_times")
        scan_final_twist_turns = record.get("scan_final_twist_turns")
        scan_objective_values = record.get("scan_objective_values")
        scan_success_mask = record.get("scan_success_mask")
        values = record.get("values")
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
            or not isinstance(values, dict)
            or not isinstance(left_plane_jerk, list)
            or not isinstance(right_plane_jerk, list)
        ):
            return None

        expected_names = set(GUI_VALUE_NAMES)
        if set(values) != expected_names:
            return None

        try:
            optimized_values = {name: float(values[name]) for name in expected_names}
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
    ) -> None:
        """Persist optimized GUI values for reuse in later GUI sessions."""

        cache_path = self._optimization_cache_path()
        cache = self._read_optimization_cache_file()
        record = {
            "signature": self._optimization_cache_signature(),
            "values": {name: float(value) for name, value in optimized_values.items()},
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
        cache["records"][self._optimization_cache_key()] = record
        cache["progress_records"].pop(self._optimization_cache_key(), None)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")

    def _store_cached_dms_solution(
        self,
        optimized_values: dict[str, float],
        *,
        left_plane_jerk: np.ndarray,
        right_plane_jerk: np.ndarray,
        scan_start_times: np.ndarray,
        scan_final_twist_turns: np.ndarray,
        scan_objective_values: np.ndarray,
        scan_success_mask: np.ndarray,
        final_twist_turns: float,
        solver_status: str,
    ) -> None:
        """Persist one DMS optimum for reuse in later GUI sessions."""

        cache_path = self._optimization_cache_path()
        cache = self._read_optimization_cache_file()
        cache["records"][self._optimization_cache_key()] = {
            "signature": self._optimization_cache_signature(),
            "in_progress": False,
            "values": {name: float(value) for name, value in optimized_values.items()},
            "left_plane_jerk": np.asarray(left_plane_jerk, dtype=float).tolist(),
            "right_plane_jerk": np.asarray(right_plane_jerk, dtype=float).tolist(),
            "scan_start_times": np.asarray(scan_start_times, dtype=float).tolist(),
            "scan_final_twist_turns": np.asarray(scan_final_twist_turns, dtype=float).tolist(),
            "scan_objective_values": np.asarray(scan_objective_values, dtype=float).tolist(),
            "scan_success_mask": np.asarray(scan_success_mask, dtype=bool).tolist(),
            "final_twist_turns": float(final_twist_turns),
            "solver_status": str(solver_status),
        }
        cache["progress_records"].pop(self._optimization_cache_key(), None)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")

    def _store_cached_dms_progress(
        self,
        optimized_values: dict[str, float],
        *,
        left_plane_jerk: np.ndarray,
        right_plane_jerk: np.ndarray,
        scan_start_times: np.ndarray,
        scan_final_twist_turns: np.ndarray,
        scan_objective_values: np.ndarray,
        scan_success_mask: np.ndarray,
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
        cache["progress_records"][self._optimization_cache_key()] = {
            "signature": self._optimization_cache_signature(),
            "in_progress": True,
            "values": {name: float(value) for name, value in optimized_values.items()},
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
        }
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
        self._refresh_plot()
        self._start_animation_loop()

    def _display_q_history(self, result) -> np.ndarray:
        """Return the generalized coordinates used for display."""

        q_history = np.asarray(result.q, dtype=float).copy()
        if self._animation_reference() == ANIMATION_REFERENCE_OPTIONS[1]:
            q_history[:, :6] = 0.0
        return q_history

    def _display_qdot_history(self, result) -> np.ndarray:
        """Return the generalized velocities used for display."""

        qdot_history = np.asarray(result.qdot, dtype=float).copy()
        if self._animation_reference() == ANIMATION_REFERENCE_OPTIONS[1]:
            qdot_history[:, :6] = 0.0
        return qdot_history

    def _animation_reference(self) -> str:
        """Return the currently selected animation reference frame."""

        if hasattr(self, "animation_reference_var"):
            return self.animation_reference_var.get()
        if hasattr(self, "animation_mode_var") and self.animation_mode_var.get() == ANIMATION_MODE_OPTIONS[1]:
            return ANIMATION_REFERENCE_OPTIONS[2]
        if hasattr(self, "root_initial_mode") and self.root_initial_mode.get() == ROOT_INITIAL_OPTIONS[0]:
            return ANIMATION_REFERENCE_OPTIONS[1]
        return ANIMATION_REFERENCE_OPTIONS[0]

    def _apply_animation_reference(self, reference: str) -> None:
        """Map the user-facing animation reference to the internal display settings."""

        if reference == ANIMATION_REFERENCE_OPTIONS[2]:
            self.root_initial_mode.set(ROOT_INITIAL_OPTIONS[1])
            self.animation_mode_var.set(ANIMATION_MODE_OPTIONS[1])
        elif reference == ANIMATION_REFERENCE_OPTIONS[1]:
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

        display_q = self._display_q_history(self._last_simulation)
        display_qdot = self._display_qdot_history(self._last_simulation)
        trajectories = marker_trajectories(self._last_model_path, display_q)
        frame_trajectories = segment_frame_trajectories(
            self._last_model_path,
            display_q,
            ALL_FRAME_SEGMENTS,
        )
        observables = system_observables(self._last_model_path, display_q, display_qdot)
        deviations = arm_deviation_from_frames(frame_trajectories, display_q[:, 3])
        btp_trajectories = arm_btp_reference_trajectories(trajectories, display_q[:, 3])
        self._visualization_data = {
            "result": self._last_simulation,
            "display_q": display_q,
            "display_qdot": display_qdot,
            "trajectories": trajectories,
            "btp_trajectories": btp_trajectories,
            "frames": frame_trajectories,
            "observables": observables,
            "deviations": deviations,
        }

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
        self._animation_axis.clear()
        self._animation_axis.set_xlabel("Mediolat.")
        self._animation_axis.set_ylabel("Ant.-post.")
        self._animation_axis.set_zlabel("Longitudinal")
        self._animation_axis.set_box_aspect((1.0, 1.0, 1.0))
        self._apply_camera_view()

        all_points = np.concatenate(list(trajectories.values()), axis=0)
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
        self._animation_axis.clear()
        self._animation_axis.set_xlabel("Axe somersault (m)")
        self._animation_axis.set_ylabel("Axe twist BTP (m)")
        self._animation_axis.set_zlabel("Hors BTP (m)")
        self._animation_axis.set_box_aspect((1.0, 1.0, 0.7))
        self._animation_axis.view_init(
            elev=BTP_CAMERA_ELEVATION_DEG,
            azim=BTP_CAMERA_AZIMUTH_DEG,
        )

        all_points = np.concatenate(list(projected.values()), axis=0)
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

        if self._animation_reference() == ANIMATION_REFERENCE_OPTIONS[1]:
            self._animation_axis.view_init(
                elev=ROOT_VIEW_CAMERA_ELEVATION_DEG,
                azim=ROOT_VIEW_CAMERA_AZIMUTH_DEG,
            )
            return
        self._animation_axis.view_init(
            elev=DEFAULT_CAMERA_ELEVATION_DEG,
            azim=DEFAULT_CAMERA_AZIMUTH_DEG,
        )

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

    def _on_close(self) -> None:
        """Close the GUI after cancelling any pending Tk callbacks."""

        self._is_closing = True
        self._stop_animation_loop()
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
        frame_trajectories = self._visualization_data["frames"]
        observables = self._visualization_data["observables"]

        for artist, (start_name, end_name) in zip(self._line_artists, SKELETON_CONNECTIONS):
            segment = np.vstack(
                (trajectories[start_name][frame_index], trajectories[end_name][frame_index])
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
        deviations = self._visualization_data["deviations"]

        for artist, (start_name, end_name) in zip(self._line_artists, SKELETON_CONNECTIONS):
            segment = np.vstack((projected[start_name][frame_index], projected[end_name][frame_index]))
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
            if mode in seen_modes:
                continue
            seen_modes.add(mode)
            bundle = self._load_cached_scan_bundle_for_mode(mode)
            if bundle is not None:
                datasets.append(bundle)
        return datasets

    def _refresh_scan_plot(self) -> None:
        """Draw the embedded scan figure showing the final twist count as a function of `t1`."""

        self._plot_axis.clear()
        datasets = self._scan_plot_datasets()
        self._plot_axis.set_xlabel("Debut bras droit t1 (s)")
        self._plot_axis.set_ylabel("Vrilles finales (tours)")
        self._plot_axis.set_title("Vrilles finales en fonction de t1")
        self._plot_axis.grid(True, alpha=0.3)
        if not datasets:
            self._plot_axis.text(
                0.5,
                0.5,
                "Aucun scan disponible.\nLancez Optimize.",
                ha="center",
                va="center",
                transform=self._plot_axis.transAxes,
            )
            self._plot_canvas.draw_idle()
            return

        for dataset in datasets:
            mode = str(dataset["mode"])
            style = SCAN_PLOT_STYLE_BY_MODE[mode]
            start_times = np.asarray(dataset["start_times"], dtype=float)
            final_twist_turns = np.asarray(dataset["final_twist_turns"], dtype=float)
            success_mask = np.asarray(dataset["success_mask"], dtype=bool)
            best_start_time = float(dataset["best_start_time"])
            best_index = int(np.argmin(np.abs(start_times - best_start_time)))
            self._plot_axis.plot(
                start_times,
                final_twist_turns,
                color=style["color"],
                linewidth=1.8,
                marker=style["marker"],
                markersize=4.0,
                label=style["label"],
            )
            if np.any(~success_mask):
                self._plot_axis.scatter(
                    start_times[~success_mask],
                    final_twist_turns[~success_mask],
                    color=style["color"],
                    marker="x",
                    s=36,
                )
            self._plot_axis.scatter(
                [start_times[best_index]],
                [final_twist_turns[best_index]],
                color=style["color"],
                edgecolors="black",
                linewidths=0.8,
                s=64,
                zorder=3,
            )
        self._plot_axis.legend(loc="best")
        self._plot_canvas.draw_idle()

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
        elif y_choice == "Deviation bras gauche":
            y_data = np.rad2deg(deviations["left"])
            y_label = "Deviation bras gauche / BTP (deg)"
        elif y_choice == "Deviation bras droit":
            y_data = np.rad2deg(deviations["right"])
            y_label = "Deviation bras droit / BTP (deg)"
        else:
            y_data = np.rad2deg(deviations["right"])
            y_label = "Deviation bras droit / BTP (deg)"
        if y_choice not in {"Cinematique bras", "Vitesses bras"}:
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
        if self.plot_y_var.get() == "Vrilles selon t1":
            self._refresh_scan_plot()
            return

        x_data, y_data, x_label, y_label, title, curve_labels = self._plot_data()
        self._plot_axis.clear()
        if np.asarray(y_data).ndim == 2:
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
        else:
            self._run_simulation_with_motion(prescribed_motion)
        if status_suffix is not None:
            self.result_var.set(f"{self.result_var.get()} | {status_suffix}")

    def _show_first_arm_jerk_comparison(self) -> None:
        """Open an external comparison window for the first-arm plane motion."""

        show_first_arm_piecewise_constant_comparison(
            _variables_from_gui(self._current_values()),
            total_time=self._standard_optimization_configuration().final_time,
            jerk_step=0.02,
            sample_step=0.005,
        )

    def _report_callback_exception(self, exception_class, exception, trace) -> None:
        """Report one Tkinter callback exception without leaving the GUI silent."""

        traceback.print_exception(exception_class, exception, trace)
        self.result_var.set(f"Erreur GUI: {exception}")

    def _optimize_strategy(self) -> None:
        """Optimize the current strategy with IPOPT, then update the GUI and rerun the simulation."""

        try:
            current_values = self._current_values()
            initial_guess = _variables_from_gui(current_values)
            mode = self.optimization_mode_var.get()
            self._auto_runner.cancel()
            self.result_var.set("Optimisation en cours... voir les iterations IPOPT dans le terminal.")
            self.root.update_idletasks()

            use_cache = not self._should_ignore_optimization_cache()

            if mode == "Optimize DMS" and use_cache:
                cached_dms_solution = self._load_cached_dms_solution()
                if cached_dms_solution is not None:
                    cached_values, cached_motion, cached_final_twist_turns, cached_solver_status, cached_scan_data = (
                        cached_dms_solution
                    )
                    if cached_scan_data is not None:
                        self._schedule_scan_figure(
                            start_times=cached_scan_data["start_times"],
                            final_twist_turns=cached_scan_data["final_twist_turns"],
                            objective_values=cached_scan_data["objective_values"],
                            success_mask=cached_scan_data["success_mask"],
                            best_start_time=cached_values["right_arm_start"],
                        )
                    self._apply_optimized_values(
                        cached_values,
                        prescribed_motion=cached_motion,
                        status_suffix=(
                            f"optimum DMS charge depuis le cache: "
                            f"{cached_final_twist_turns:.2f} tours ({cached_solver_status})"
                        ),
                    )
                    return
            elif mode != "Optimize DMS" and use_cache:
                cached_values = self._load_cached_optimized_values()
                if cached_values is not None:
                    cached_scan_data = self._load_cached_optimized_scan_data()
                    if cached_scan_data is not None:
                        self._schedule_scan_figure(
                            start_times=cached_scan_data["start_times"],
                            final_twist_turns=cached_scan_data["final_twist_turns"],
                            objective_values=cached_scan_data["objective_values"],
                            best_start_time=cached_values["right_arm_start"],
                        )
                        self._show_embedded_scan_plot()
                    self._apply_optimized_values(
                        cached_values,
                        status_suffix="optimum charge depuis le cache",
                    )
                    return

            if mode == "Optimize DMS":
                optimizer = DirectMultipleShootingOptimizer.from_builder(
                    self._model_path(),
                    configuration=self._standard_optimization_configuration(),
                    shooting_step=DMS_SHOOTING_STEP,
                    jerk_regularization=DMS_JERK_REGULARIZATION,
                )
                candidate_start_times = np.asarray(optimizer.candidate_start_times(), dtype=float)
                cached_progress = None if not use_cache else self._load_cached_dms_progress()
                start_index = 0
                scan_start_times: list[float] = []
                scan_final_twist_turns: list[float] = []
                scan_objective_values: list[float] = []
                scan_success_mask: list[bool] = []
                previous_result = None
                best_result = None

                if cached_progress is not None:
                    scan_start_times = list(cached_progress["start_times"])
                    scan_final_twist_turns = list(cached_progress["final_twist_turns"])
                    scan_objective_values = list(cached_progress["objective_values"])
                    scan_success_mask = list(cached_progress["success_mask"])
                    start_index = len(scan_start_times)
                    best_motion = self._rebuild_cached_dms_motion(
                        cached_progress["optimized_values"],
                        left_plane_jerk=cached_progress["left_plane_jerk"],
                        right_plane_jerk=cached_progress["right_plane_jerk"],
                    )
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
                        prescribed_motion=best_motion,
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

                for index in range(start_index, len(candidate_start_times)):
                    current_start_time = float(candidate_start_times[index])
                    self.result_var.set(
                        f"Optimisation DMS en cours... t1={current_start_time:.2f} s "
                        f"({index + 1}/{len(candidate_start_times)})"
                    )
                    self.root.update_idletasks()
                    current_result = optimizer.solve_fixed_start(
                        initial_guess,
                        right_arm_start=current_start_time,
                        previous_result=previous_result,
                        max_iter=50,
                        print_level=5,
                        print_time=True,
                        show_jerk_diagnostics=False,
                    )
                    scan_start_times.append(current_start_time)
                    scan_final_twist_turns.append(current_result.final_twist_turns)
                    scan_objective_values.append(current_result.objective)
                    scan_success_mask.append(bool(current_result.success))
                    previous_result = current_result

                    if best_result is None:
                        best_result = current_result
                    elif current_result.success and not best_result.success:
                        best_result = current_result
                    elif current_result.success == best_result.success and current_result.objective < best_result.objective:
                        best_result = current_result

                    optimized_values_checkpoint = _gui_values_from_variables(best_result.variables)
                    self._store_cached_dms_progress(
                        optimized_values_checkpoint,
                        left_plane_jerk=np.asarray(best_result.left_plane_jerk, dtype=float),
                        right_plane_jerk=np.asarray(best_result.right_plane_jerk, dtype=float),
                        scan_start_times=np.asarray(scan_start_times, dtype=float),
                        scan_final_twist_turns=np.asarray(scan_final_twist_turns, dtype=float),
                        scan_objective_values=np.asarray(scan_objective_values, dtype=float),
                        scan_success_mask=np.asarray(scan_success_mask, dtype=bool),
                        last_completed_index=index,
                        last_warm_start_primal=(
                            None
                            if getattr(current_result, "warm_start_primal", None) is None
                            else np.asarray(current_result.warm_start_primal, dtype=float)
                        ),
                        last_warm_start_lam_x=(
                            None
                            if getattr(current_result, "warm_start_lam_x", None) is None
                            else np.asarray(current_result.warm_start_lam_x, dtype=float)
                        ),
                        last_warm_start_lam_g=(
                            None
                            if getattr(current_result, "warm_start_lam_g", None) is None
                            else np.asarray(current_result.warm_start_lam_g, dtype=float)
                        ),
                        final_twist_turns=float(best_result.final_twist_turns),
                        solver_status=str(best_result.solver_status),
                    )

                if best_result is None:
                    raise RuntimeError("No DMS candidate was evaluated.")
                result = best_result
                optimized_values = _gui_values_from_variables(result.variables)
                scan_data = self._scan_data_from_lists(
                    start_times=scan_start_times,
                    final_twist_turns=scan_final_twist_turns,
                    objective_values=scan_objective_values,
                    success_mask=scan_success_mask,
                )
                self._schedule_scan_figure(
                    start_times=scan_data["start_times"],
                    final_twist_turns=scan_data["final_twist_turns"],
                    objective_values=scan_data["objective_values"],
                    success_mask=scan_data["success_mask"],
                    best_start_time=result.variables.right_arm_start,
                )
                self._schedule_dms_jerk_diagnostic_figure(optimizer=optimizer, result=result)
                self._store_cached_dms_solution(
                    optimized_values,
                    left_plane_jerk=result.left_plane_jerk,
                    right_plane_jerk=result.right_plane_jerk,
                    scan_start_times=scan_data["start_times"],
                    scan_final_twist_turns=scan_data["final_twist_turns"],
                    scan_objective_values=scan_data["objective_values"],
                    scan_success_mask=scan_data["success_mask"],
                    final_twist_turns=result.final_twist_turns,
                    solver_status=result.solver_status,
                )
                self._apply_optimized_values(
                    optimized_values,
                    prescribed_motion=result.prescribed_motion,
                    status_suffix=f"optimum DMS: {result.final_twist_turns:.2f} tours ({result.solver_status})",
                )
                return

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
                result = optimizer.optimize_right_arm_start_only(
                    initial_guess.right_arm_start,
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

            optimized_values = _gui_values_from_variables(result.variables)
            self._schedule_scan_figure(
                start_times=sweep.start_times,
                final_twist_turns=sweep.final_twist_turns,
                objective_values=sweep.objective_values,
                best_start_time=result.variables.right_arm_start,
            )
            self._show_embedded_scan_plot()
            self._store_cached_optimized_values(
                optimized_values,
                final_twist_turns=result.final_twist_turns,
                solver_status=result.solver_status,
                scan_start_times=sweep.start_times,
                scan_final_twist_turns=sweep.final_twist_turns,
                scan_objective_values=sweep.objective_values,
            )
            self._apply_optimized_values(
                optimized_values,
                status_suffix=(
                    f"optimum balayage 1D: {result.final_twist_turns:.2f} tours "
                    f"({result.solver_status})"
                ),
            )
        except Exception as error:
            traceback.print_exc()
            self.result_var.set(f"Erreur optimisation: {error}")


def launch_gui() -> None:
    """Launch the Tkinter GUI."""

    root = tk.Tk()
    BestTiltingPlaneApp(root)
    root.mainloop()
