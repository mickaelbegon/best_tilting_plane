"""Interactive GUI for the predictive twisting simulation."""

from __future__ import annotations

import json
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

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
from best_tilting_plane.optimization import TwistStrategyOptimizer
from best_tilting_plane.simulation import (
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


SLIDER_DEFINITIONS = (
    SliderDefinition("right_arm_start", "Start bras droit (s)", 0.0, 0.7, 0.10),
    SliderDefinition(
        "left_plane_initial",
        "Plan init. gauche (deg)",
        LEFT_ARM_PLANE_BOUNDS_DEG[0],
        LEFT_ARM_PLANE_BOUNDS_DEG[1],
        0.0,
    ),
    SliderDefinition(
        "left_plane_final",
        "Plan fin. gauche (deg)",
        LEFT_ARM_PLANE_BOUNDS_DEG[0],
        LEFT_ARM_PLANE_BOUNDS_DEG[1],
        0.0,
    ),
    SliderDefinition(
        "right_plane_initial",
        "Plan init. droit (deg)",
        RIGHT_ARM_PLANE_BOUNDS_DEG[0],
        RIGHT_ARM_PLANE_BOUNDS_DEG[1],
        0.0,
    ),
    SliderDefinition(
        "right_plane_final",
        "Plan fin. droit (deg)",
        RIGHT_ARM_PLANE_BOUNDS_DEG[0],
        RIGHT_ARM_PLANE_BOUNDS_DEG[1],
        0.0,
    ),
)
PLOT_X_OPTIONS = ("Temps", "Somersault")
PLOT_MODE_OPTIONS = ("Courbe", "Bras hors BTP (dessus)")
ANIMATION_MODE_OPTIONS = ("Animation 3D", "Bras / BTP")
OPTIMIZATION_MODE_OPTIONS = ("Optimize 2D", "Optimize 5D")
PLOT_Y_OPTIONS = (
    "Somersault",
    "Tilt",
    "Twist",
    "Deviation bras gauche",
    "Deviation bras droit",
)
ROOT_INITIAL_OPTIONS = ("Avec q racine(0)=0", "Sans q racine(0)=0")
TOP_VIEW_LEFT_CHAIN = ("shoulder_left", "elbow_left", "wrist_left", "hand_left")
TOP_VIEW_RIGHT_CHAIN = ("shoulder_right", "elbow_right", "wrist_right", "hand_right")
DEFAULT_CAMERA_ELEVATION_DEG = 20.0
DEFAULT_CAMERA_AZIMUTH_DEG = -60.0
BTP_CAMERA_ELEVATION_DEG = 20.0
BTP_CAMERA_AZIMUTH_DEG = -55.0
TOP_VIEW_CAMERA_ELEVATION_DEG = 90.0
TOP_VIEW_CAMERA_AZIMUTH_DEG = -90.0
ALL_FRAME_SEGMENTS = tuple(
    dict.fromkeys(ARM_SEGMENTS_FOR_VISUALIZATION + ARM_SEGMENTS_FOR_DEVIATION)
)
ANIMATION_INTERVAL_MS = 35
STANDARD_RK4_STEP = 0.005
OPTIMIZATION_CACHE_VERSION = 1


def _variables_from_gui(values: dict[str, float]) -> TwistOptimizationVariables:
    """Convert GUI values into the optimization-variable structure."""

    return TwistOptimizationVariables(
        right_arm_start=values["right_arm_start"],
        left_plane_initial=np.deg2rad(values["left_plane_initial"]),
        left_plane_final=np.deg2rad(values["left_plane_final"]),
        right_plane_initial=np.deg2rad(values["right_plane_initial"]),
        right_plane_final=np.deg2rad(values["right_plane_final"]),
    )


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
        self._auto_runner = DebouncedRunner(self.root, self._run_simulation, delay_ms=250)

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
                resolution=0.01,
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

        ttk.Label(controls, text="Mode conditions initiales").grid(
            row=len(SLIDER_DEFINITIONS) + 1, column=0, sticky="w", pady=(8, 4)
        )
        self.root_initial_mode = tk.StringVar(value=ROOT_INITIAL_OPTIONS[0])
        root_mode_box = ttk.Combobox(
            controls,
            textvariable=self.root_initial_mode,
            values=ROOT_INITIAL_OPTIONS,
            state="readonly",
            width=24,
        )
        root_mode_box.grid(
            row=len(SLIDER_DEFINITIONS) + 1, column=1, columnspan=2, sticky="ew", pady=(8, 4)
        )
        root_mode_box.bind("<<ComboboxSelected>>", lambda _event: self._on_root_mode_change())

        ttk.Label(controls, text="Mode animation").grid(
            row=len(SLIDER_DEFINITIONS) + 2, column=0, sticky="w", pady=4
        )
        self.animation_mode_var = tk.StringVar(value=ANIMATION_MODE_OPTIONS[0])
        animation_mode_box = ttk.Combobox(
            controls,
            textvariable=self.animation_mode_var,
            values=ANIMATION_MODE_OPTIONS,
            state="readonly",
            width=24,
        )
        animation_mode_box.grid(
            row=len(SLIDER_DEFINITIONS) + 2, column=1, columnspan=2, sticky="ew", pady=4
        )
        animation_mode_box.bind("<<ComboboxSelected>>", lambda _event: self._on_animation_mode_change())

        ttk.Label(controls, text="Mode figure").grid(
            row=len(SLIDER_DEFINITIONS) + 3, column=0, sticky="w", pady=4
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
            row=len(SLIDER_DEFINITIONS) + 3, column=1, columnspan=2, sticky="ew", pady=4
        )
        plot_mode_box.bind("<<ComboboxSelected>>", lambda _event: self._refresh_plot())

        ttk.Label(controls, text="Figure x").grid(
            row=len(SLIDER_DEFINITIONS) + 4, column=0, sticky="w", pady=4
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
            row=len(SLIDER_DEFINITIONS) + 4, column=1, columnspan=2, sticky="ew", pady=4
        )
        plot_x_box.bind("<<ComboboxSelected>>", lambda _event: self._refresh_plot())

        ttk.Label(controls, text="Figure y").grid(
            row=len(SLIDER_DEFINITIONS) + 5, column=0, sticky="w", pady=4
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
            row=len(SLIDER_DEFINITIONS) + 5, column=1, columnspan=2, sticky="ew", pady=4
        )
        plot_y_box.bind("<<ComboboxSelected>>", lambda _event: self._refresh_plot())

        ttk.Button(controls, text="Simulate", command=self._run_simulation).grid(
            row=len(SLIDER_DEFINITIONS) + 6, column=0, sticky="w", pady=(10, 0)
        )
        self.optimization_mode_var = tk.StringVar(value=OPTIMIZATION_MODE_OPTIONS[0])
        optimization_mode_box = ttk.Combobox(
            controls,
            textvariable=self.optimization_mode_var,
            values=OPTIMIZATION_MODE_OPTIONS,
            state="readonly",
            width=18,
        )
        optimization_mode_box.grid(
            row=len(SLIDER_DEFINITIONS) + 6, column=1, sticky="ew", pady=(10, 0), padx=(0, 8)
        )
        ttk.Button(controls, text="Optimize", command=self._optimize_strategy).grid(
            row=len(SLIDER_DEFINITIONS) + 6, column=2, sticky="w", pady=(10, 0)
        )

        self.result_var = tk.StringVar(value="Aucune simulation lancée.")
        ttk.Label(controls, textvariable=self.result_var, wraplength=360, justify="left").grid(
            row=len(SLIDER_DEFINITIONS) + 7, column=0, columnspan=3, sticky="w", pady=(10, 0)
        )
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
            row=len(SLIDER_DEFINITIONS) + 8, column=0, columnspan=3, sticky="w", pady=(8, 0)
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

        return {name: float(variable.get()) for name, variable in self._entries.items()}

    def _set_values(self, values: dict[str, float]) -> None:
        """Write a new set of values back to the sliders and entry boxes."""

        self._auto_simulation_suspended = True
        for name, value in values.items():
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

        return self.optimization_mode_var.get().lower().replace(" ", "_")

    def _optimization_cache_signature(self) -> dict[str, float | int | str]:
        """Describe the numerical setup that must match for a cached optimum to be reused."""

        configuration = self._standard_optimization_configuration()
        return {
            "version": OPTIMIZATION_CACHE_VERSION,
            "mode": self._optimization_cache_key(),
            "final_time": float(configuration.final_time),
            "steps": int(configuration.steps),
            "somersault_rate": float(configuration.somersault_rate),
            "integrator": configuration.integrator,
            "rk4_step": float(configuration.rk4_step) if configuration.rk4_step is not None else None,
        }

    def _read_optimization_cache_file(self) -> dict[str, object]:
        """Return the JSON cache content, or an empty structure if it does not exist."""

        cache_path = self._optimization_cache_path()
        if not cache_path.exists():
            return {"records": {}}
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"records": {}}
        if not isinstance(data, dict):
            return {"records": {}}
        records = data.get("records")
        if not isinstance(records, dict):
            return {"records": {}}
        return {"records": records}

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
        expected_names = {definition.name for definition in SLIDER_DEFINITIONS}
        if set(values) != expected_names:
            return None
        try:
            return {name: float(values[name]) for name in expected_names}
        except (TypeError, ValueError):
            return None

    def _store_cached_optimized_values(
        self,
        optimized_values: dict[str, float],
        *,
        final_twist_turns: float,
        solver_status: str,
    ) -> None:
        """Persist optimized GUI values for reuse in later GUI sessions."""

        cache_path = self._optimization_cache_path()
        cache = self._read_optimization_cache_file()
        cache["records"][self._optimization_cache_key()] = {
            "signature": self._optimization_cache_signature(),
            "values": {name: float(value) for name, value in optimized_values.items()},
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
        result = simulator.simulate()
        self._last_simulation = result
        self._last_model_path = Path(simulator.model_path)
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
        if self.root_initial_mode.get() == ROOT_INITIAL_OPTIONS[0]:
            q_history[:, :6] = 0.0
        return q_history

    def _display_qdot_history(self, result) -> np.ndarray:
        """Return the generalized velocities used for display."""

        qdot_history = np.asarray(result.qdot, dtype=float).copy()
        if self.root_initial_mode.get() == ROOT_INITIAL_OPTIONS[0]:
            qdot_history[:, :6] = 0.0
        return qdot_history

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

    def _on_root_mode_change(self) -> None:
        """Refresh the displays when toggling the `q(root)=0` mode."""

        if self._last_simulation is None:
            return
        self._refresh_visualization_data()
        self._prepare_animation_scene()
        self._refresh_plot()

    def _on_animation_mode_change(self) -> None:
        """Refresh the animation panel when toggling the animation mode."""

        if self._visualization_data is None:
            return
        self._prepare_animation_scene()
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
        """Prepare the arm animation expressed in the best-tilting-plane frame."""

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
        self._line_artists = ()
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
        """Apply the default or top-down camera depending on the root display mode."""

        if self.root_initial_mode.get() == ROOT_INITIAL_OPTIONS[0]:
            self._animation_axis.view_init(
                elev=TOP_VIEW_CAMERA_ELEVATION_DEG,
                azim=TOP_VIEW_CAMERA_AZIMUTH_DEG,
            )
            return
        self._animation_axis.view_init(
            elev=DEFAULT_CAMERA_ELEVATION_DEG,
            azim=DEFAULT_CAMERA_AZIMUTH_DEG,
        )

    def _stop_animation_loop(self) -> None:
        """Cancel the currently scheduled animation callback, if any."""

        if self._animation_after_id is not None:
            self.root.after_cancel(self._animation_after_id)
            self._animation_after_id = None
        self._animation_playing = False
        self.play_pause_label.set("Play")

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

        if self._visualization_data is None:
            return

        result = self._visualization_data["result"]
        current_index = self._animation_frame_index
        self._draw_animation_frame(current_index)
        self._sync_time_slider_to_frame(current_index)
        if self._plot_requires_frame_sync():
            self._refresh_plot()
        self._animation_frame_index = (current_index + 1) % result.time.size
        self._animation_after_id = self.root.after(ANIMATION_INTERVAL_MS, self._animate_next_frame)

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
        """Draw one frame of the arm motion expressed in the best-tilting-plane frame."""

        result = self._visualization_data["result"]
        projected = self._visualization_data["btp_trajectories"]
        deviations = self._visualization_data["deviations"]

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
            "Animation bras / BTP | "
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

    def _plot_data(self) -> tuple[np.ndarray, np.ndarray, str, str, str]:
        """Return the currently selected x/y data and corresponding labels."""

        if self._visualization_data is None:
            raise RuntimeError("No simulation available for plotting.")

        result = self._visualization_data["result"]
        deviations = self._visualization_data["deviations"]

        if self.plot_x_var.get() == "Somersault":
            x_data = np.rad2deg(self._root_series(result, 0))
            x_label = "Somersault (deg)"
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
        elif y_choice == "Deviation bras gauche":
            y_data = np.rad2deg(deviations["left"])
            y_label = "Deviation bras gauche / BTP (deg)"
        elif y_choice == "Deviation bras droit":
            y_data = np.rad2deg(deviations["right"])
            y_label = "Deviation bras droit / BTP (deg)"
        else:
            y_data = np.rad2deg(deviations["right"])
            y_label = "Deviation bras droit / BTP (deg)"

        title = f"{y_choice} en fonction de {self.plot_x_var.get().lower()}"
        return x_data, y_data, x_label, y_label, title

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
        if self.root_initial_mode.get() == ROOT_INITIAL_OPTIONS[0]:
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

        x_data, y_data, x_label, y_label, title = self._plot_data()
        self._plot_axis.clear()
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
        self, optimized_values: dict[str, float], *, status_suffix: str | None = None
    ) -> None:
        """Push optimized parameters to the GUI, rerun the simulation, and restart the animation."""

        self._set_values(optimized_values)
        self.root.update_idletasks()
        self._run_simulation()
        if status_suffix is not None:
            self.result_var.set(f"{self.result_var.get()} | {status_suffix}")

    def _optimize_strategy(self) -> None:
        """Optimize the current strategy with IPOPT, then update the GUI and rerun the simulation."""

        current_values = self._current_values()
        initial_guess = _variables_from_gui(current_values)
        self._auto_runner.cancel()
        self.result_var.set("Optimisation en cours... voir les iterations IPOPT dans le terminal.")
        self.root.update_idletasks()

        cached_values = self._load_cached_optimized_values()
        if cached_values is not None:
            self._apply_optimized_values(
                cached_values,
                status_suffix="optimum charge depuis le cache",
            )
            return

        optimizer = TwistStrategyOptimizer.from_builder(
            self._model_path(),
            configuration=self._standard_optimization_configuration(),
        )
        if self.optimization_mode_var.get() == "Optimize 5D":
            result = optimizer.optimize(initial_guess, max_iter=25, print_level=5, print_time=True)
        else:
            result = optimizer.optimize_right_arm_start_only(
                initial_guess.right_arm_start,
                max_iter=25,
                print_level=5,
                print_time=True,
            )

        optimized_values = {
            "right_arm_start": result.variables.right_arm_start,
            "left_plane_initial": np.rad2deg(result.variables.left_plane_initial),
            "left_plane_final": np.rad2deg(result.variables.left_plane_final),
            "right_plane_initial": np.rad2deg(result.variables.right_plane_initial),
            "right_plane_final": np.rad2deg(result.variables.right_plane_final),
        }
        self._store_cached_optimized_values(
            optimized_values,
            final_twist_turns=result.final_twist_turns,
            solver_status=result.solver_status,
        )
        self._apply_optimized_values(
            optimized_values,
            status_suffix=f"optimum IPOPT: {result.final_twist_turns:.2f} tours ({result.solver_status})",
        )


def launch_gui() -> None:
    """Launch the Tkinter GUI."""

    root = tk.Tk()
    BestTiltingPlaneApp(root)
    root.mainloop()
