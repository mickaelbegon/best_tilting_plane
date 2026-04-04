"""A first interactive GUI for the predictive twisting simulation."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

import numpy as np

from best_tilting_plane.modeling import (
    ARM_ELEVATION_SEQUENCE,
    ARM_PLANE_SEQUENCE,
    ARM_SEGMENTS_FOR_VISUALIZATION,
    GLOBAL_AXIS_LABELS,
    ROOT_ROTATION_SEQUENCE,
    ReducedAerialBiomod,
)
from best_tilting_plane.optimization import TwistStrategyOptimizer
from best_tilting_plane.gui.debounce import DebouncedRunner
from best_tilting_plane.simulation import (
    PredictiveAerialTwistSimulator,
    SimulationConfiguration,
    TwistOptimizationVariables,
)
from best_tilting_plane.visualization import (
    SKELETON_CONNECTIONS,
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
    SliderDefinition("left_plane_initial", "Plan init. gauche (deg)", -180.0, 180.0, 0.0),
    SliderDefinition("left_plane_final", "Plan fin. gauche (deg)", -180.0, 180.0, 0.0),
    SliderDefinition("right_plane_initial", "Plan init. droit (deg)", -180.0, 180.0, 0.0),
    SliderDefinition("right_plane_final", "Plan fin. droit (deg)", -180.0, 180.0, 0.0),
)


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
    """Simple Tkinter GUI that launches simulations and visualization windows."""

    def __init__(self, root: tk.Tk) -> None:
        """Create the main window and its controls."""

        self.root = root
        self.root.title("Best Tilting Plane")

        self._entries: dict[str, tk.StringVar] = {}
        self._scales: dict[str, tk.Scale] = {}
        self._last_simulation = None
        self._last_model_path: Path | None = None
        self._result_figure = None
        self._animation_figure = None
        self._animation = None
        self._auto_simulation_suspended = False
        self._auto_runner = DebouncedRunner(self.root, self._run_simulation, delay_ms=250)

        frame = ttk.Frame(root, padding=12)
        frame.grid(sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        for row, definition in enumerate(SLIDER_DEFINITIONS):
            ttk.Label(frame, text=definition.label).grid(
                row=row, column=0, sticky="w", padx=(0, 8), pady=4
            )
            scale = tk.Scale(
                frame,
                orient=tk.HORIZONTAL,
                from_=definition.minimum,
                to=definition.maximum,
                resolution=0.01,
                length=320,
            )
            scale.set(definition.default)
            scale.grid(row=row, column=1, sticky="ew", pady=4)
            entry_var = tk.StringVar(value=f"{definition.default:.2f}")
            entry = ttk.Entry(frame, textvariable=entry_var, width=10)
            entry.grid(row=row, column=2, sticky="e", pady=4)
            frame.columnconfigure(1, weight=1)

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
        ttk.Checkbutton(frame, text="Afficher le best tilting plane", variable=self.show_btp).grid(
            row=len(SLIDER_DEFINITIONS), column=0, columnspan=3, sticky="w", pady=(8, 6)
        )

        ttk.Button(frame, text="Simulate", command=self._run_simulation).grid(
            row=len(SLIDER_DEFINITIONS) + 1, column=0, sticky="w", pady=(6, 0)
        )
        ttk.Button(frame, text="Optimize", command=self._optimize_strategy).grid(
            row=len(SLIDER_DEFINITIONS) + 1, column=1, sticky="w", pady=(6, 0)
        )

        self.result_var = tk.StringVar(value="Aucune simulation lancée.")
        ttk.Label(frame, textvariable=self.result_var).grid(
            row=len(SLIDER_DEFINITIONS) + 1, column=2, sticky="w", pady=(6, 0)
        )
        self.sequence_var = tk.StringVar(
            value=(
                f"BioMod root: translations xyz, rotations {ROOT_ROTATION_SEQUENCE} = "
                f"(somersault, tilt, twist) | Bras: plan {ARM_PLANE_SEQUENCE[0]}, elevation {ARM_ELEVATION_SEQUENCE[0]}"
            )
        )
        ttk.Label(frame, textvariable=self.sequence_var, wraplength=720).grid(
            row=len(SLIDER_DEFINITIONS) + 2, column=0, columnspan=3, sticky="w", pady=(8, 0)
        )

        self._run_simulation()

    def _on_slider_change(self, name: str, variable: tk.StringVar, value: str) -> None:
        """Update the entry field and trigger a debounced simulation."""

        self._sync_entry(name, variable, value)
        if not self._auto_simulation_suspended:
            self.result_var.set("Simulation automatique...")
            self._auto_runner.schedule()

    def _sync_entry(self, _name: str, variable: tk.StringVar, value: str) -> None:
        """Update the entry field when the slider moves."""

        variable.set(f"{float(value):.2f}")

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

    def _run_simulation(self) -> None:
        """Simulate the current strategy and open the result windows."""

        import matplotlib.pyplot as plt
        from matplotlib import animation
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        variables = _variables_from_gui(self._current_values())
        model_path = self._model_path()

        simulator = PredictiveAerialTwistSimulator.from_builder(
            model_path,
            variables,
            configuration=SimulationConfiguration(integrator="auto"),
        )
        result = simulator.simulate()
        self._last_simulation = result
        self._last_model_path = Path(simulator.model_path)

        integrator_label = result.integrator_method.upper()
        if result.rk4_step is not None:
            integrator_label += f" (dt={result.rk4_step:.4f} s)"
        self.result_var.set(
            f"Nombre de vrilles final: {result.final_twist_turns:.2f} tours | "
            f"Integrateur: {integrator_label}"
        )

        trajectories = marker_trajectories(simulator.model_path, result.q)
        frame_trajectories = segment_frame_trajectories(
            simulator.model_path,
            result.q,
            ARM_SEGMENTS_FOR_VISUALIZATION,
        )
        observables = system_observables(simulator.model_path, result.q, result.qdot)

        if self._result_figure is not None:
            plt.close(self._result_figure)
        if self._animation_figure is not None:
            plt.close(self._animation_figure)

        self._result_figure, (result_axis, momentum_axis) = plt.subplots(
            2, 1, figsize=(8, 7), sharex=True
        )
        result_axis.plot(
            result.time, result.q[:, 5] / (2.0 * np.pi), color="tab:blue", linewidth=2.0
        )
        result_axis.set_ylabel("Vrilles (tours)")
        result_axis.set_title(
            f"Vrilles finales: {result.final_twist_turns:.2f} tours | {integrator_label}"
        )
        result_axis.grid(True, alpha=0.3)
        momentum_axis.plot(
            result.time,
            observables["angular_momentum"][:, 0],
            color="tab:red",
            linewidth=1.5,
            label="H_x",
        )
        momentum_axis.plot(
            result.time,
            observables["angular_momentum"][:, 1],
            color="tab:green",
            linewidth=1.5,
            label="H_y",
        )
        momentum_axis.plot(
            result.time,
            observables["angular_momentum"][:, 2],
            color="tab:blue",
            linewidth=1.5,
            label="H_z",
        )
        momentum_axis.set_xlabel("Temps (s)")
        momentum_axis.set_ylabel("H au CoM")
        momentum_axis.set_title("Moment cinetique du systeme au centre de masse")
        momentum_axis.grid(True, alpha=0.3)
        momentum_axis.legend(loc="upper right")

        self._animation_figure = plt.figure(figsize=(8, 7))
        axis = self._animation_figure.add_subplot(111, projection="3d")
        axis.set_title("Animation 3D")
        axis.set_xlabel("Mediolat.")
        axis.set_ylabel("Ant.-post.")
        axis.set_zlabel("Longitudinal")

        all_points = np.concatenate(list(trajectories.values()), axis=0)
        span = np.max(all_points, axis=0) - np.min(all_points, axis=0)
        center = np.mean(all_points, axis=0)
        radius = 0.6 * np.max(span)
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)

        line_artists = [
            axis.plot([], [], [], color="black", linewidth=2.0)[0] for _ in SKELETON_CONNECTIONS
        ]
        frame_artists = {
            segment_name: tuple(
                axis.plot([], [], [], color=color, linewidth=2.0)[0]
                for color in ("tab:red", "tab:green", "tab:blue")
            )
            for segment_name in ARM_SEGMENTS_FOR_VISUALIZATION
        }
        angular_momentum_artist = axis.plot([], [], [], color="tab:orange", linewidth=3.0)[0]
        plane_artist = None

        if self.show_btp.get():
            plane_artist = Poly3DCollection(
                [np.zeros((4, 3))], alpha=0.18, facecolor="tab:orange", edgecolor="none"
            )
            axis.add_collection3d(plane_artist)

        def update(frame_index: int):
            for artist, (start_name, end_name) in zip(line_artists, SKELETON_CONNECTIONS):
                segment = np.vstack(
                    (trajectories[start_name][frame_index], trajectories[end_name][frame_index])
                )
                artist.set_data(segment[:, 0], segment[:, 1])
                artist.set_3d_properties(segment[:, 2])

            for segment_name, artists in frame_artists.items():
                origin = frame_trajectories[segment_name]["origin"][frame_index]
                axes_matrix = frame_trajectories[segment_name]["axes"][frame_index]
                for axis_index, artist in enumerate(artists):
                    endpoint = origin + 0.15 * axes_matrix[:, axis_index]
                    points = np.vstack((origin, endpoint))
                    artist.set_data(points[:, 0], points[:, 1])
                    artist.set_3d_properties(points[:, 2])

            com = observables["center_of_mass"][frame_index]
            angular_momentum = observables["angular_momentum"][frame_index]
            scale = 0.08
            momentum_points = np.vstack((com, com + scale * angular_momentum))
            angular_momentum_artist.set_data(momentum_points[:, 0], momentum_points[:, 1])
            angular_momentum_artist.set_3d_properties(momentum_points[:, 2])

            if plane_artist is not None:
                corners = best_tilting_plane_corners(
                    trajectories["pelvis_origin"][frame_index],
                    somersault_angle=result.q[frame_index, 3],
                )
                plane_artist.set_verts([corners])

            axis.set_title(
                "Animation 3D | "
                f"t = {result.time[frame_index]:.2f} s | "
                f"vrilles = {result.q[frame_index, 5] / (2*np.pi):.2f} | "
                f"H(CoM) = {np.array2string(angular_momentum, precision=2)} | "
                f"axes: {GLOBAL_AXIS_LABELS}"
            )
            flat_frame_artists = tuple(
                artist for artists in frame_artists.values() for artist in artists
            )
            return (
                tuple(line_artists)
                + flat_frame_artists
                + (angular_momentum_artist,)
                + (() if plane_artist is None else (plane_artist,))
            )

        self._animation = animation.FuncAnimation(
            self._animation_figure,
            update,
            frames=result.time.size,
            interval=35,
            blit=False,
            repeat=True,
        )
        plt.show(block=False)
        plt.pause(0.001)

    def _optimize_strategy(self) -> None:
        """Optimize the current strategy with IPOPT, then update the GUI and rerun the simulation."""

        current_values = self._current_values()
        initial_guess = _variables_from_gui(current_values)
        self._auto_runner.cancel()
        self.result_var.set("Optimisation en cours... voir les iterations IPOPT dans le terminal.")
        self.root.update_idletasks()

        optimizer = TwistStrategyOptimizer.from_builder(self._model_path())
        result = optimizer.optimize(initial_guess, max_iter=25, print_level=5, print_time=True)

        optimized_values = {
            "right_arm_start": result.variables.right_arm_start,
            "left_plane_initial": np.rad2deg(result.variables.left_plane_initial),
            "left_plane_final": np.rad2deg(result.variables.left_plane_final),
            "right_plane_initial": np.rad2deg(result.variables.right_plane_initial),
            "right_plane_final": np.rad2deg(result.variables.right_plane_final),
        }
        self._set_values(optimized_values)
        self.result_var.set(
            f"Optimum IPOPT: {result.final_twist_turns:.2f} tours ({result.solver_status})"
        )
        self._run_simulation()


def launch_gui() -> None:
    """Launch the Tkinter GUI."""

    root = tk.Tk()
    BestTiltingPlaneApp(root)
    root.mainloop()
