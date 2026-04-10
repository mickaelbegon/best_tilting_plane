"""Generation of a reduced zero-gravity `bioMod` for the aerial twisting simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from best_tilting_plane.anthropometry import (
    DeLevaSex,
    SegmentInertialParameters,
    de_leva_segment_table,
)
from best_tilting_plane.modeling.segments import BodyDimensions


def _diag_inertia(
    mass: float, length: float, radii: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Compute the diagonal inertia tensor from normalized radii of gyration."""

    return tuple(mass * (radius * length) ** 2 for radius in radii)


def _matrix_block(translation: tuple[float, float, float]) -> str:
    """Return a 4x4 homogeneous transform block with identity rotation."""

    tx, ty, tz = translation
    return (
        "\tRTinMatrix\t1\n"
        "\tRT\n"
        f"\t\t1.000000\t0.000000\t0.000000\t{tx:.6f}\n"
        f"\t\t0.000000\t1.000000\t0.000000\t{ty:.6f}\n"
        f"\t\t0.000000\t0.000000\t1.000000\t{tz:.6f}\n"
        "\t\t0.000000\t0.000000\t0.000000\t1.000000\n"
    )


def _segment_block(
    name: str,
    parent: str,
    translation: tuple[float, float, float],
    *,
    translations: str | None = None,
    rotations: str | None = None,
    mass: float = 0.0,
    center_of_mass: tuple[float, float, float] = (0.0, 0.0, 0.0),
    inertia: tuple[float, float, float] = (1e-6, 1e-6, 1e-6),
    ranges_q: tuple[tuple[float, float], ...] | None = None,
) -> str:
    """Serialize one `bioMod` segment block."""

    out = [f"segment\t{name}\n", f"\tparent\t{parent}\n", _matrix_block(translation)]
    if translations is not None:
        out.append(f"\ttranslations\t{translations}\n")
    if rotations is not None:
        out.append(f"\trotations\t{rotations}\n")
    if ranges_q:
        out.append("\trangesQ\n")
        out.extend(f"\t\t{low:.6f}\t{high:.6f}\n" for low, high in ranges_q)
    if mass > 0.0:
        out.append(f"\tmass\t{mass:.6f}\n")
        out.append(
            f"\tCenterOfMass\t{center_of_mass[0]:.6f}\t{center_of_mass[1]:.6f}\t{center_of_mass[2]:.6f}\n"
        )
        out.append("\tinertia\n")
        out.append(f"\t\t{inertia[0]:.6f}\t0.000000\t0.000000\n")
        out.append(f"\t\t0.000000\t{inertia[1]:.6f}\t0.000000\n")
        out.append(f"\t\t0.000000\t0.000000\t{inertia[2]:.6f}\n")
    out.append("endsegment\n\n")
    return "".join(out)


def _marker_block(name: str, parent: str, position: tuple[float, float, float]) -> str:
    """Serialize one marker block."""

    return (
        f"marker\t{name}\n"
        f"\tparent\t{parent}\n"
        f"\tposition\t{position[0]:.6f}\t{position[1]:.6f}\t{position[2]:.6f}\n"
        "\ttechnical\t1\n"
        "\tanatomical\t0\n"
        "endmarker\n\n"
    )


@dataclass(frozen=True)
class ReducedAerialBiomod:
    """Builder for the reduced 10-DoF whole-body `bioMod`."""

    mass: float = 75.0
    height: float = 1.75
    sex: DeLevaSex = DeLevaSex.MALE

    @property
    def dimensions(self) -> BodyDimensions:
        """Return the body dimensions associated with the current height."""

        return BodyDimensions.from_height(self.height)

    @property
    def q_size(self) -> int:
        """Return the total number of generalized coordinates."""

        return 10

    def to_biomod_string(self) -> str:
        """Serialize the reduced whole-body model as a `bioMod` string."""

        dims = self.dimensions
        table = de_leva_segment_table(self.sex)

        trunk_mass = self.mass * table["trunk"].mass_fraction
        trunk_com = (0.0, 0.0, table["trunk"].center_of_mass_fraction * dims.trunk_length)
        trunk_inertia = _diag_inertia(
            trunk_mass, dims.trunk_length, table["trunk"].radii_of_gyration
        )

        head_mass = self.mass * table["head"].mass_fraction
        head_com = (0.0, 0.0, table["head"].center_of_mass_fraction * dims.head_length)
        head_inertia = _diag_inertia(head_mass, dims.head_length, table["head"].radii_of_gyration)

        upper_arm_mass = self.mass * table["upper_arm"].mass_fraction
        upper_arm_com = (
            0.0,
            0.0,
            -table["upper_arm"].center_of_mass_fraction * dims.upper_arm_length,
        )
        upper_arm_inertia = _diag_inertia(
            upper_arm_mass, dims.upper_arm_length, table["upper_arm"].radii_of_gyration
        )

        forearm_mass = self.mass * table["forearm"].mass_fraction
        forearm_com = (0.0, 0.0, -table["forearm"].center_of_mass_fraction * dims.forearm_length)
        forearm_inertia = _diag_inertia(
            forearm_mass, dims.forearm_length, table["forearm"].radii_of_gyration
        )

        hand_mass = self.mass * table["hand"].mass_fraction
        hand_com = (0.0, 0.0, -table["hand"].center_of_mass_fraction * dims.hand_length)
        hand_inertia = _diag_inertia(hand_mass, dims.hand_length, table["hand"].radii_of_gyration)

        thigh_mass = self.mass * table["thigh"].mass_fraction
        thigh_com = (0.0, 0.0, -table["thigh"].center_of_mass_fraction * dims.thigh_length)
        thigh_inertia = _diag_inertia(
            thigh_mass, dims.thigh_length, table["thigh"].radii_of_gyration
        )

        shank_mass = self.mass * table["shank"].mass_fraction
        shank_com = (0.0, 0.0, -table["shank"].center_of_mass_fraction * dims.shank_length)
        shank_inertia = _diag_inertia(
            shank_mass, dims.shank_length, table["shank"].radii_of_gyration
        )

        foot_mass = self.mass * table["foot"].mass_fraction
        foot_com = (0.0, 0.5 * dims.foot_length, 0.0)
        foot_inertia = _diag_inertia(foot_mass, dims.foot_length, table["foot"].radii_of_gyration)

        parts = [
            "version 4\n\n",
            "gravity\t0.0\t0.0\t0.0\n\n",
            "// x: mediolateral, y: anteroposterior, z: longitudinal\n",
            "// root rotations order: somersault, tilt, twist\n\n",
            _segment_block(
                "pelvis",
                "base",
                (0.0, 0.0, 0.0),
                translations="xyz",
                rotations="xyz",
                mass=trunk_mass,
                center_of_mass=trunk_com,
                inertia=trunk_inertia,
                ranges_q=(
                    (-10.0, 10.0),
                    (-10.0, 10.0),
                    (-10.0, 10.0),
                    (-6.283185, 6.283185),
                    (-6.283185, 6.283185),
                    (-6.283185, 6.283185),
                ),
            ),
            _segment_block(
                "head",
                "pelvis",
                (0.0, 0.0, dims.trunk_length),
                mass=head_mass,
                center_of_mass=head_com,
                inertia=head_inertia,
            ),
        ]

        for side, sign in (("right", 1.0), ("left", -1.0)):
            shoulder_translation = (sign * dims.shoulder_half_width, 0.0, dims.trunk_length)
            hip_translation = (sign * dims.hip_half_width, 0.0, 0.0)

            parts.extend(
                [
                    _segment_block(
                        f"shoulder_{side}_plane",
                        "pelvis",
                        shoulder_translation,
                        rotations="z",
                        ranges_q=((-3.141593, 3.141593),),
                    ),
                    _segment_block(
                        f"upper_arm_{side}",
                        f"shoulder_{side}_plane",
                        (0.0, 0.0, 0.0),
                        rotations="y",
                        mass=upper_arm_mass,
                        center_of_mass=upper_arm_com,
                        inertia=upper_arm_inertia,
                        ranges_q=((-3.141593, 3.141593),),
                    ),
                    _segment_block(
                        f"forearm_{side}",
                        f"upper_arm_{side}",
                        (0.0, 0.0, -dims.upper_arm_length),
                        mass=forearm_mass,
                        center_of_mass=forearm_com,
                        inertia=forearm_inertia,
                    ),
                    _segment_block(
                        f"hand_{side}",
                        f"forearm_{side}",
                        (0.0, 0.0, -dims.forearm_length),
                        mass=hand_mass,
                        center_of_mass=hand_com,
                        inertia=hand_inertia,
                    ),
                    _segment_block(
                        f"thigh_{side}",
                        "pelvis",
                        hip_translation,
                        mass=thigh_mass,
                        center_of_mass=thigh_com,
                        inertia=thigh_inertia,
                    ),
                    _segment_block(
                        f"shank_{side}",
                        f"thigh_{side}",
                        (0.0, 0.0, -dims.thigh_length),
                        mass=shank_mass,
                        center_of_mass=shank_com,
                        inertia=shank_inertia,
                    ),
                    _segment_block(
                        f"foot_{side}",
                        f"shank_{side}",
                        (0.0, 0.0, -dims.shank_length),
                        mass=foot_mass,
                        center_of_mass=foot_com,
                        inertia=foot_inertia,
                    ),
                ]
            )

        parts.extend(
            [
                _marker_block("pelvis_origin", "pelvis", (0.0, 0.0, 0.0)),
                _marker_block("head_top", "head", (0.0, 0.0, dims.head_length)),
            ]
        )

        for side, sign in (("right", 1.0), ("left", -1.0)):
            parts.extend(
                [
                    _marker_block(
                        f"shoulder_{side}",
                        "pelvis",
                        (sign * dims.shoulder_half_width, 0.0, dims.trunk_length),
                    ),
                    _marker_block(
                        f"elbow_{side}", f"upper_arm_{side}", (0.0, 0.0, -dims.upper_arm_length)
                    ),
                    _marker_block(
                        f"wrist_{side}", f"forearm_{side}", (0.0, 0.0, -dims.forearm_length)
                    ),
                    _marker_block(f"hand_{side}", f"hand_{side}", (0.0, 0.0, -dims.hand_length)),
                    _marker_block(f"hip_{side}", "pelvis", (sign * dims.hip_half_width, 0.0, 0.0)),
                    _marker_block(f"knee_{side}", f"thigh_{side}", (0.0, 0.0, -dims.thigh_length)),
                    _marker_block(f"ankle_{side}", f"shank_{side}", (0.0, 0.0, -dims.shank_length)),
                    _marker_block(f"toe_{side}", f"foot_{side}", (0.0, dims.foot_length, 0.0)),
                ]
            )

        return "".join(parts)

    def write(self, path: str | Path) -> Path:
        """Write the generated `bioMod` to disk and return its path."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_biomod_string(), encoding="utf-8")
        return target
