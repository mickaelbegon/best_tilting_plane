"""Shared IPOPT solver-option helpers."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import sys

PREFERRED_HSL_ENV_NAMES = ("vitpose-ekf", "Dev_bioptim", "casadi3_5")
HSL_LIBRARY_NAMES = (
    "libhsl.dylib",
    "libhsl.so",
    "libcoinhsl.dylib",
    "libcoinhsl.so",
)
HSL_ENVIRONMENT_VARIABLE = "BEST_TILTING_PLANE_IPOPT_HSL"


def _append_unique(paths: list[Path], seen: set[Path], candidate: Path) -> None:
    """Add one existing path while preserving insertion order."""

    resolved = candidate.expanduser().resolve()
    if resolved in seen:
        return
    seen.add(resolved)
    paths.append(resolved)


def _candidate_conda_prefixes() -> list[Path]:
    """Return the conda prefixes worth probing for an HSL shared library."""

    candidates: list[Path] = []
    seen: set[Path] = set()

    current_prefix = Path(os.environ.get("CONDA_PREFIX", sys.prefix)).expanduser()
    if current_prefix.exists():
        _append_unique(candidates, seen, current_prefix)

    for root in (Path.home() / "miniconda3", Path.home() / "opt" / "miniconda3"):
        if not root.exists():
            continue
        _append_unique(candidates, seen, root)

        env_root = root / "envs"
        if not env_root.exists():
            continue

        for env_name in PREFERRED_HSL_ENV_NAMES:
            env_prefix = env_root / env_name
            if env_prefix.exists():
                _append_unique(candidates, seen, env_prefix)

        for env_prefix in sorted(path for path in env_root.iterdir() if path.is_dir()):
            _append_unique(candidates, seen, env_prefix)

    return candidates


@lru_cache(maxsize=1)
def locate_ipopt_hsl_library() -> str | None:
    """Locate one HSL library that IPOPT can use for MA57."""

    explicit_path = os.environ.get(HSL_ENVIRONMENT_VARIABLE)
    if explicit_path:
        explicit_candidate = Path(explicit_path).expanduser()
        if explicit_candidate.is_file():
            return str(explicit_candidate.resolve())

    for prefix in _candidate_conda_prefixes():
        lib_dir = prefix / "lib"
        for library_name in HSL_LIBRARY_NAMES:
            candidate = lib_dir / library_name
            if candidate.is_file():
                return str(candidate.resolve())

    return None


def build_ipopt_solver_options(
    *,
    max_iter: int,
    print_level: int,
    print_time: bool,
    expand: bool | None = None,
    warm_start: bool = False,
) -> dict[str, object]:
    """Build one consistent IPOPT option dictionary for every CasADi solver."""

    options: dict[str, object] = {
        "ipopt.max_iter": int(max_iter),
        "ipopt.print_level": int(print_level),
        "print_time": int(bool(print_time)),
    }
    if expand is not None:
        options["expand"] = bool(expand)

    hsllib = locate_ipopt_hsl_library()
    if hsllib is not None:
        options["ipopt.linear_solver"] = "ma57"
        options["ipopt.hsllib"] = hsllib

    if warm_start:
        options["ipopt.warm_start_init_point"] = "yes"
        options["ipopt.warm_start_bound_push"] = 1e-8
        options["ipopt.warm_start_mult_bound_push"] = 1e-8

    return options
