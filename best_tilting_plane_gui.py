"""Convenience launcher for the Best Tilting Plane GUI."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from best_tilting_plane.gui import launch_gui

if __name__ == "__main__":
    launch_gui()
