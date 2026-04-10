"""Helpers to present external matplotlib figures without blocking the GUI."""

from __future__ import annotations


def present_external_figure(figure) -> None:
    """Show one matplotlib figure promptly and non-blockingly when possible."""

    import matplotlib.pyplot as plt

    if "agg" in plt.get_backend().lower():
        return

    canvas = getattr(figure, "canvas", None)
    if canvas is not None:
        try:
            canvas.draw()
        except Exception:
            try:
                canvas.draw_idle()
            except Exception:
                pass

    manager = None if canvas is None else getattr(canvas, "manager", None)
    try:
        if manager is not None and hasattr(manager, "show"):
            manager.show()
        else:
            plt.show(block=False)
    except Exception:
        try:
            plt.show(block=False)
        except Exception:
            return

    if canvas is not None and hasattr(canvas, "flush_events"):
        try:
            canvas.flush_events()
        except Exception:
            pass
    try:
        plt.pause(0.001)
    except Exception:
        pass
