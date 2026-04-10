"""Helpers to present external matplotlib figures without blocking the GUI."""

from __future__ import annotations

_OPEN_EXTERNAL_FIGURES: list[object] = []


def present_external_figure(figure) -> None:
    """Show one matplotlib figure promptly and non-blockingly when possible."""

    import matplotlib.pyplot as plt

    if "agg" in plt.get_backend().lower():
        return

    if figure not in _OPEN_EXTERNAL_FIGURES:
        _OPEN_EXTERNAL_FIGURES.append(figure)
        canvas = getattr(figure, "canvas", None)
        if canvas is not None and hasattr(canvas, "mpl_connect"):
            def _forget_figure(_event) -> None:
                try:
                    _OPEN_EXTERNAL_FIGURES.remove(figure)
                except ValueError:
                    pass

            try:
                canvas.mpl_connect("close_event", _forget_figure)
            except Exception:
                pass

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
        if hasattr(figure, "show"):
            figure.show()
        if manager is not None and hasattr(manager, "show"):
            manager.show()
        else:
            plt.show(block=False)
    except Exception:
        try:
            plt.show(block=False)
        except Exception:
            return

    manager_window = None if manager is None else getattr(manager, "window", None)
    if manager_window is not None:
        try:
            manager_window.deiconify()
        except Exception:
            pass
        try:
            manager_window.lift()
        except Exception:
            pass
        try:
            manager_window.focus_force()
        except Exception:
            pass

    if canvas is not None and hasattr(canvas, "flush_events"):
        try:
            canvas.flush_events()
        except Exception:
            pass
    try:
        plt.pause(0.001)
    except Exception:
        pass
