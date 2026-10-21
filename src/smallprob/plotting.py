"""Optional, explicit Matplotlib defaults for notebooks and exported figures.

Importing this module does not import Matplotlib or change global settings.
Call ``use_style()`` once, or use ``with plot_style():`` for a temporary style.
"""
from contextlib import contextmanager


def _settings():
    try:
        from matplotlib import cycler
    except ImportError as exc:
        raise ImportError("Plot styling requires Matplotlib: install smallprob[viz].") from exc
    return {
        "figure.figsize": (8, 3.6),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11,
        "axes.labelcolor": "#222522",
        "text.color": "#222522",
        "xtick.color": "#525650",
        "ytick.color": "#525650",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#afb3ac",
        "axes.linewidth": 0.7,
        "axes.axisbelow": True,
        "axes.grid": True,
        "grid.color": "#e4e6e0",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "lines.linewidth": 2,
        "lines.markersize": 5,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "axes.prop_cycle": cycler(color=["#b52d24", "#246892", "#80704b", "#705b91", "#38876a"]),
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "smallprob",
    }


def use_style():
    """Apply readable defaults, with 150-dpi display and 300-dpi raster export.

    SVG/PDF remain vector formats. This explicitly updates Matplotlib's global
    rcParams; use :func:`plot_style` to restore previous settings afterwards.
    """
    settings = _settings()
    import matplotlib as mpl
    mpl.rcParams.update(settings)


@contextmanager
def plot_style():
    """Temporarily apply the Smallprob plotting style, restoring settings on exit."""
    settings = _settings()
    import matplotlib as mpl
    with mpl.rc_context(settings):
        yield
