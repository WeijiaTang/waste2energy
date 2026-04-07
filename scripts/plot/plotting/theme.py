from __future__ import annotations

from matplotlib.ticker import MaxNLocator

from scripts.plot.common import CLAIM_COLORS, PATHWAY_COLORS


def configure_publication_theme():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scienceplots  # noqa: F401

    plt.style.use(["science", "nature", "no-latex"])
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.6,
            "axes.titlesize": 8.6,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.65,
            "grid.color": "#E8EDF3",
            "grid.linewidth": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    sns.set_theme(style="white")
    return plt


def pathway_color(pathway: str) -> str:
    return PATHWAY_COLORS.get(str(pathway), "#7C7C7C")


def claim_color(group: str) -> str:
    return CLAIM_COLORS.get(str(group), "#7C7C7C")


def scenario_marker(scenario_name: str) -> str:
    markers = {
        "baseline_region_case": "o",
        "high_supply_case": "s",
        "policy_support_case": "D",
    }
    return markers.get(str(scenario_name), "o")


def soften_hex(hex_color: str, weight: float = 0.82) -> str:
    color = hex_color.lstrip("#")
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    mixed = [
        int(channel * (1.0 - weight) + 255 * weight)
        for channel in (red, green, blue)
    ]
    return "#{:02X}{:02X}{:02X}".format(*mixed)


def style_axis(ax, *, grid_axis: str = "y", grid: bool = True) -> None:
    if grid:
        ax.grid(True, axis=grid_axis, color="#E4EAF1", linewidth=0.55)
    else:
        ax.grid(False)
    ax.tick_params(length=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#8893A0")
    ax.spines["bottom"].set_color("#8893A0")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))


def narrative_background(ax, *, facecolor: str = "#F7F9FC") -> None:
    ax.set_facecolor(facecolor)


def style_polar_axis(ax) -> None:
    ax.set_facecolor("#FBFCFE")
    ax.spines["polar"].set_visible(False)
    ax.grid(color="#E2E8F0", linewidth=0.6)
    ax.set_theta_offset(3.141592653589793 / 2)
    ax.set_theta_direction(-1)
    ax.tick_params(pad=1)


def add_landscape_zones(ax) -> None:
    ax.axvspan(0, 8, color="#F8FAFC", zorder=0)
    ax.axvspan(8, 55, color="#F1F5F9", zorder=0)
    ax.axvspan(55, 105, color="#ECFDF5", zorder=0)
    ax.axhspan(0, 20, color="#FFF7ED", zorder=0)
