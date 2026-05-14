from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from scripts.plot.common import RESULTS_PLOT_DIR, ensure_results_dir
from scripts.plot.plotting.theme import configure_publication_theme


OCEAN = {
    "ink": "#263238",
    "muted": "#64748B",
    "line": "#CBD5E1",
    "blue": "#2563EB",
    "teal": "#0B525B",
    "mint": "#2A9D8F",
    "gold": "#E9C46A",
    "orange": "#F4A261",
    "coral": "#E76F51",
    "green": "#6D8F3C",
    "slate_bg": "#F8FAFC",
    "blue_bg": "#EAF2FF",
    "teal_bg": "#E6F4F1",
    "gold_bg": "#FFF7DF",
    "coral_bg": "#FFF0EB",
    "gray_bg": "#F1F5F9",
}


def _round_box(ax, xy, width, height, title, body, *, fc, ec, title_color=None, fontsize=8.2):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=1.1,
        edgecolor=ec,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=fontsize + 0.9,
        color=title_color or ec,
        fontweight="bold",
        zorder=3,
    )
    ax.text(
        x + width / 2,
        y + height * 0.33,
        body,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=OCEAN["ink"],
        linespacing=1.15,
        zorder=3,
    )
    return box


def _arrow(ax, start, end, *, color=OCEAN["muted"], rad=0.0, lw=1.35, ls="-"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=color,
        linestyle=ls,
        shrinkA=3,
        shrinkB=3,
        connectionstyle=f"arc3,rad={rad}",
        zorder=4,
    )
    ax.add_patch(arrow)
    return arrow


def _tag(ax, x, y, text, *, fc, color="white", fontsize=7.2, width=None):
    if width is None:
        width = max(0.105, 0.0085 * len(text))
    height = 0.045
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0,
        facecolor=fc,
        zorder=5,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=color, fontweight="bold", zorder=6)


def _mini_bar(ax, x, y, w, h, fractions, colors, labels):
    left = x
    for frac, color, label in zip(fractions, colors, labels):
        ax.add_patch(Rectangle((left, y), w * frac, h, facecolor=color, edgecolor="white", linewidth=0.8, zorder=4))
        if frac > 0.13:
            ax.text(left + w * frac / 2, y + h / 2, label, ha="center", va="center", fontsize=6.5, color="white", fontweight="bold", zorder=5)
        left += w * frac
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor="#94A3B8", linewidth=0.8, zorder=5))


def build_graphical_abstract():
    plt = configure_publication_theme()
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.02,
        0.965,
        "Evidence-qualified portfolio screening for mixed organic waste-to-energy management",
        fontsize=13.5,
        fontweight="bold",
        color=OCEAN["ink"],
        ha="left",
        va="top",
    )
    ax.text(
        0.02,
        0.918,
        "The workflow converts uneven literature evidence into boundary-aware, claim-limited regional screening recommendations.",
        fontsize=8.6,
        color=OCEAN["muted"],
        ha="left",
        va="top",
    )

    # Pipeline panels
    y = 0.59
    h = 0.20
    w = 0.158
    xs = [0.025, 0.215, 0.405, 0.595, 0.785]
    stages = [
        ("1 Literature cases", "prototype rows\nPyrolysis | HTC | AD", OCEAN["blue_bg"], OCEAN["blue"]),
        ("2 Evidence audit", "leave-study-out\nclaim strength tier", OCEAN["teal_bg"], OCEAN["mint"]),
        ("3 Planning candidates", "scenario-specific\nregional portfolios", OCEAN["gold_bg"], "#B7791F"),
        ("4 Constrained screen", "share, carbon and\npolicy guardrails", OCEAN["gray_bg"], "#475569"),
        ("5 Boundary diagnosis", "coproduct markets\nclaim ceiling", OCEAN["coral_bg"], OCEAN["coral"]),
    ]
    centers = []
    for x, (title, body, fc, ec) in zip(xs, stages):
        _round_box(ax, (x, y), w, h, title, body, fc=fc, ec=ec)
        centers.append((x + w / 2, y + h / 2))
    for i in range(len(centers) - 1):
        _arrow(ax, (xs[i] + w + 0.006, y + h / 2), (xs[i + 1] - 0.006, y + h / 2), color=OCEAN["muted"])

    # Evidence and screening result band
    band = FancyBboxPatch(
        (0.025, 0.205),
        0.95,
        0.275,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=0.9,
        edgecolor="#D5DEE8",
        facecolor="#FBFCFE",
        zorder=1,
    )
    ax.add_patch(band)
    ax.text(0.047, 0.455, "Waste-management interpretation", fontsize=10.6, fontweight="bold", color=OCEAN["ink"], ha="left", va="center")

    # Baseline card
    _round_box(
        ax,
        (0.055, 0.265),
        0.275,
        0.135,
        "Conservative baseline",
        "pyrolysis-led screening anchor\nunder evidence-qualified credits",
        fc="#FFF8F4",
        ec=OCEAN["coral"],
        title_color=OCEAN["coral"],
        fontsize=7.5,
    )
    _mini_bar(ax, 0.085, 0.225, 0.215, 0.035, [0.89, 0.11], ["#A44A3F", "#0B525B"], ["Pyrolysis", ""])
    ax.text(0.192, 0.207, "small HTC diversifier in synchronized baseline outputs", ha="center", va="top", fontsize=6.6, color=OCEAN["muted"])

    # Boundary reversal card
    _round_box(
        ax,
        (0.365, 0.265),
        0.275,
        0.135,
        "Hydrochar-credit boundary",
        "can reverse thermochemical\nportfolio identity to HTC",
        fc="#ECFDF8",
        ec=OCEAN["teal"],
        title_color=OCEAN["teal"],
        fontsize=7.5,
    )
    _mini_bar(ax, 0.395, 0.225, 0.215, 0.035, [1.0], ["#0B525B"], ["HTC"])
    _tag(ax, 0.502, 0.207, "market evidence is decision-critical", fc=OCEAN["teal"], fontsize=6.7, width=0.22)

    # AD benchmark card
    _round_box(
        ax,
        (0.675, 0.265),
        0.275,
        0.135,
        "AD benchmark",
        "management-relevant but\nproxy/evidence-limited here",
        fc="#F4F8EC",
        ec=OCEAN["green"],
        title_color=OCEAN["green"],
        fontsize=7.5,
    )
    _mini_bar(ax, 0.705, 0.225, 0.215, 0.035, [1.0], ["#E2E8F0"], [""])
    _tag(ax, 0.812, 0.207, "no deployment-level AD claim", fc=OCEAN["green"], fontsize=6.7, width=0.205)

    # Claim ceiling below pipeline
    _arrow(ax, (0.865, 0.585), (0.865, 0.49), color=OCEAN["coral"], rad=0.0, lw=1.5)
    ax.text(
        0.865,
        0.515,
        "claim ceiling",
        ha="center",
        va="center",
        fontsize=7.4,
        color=OCEAN["coral"],
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor=OCEAN["coral"], linewidth=0.8),
        zorder=7,
    )

    # Footer legend
    legend_y = 0.095
    ax.text(0.025, legend_y, "Pathway colors:", fontsize=7.8, color=OCEAN["muted"], ha="left", va="center", fontweight="bold")
    legend_items = [("Pyrolysis", "#A44A3F"), ("HTC", "#0B525B"), ("AD", "#6D8F3C"), ("Unselected / outside claim", "#E2E8F0")]
    lx = 0.15
    for label, color in legend_items:
        ax.add_patch(Rectangle((lx, legend_y - 0.012), 0.018, 0.024, facecolor=color, edgecolor="#CBD5E1", linewidth=0.5))
        ax.text(lx + 0.024, legend_y, label, fontsize=7.4, color=OCEAN["muted"], ha="left", va="center")
        lx += 0.155 if label != "Unselected / outside claim" else 0.23

    ax.text(
        0.975,
        0.095,
        "Use: pre-feasibility screening and evidence-gap prioritization, not facility siting.",
        fontsize=7.3,
        color=OCEAN["muted"],
        ha="right",
        va="center",
    )
    return fig


def main() -> int:
    output_dir = ensure_results_dir(RESULTS_PLOT_DIR)
    fig = build_graphical_abstract()
    outputs = {}
    for extension in ("pdf", "png", "tiff", "eps"):
        path = output_dir / extension / f"paper1_graphical_abstract.{extension}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.04, "facecolor": "white"}
        if extension in {"png", "tiff"}:
            kwargs["dpi"] = 600
        if extension == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, format=extension, **kwargs)
        outputs[extension] = str(path)
    print("Generated graphical abstract:")
    for key, value in outputs.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
