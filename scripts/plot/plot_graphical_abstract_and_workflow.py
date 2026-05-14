
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "plot"
FORMATS = ("pdf", "png", "tiff", "eps")

COLORS = {
    "blue": "#0B525B",
    "red": "#A44A3F",
    "green": "#1B998B",
    "orange": "#D97706",
    "purple": "#7C3AED",
    "gray": "#64748B",
    "light": "#F8FAFC",
    "line": "#334155",
}


def _setup_dirs() -> None:
    for ext in FORMATS:
        (OUT / ext).mkdir(parents=True, exist_ok=True)


def _save(fig, stem: str) -> None:
    _setup_dirs()
    for ext in FORMATS:
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.04, "facecolor": "white"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        if ext == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(OUT / ext / f"{stem}.{ext}", format=ext, **kwargs)


def _box(ax, xy, w, h, title, subtitle="", color="#0B525B", fc="#F8FAFC", lw=1.5):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        facecolor=fc,
        edgecolor=color,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#0F172A")
    if subtitle:
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center", fontsize=7.2, color="#334155", linespacing=1.15)
    return patch


def _arrow(ax, start, end, color="#334155"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color=color))


def build_graphical_abstract():
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(12.0, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.94, "Evidence-qualified screening for mixed organic waste-to-energy management",
            fontsize=13, fontweight="bold", color="#0F172A", ha="left")
    ax.text(0.02, 0.89, "Decision support under uneven evidence, regional policy constraints, and coproduct-market uncertainty",
            fontsize=8.5, color="#475569", ha="left")

    xs = [0.03, 0.22, 0.41, 0.60, 0.79]
    y = 0.54
    w, h = 0.155, 0.22
    _box(ax, (xs[0], y), w, h, "Evidence base", "150 prototype cases\nAD / pyrolysis / HTC", COLORS["gray"])
    _box(ax, (xs[1], y), w, h, "Transfer audit", "Leave-study-out R²\nclaim ceilings", COLORS["green"])
    _box(ax, (xs[2], y), w, h, "Planning set", "450 scenario candidates\nno added evidence depth", COLORS["blue"])
    _box(ax, (xs[3], y), w, h, "Portfolio screen", "MILP constraints\nshare, diversity, carbon", COLORS["purple"])
    _box(ax, (xs[4], y), w, h, "Claim output", "screening recommendation\nnot siting prescription", COLORS["orange"])
    for i in range(4):
        _arrow(ax, (xs[i] + w + 0.01, y + h / 2), (xs[i+1] - 0.01, y + h / 2))

    # Boundary diagnosis panel
    panel = FancyBboxPatch((0.06, 0.10), 0.88, 0.27, boxstyle="round,pad=0.018,rounding_size=0.035",
                           facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.2)
    ax.add_patch(panel)
    ax.text(0.08, 0.32, "Key decision insight", fontsize=10.5, fontweight="bold", color="#0F172A", ha="left")

    # Three result badges
    _box(ax, (0.09, 0.15), 0.24, 0.12, "Conservative baseline", "pyrolysis-led screening anchor", COLORS["red"], fc="#FFF7ED")
    _box(ax, (0.38, 0.15), 0.24, 0.12, "Hydrochar credit", "can reverse to HTC-led portfolios", COLORS["blue"], fc="#ECFEFF")
    _box(ax, (0.67, 0.15), 0.22, 0.12, "AD benchmark", "proxy/evidence-limited; not a verdict", COLORS["green"], fc="#F0FDF4")
    _arrow(ax, (0.33, 0.21), (0.38, 0.21), COLORS["line"])
    _arrow(ax, (0.62, 0.21), (0.67, 0.21), COLORS["line"])

    ax.text(0.50, 0.045, "Boundary-sensitive portfolio screening: evidence and coproduct-market assumptions determine what can be claimed.",
            ha="center", va="center", fontsize=8.5, color="#334155", fontweight="bold")
    return fig


def build_methods_workflow():
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.95, "Methods workflow and audit trail", fontsize=13, fontweight="bold", color="#0F172A")
    ax.text(0.03, 0.91, "Every stage preserves provenance so the optimized allocation and claim language can be audited.", fontsize=8.5, color="#475569")

    left_x, right_x = 0.08, 0.56
    ys = [0.78, 0.60, 0.42, 0.24]
    _box(ax, (left_x, ys[0]), 0.30, 0.12, "1. Source-preserving table", "prototype rows + pathway/cost boundary metadata", COLORS["gray"])
    _box(ax, (right_x, ys[0]), 0.30, 0.12, "2. Surrogate validation", "strict-group and leave-study-out evaluation", COLORS["green"])
    _box(ax, (left_x, ys[1]), 0.30, 0.12, "3. Evidence hierarchy", "supportive / weak / unsupported tiers", COLORS["green"])
    _box(ax, (right_x, ys[1]), 0.30, 0.12, "4. Scenario assembly", "baseline, high-supply, policy-support candidates", COLORS["blue"])
    _box(ax, (left_x, ys[2]), 0.30, 0.12, "5. Score construction", "energy, carbon, cost, uncertainty, evidence", COLORS["purple"])
    _box(ax, (right_x, ys[2]), 0.30, 0.12, "6. Constrained portfolio", "candidate caps, diversity, capacity, carbon guardrail", COLORS["purple"])
    _box(ax, (left_x, ys[3]), 0.30, 0.12, "7. Boundary sensitivity", "η sweep, objective weights, coproduct credits", COLORS["orange"])
    _box(ax, (right_x, ys[3]), 0.30, 0.12, "8. Synchronized reporting", "allocation + evidence ceiling + limitations", COLORS["orange"])

    pts = [
        (left_x+0.30, ys[0]+0.06, right_x, ys[0]+0.06),
        (right_x+0.15, ys[0], right_x+0.15, ys[1]+0.12),
        (right_x, ys[1]+0.06, left_x+0.30, ys[1]+0.06),
        (left_x+0.15, ys[1], left_x+0.15, ys[2]+0.12),
        (left_x+0.30, ys[2]+0.06, right_x, ys[2]+0.06),
        (right_x+0.15, ys[2], right_x+0.15, ys[3]+0.12),
        (right_x, ys[3]+0.06, left_x+0.30, ys[3]+0.06),
    ]
    for x1,y1,x2,y2 in pts:
        _arrow(ax, (x1,y1), (x2,y2))

    # Audit side rail
    ax.add_line(Line2D([0.04, 0.04], [0.20, 0.86], color="#CBD5E1", linewidth=3))
    for yy, label in zip([0.84,0.66,0.48,0.30], ["provenance", "evidence", "constraints", "claim ceiling"]):
        ax.add_patch(Rectangle((0.025, yy-0.015), 0.03, 0.03, facecolor="#E2E8F0", edgecolor="#94A3B8"))
        ax.text(0.065, yy, label, fontsize=7.5, color="#475569", va="center")
    return fig


def main() -> int:
    for stem, builder in [
        ("paper1_graphical_abstract", build_graphical_abstract),
        ("paper1_methods_workflow", build_methods_workflow),
    ]:
        fig = builder()
        _save(fig, stem)
        plt.close(fig)
    print("Generated graphical abstract and methods workflow in results/plot/{pdf,png,tiff,eps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
