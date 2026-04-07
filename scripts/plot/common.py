from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from waste2energy.config import FIGURES_TABLES_DIR


RESULTS_PLOT_DIR = ROOT / "results" / "plot"
EXPORT_FORMATS = ("eps", "pdf", "png", "tiff")
PATHWAY_ORDER = ["htc", "pyrolysis", "ad", "baseline"]
PATHWAY_LABELS = {
    "htc": "HTC",
    "pyrolysis": "Pyrolysis",
    "ad": "AD",
    "baseline": "Baseline",
}
SCENARIO_LABELS = {
    "baseline_region_case": "Baseline region",
    "high_supply_case": "High supply",
    "policy_support_case": "Policy support",
}
PATHWAY_COLORS = {
    "htc": "#0B525B",
    "pyrolysis": "#A44A3F",
    "ad": "#6D8F3C",
    "baseline": "#8D99AE",
}
CLAIM_COLORS = {
    "planning_ready": "#1B998B",
    "comparison_only": "#D1495B",
    "anchor_only": "#4F5D75",
    "other": "#7C7C7C",
}
SELECTION_COLORS = {
    "selected": "#111111",
    "not_selected": "#C7CCD6",
}


def configure_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scienceplots  # noqa: F401

    plt.style.use(["science", "nature", "no-latex"])
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "figure.titlesize": 10.5,
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "legend.title_fontsize": 7.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    sns.set_theme(style="whitegrid")
    return plt, sns


def load_planning_visual_bundle(
    *,
    figures_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    root = figures_dir if figures_dir else FIGURES_TABLES_DIR
    metrics = pd.read_csv(root / "paper1_planning_visual_metrics_long.csv")
    annotations = pd.read_csv(root / "paper1_planning_visual_annotations.csv")
    manifest = json.loads((root / "paper1_planning_visual_manifest.json").read_text(encoding="utf-8"))
    return metrics, annotations, manifest


def ensure_results_dir(output_dir: Path | None = None) -> Path:
    target = output_dir if output_dir else RESULTS_PLOT_DIR
    target.mkdir(parents=True, exist_ok=True)
    (target / "data").mkdir(parents=True, exist_ok=True)
    for extension in EXPORT_FORMATS:
        (target / extension).mkdir(parents=True, exist_ok=True)
    return target


def save_figure_set(fig, stem: str, *, output_dir: Path | None = None) -> dict[str, str]:
    target_dir = ensure_results_dir(output_dir)
    outputs: dict[str, str] = {}
    for extension in EXPORT_FORMATS:
        format_dir = target_dir / extension
        path = format_dir / f"{stem}.{extension}"
        save_kwargs: dict[str, Any] = {
            "bbox_inches": "tight",
            "pad_inches": 0.02,
            "facecolor": "white",
        }
        if extension in {"png", "tiff"}:
            save_kwargs["dpi"] = 600
        if extension == "tiff":
            save_kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, format=extension, **save_kwargs)
        outputs[extension] = str(path)
    return outputs


def scenario_label(value: object) -> str:
    return SCENARIO_LABELS.get(str(value), str(value))


def pathway_label(value: object) -> str:
    return PATHWAY_LABELS.get(str(value), str(value))


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.14,
        1.02,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def format_pct(value: float) -> str:
    return f"{value:.1f}%"
