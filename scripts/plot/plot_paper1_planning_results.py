from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
for candidate in (ROOT, SRC_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from scripts.plot.common import RESULTS_PLOT_DIR, load_planning_visual_bundle
from scripts.plot.plotting.data_pipeline import build_figure_ready_tables, write_figure_ready_tables
from scripts.plot.plotting.exports import build_plot_manifest, save_plot_figure_set
from scripts.plot.plotting.paper1_planning_figures import (
    build_figure1_main,
    build_figure2_tradeoff,
    build_figure3_robustness,
    build_sup_figure_s1_scenario_fingerprint,
    build_sup_figure_s2_dominance_evidence_landscape,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build publication-grade Paper 1 planning figures into results/plot.",
    )
    parser.add_argument(
        "--figures-dir",
        default=None,
        help="Directory containing paper1 planning visual source tables. Defaults to data/processed/figures_tables.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Canonical output root. Defaults to results/plot.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    figures_dir = Path(args.figures_dir) if args.figures_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_PLOT_DIR

    metrics, _, source_manifest = load_planning_visual_bundle(figures_dir=figures_dir)
    tables = build_figure_ready_tables(metrics)
    data_outputs = write_figure_ready_tables(tables, output_dir=output_dir / "data")

    fig1 = build_figure1_main(tables["figure1_main"])
    fig2 = build_figure2_tradeoff(tables["figure2_tradeoff"])
    fig3 = build_figure3_robustness(tables["figure3_robustness"])
    fig_s1 = build_sup_figure_s1_scenario_fingerprint(tables["paper1_sup_s1_scenario_fingerprint"])
    fig_s2 = build_sup_figure_s2_dominance_evidence_landscape(
        tables["paper1_sup_s2_dominance_evidence_landscape"]
    )

    outputs = {
        "paper1_fig1_decision_narrative": save_plot_figure_set(
            fig1, "paper1_fig1_decision_narrative", output_dir=output_dir
        ),
        "paper1_fig2_tradeoff_mechanism": save_plot_figure_set(
            fig2, "paper1_fig2_tradeoff_mechanism", output_dir=output_dir
        ),
        "paper1_fig3_robustness_evidence_boundary": save_plot_figure_set(
            fig3, "paper1_fig3_robustness_evidence_boundary", output_dir=output_dir
        ),
        "paper1_sup_fig_s1_scenario_fingerprint": save_plot_figure_set(
            fig_s1, "paper1_sup_fig_s1_scenario_fingerprint", output_dir=output_dir
        ),
        "paper1_sup_fig_s2_dominance_evidence_landscape": save_plot_figure_set(
            fig_s2, "paper1_sup_fig_s2_dominance_evidence_landscape", output_dir=output_dir
        ),
    }

    manifest = build_plot_manifest(
        outputs=outputs,
        data_outputs=data_outputs,
        output_dir=output_dir,
    )
    manifest["source_manifest"] = source_manifest
    manifest_path = output_dir / "paper1_planning_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "data_outputs": data_outputs,
                "figures": outputs,
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
