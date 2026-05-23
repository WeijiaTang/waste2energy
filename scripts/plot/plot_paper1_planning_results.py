from __future__ import annotations
import pandas as pd
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / 'src'
for candidate in (ROOT, SRC_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from scripts.plot.common import RESULTS_PLOT_DIR, ROOT, load_planning_visual_bundle
from scripts.plot.plotting.data_pipeline import build_figure_ready_tables, write_figure_ready_tables
from scripts.plot.plotting.exports import build_plot_manifest, save_plot_figure_set
from scripts.plot.plotting.paper1_planning_figures import (
    build_figure_score_comparison,
    build_figure_allocation_stack,
    build_figure_evidence_composition,
    build_figure_confidence_decomposition,
    build_figure_necessity_matrix,
    build_figure_mechanism_frontier,
    build_figure_boundary_regime_map,
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Build Q1-standard Paper 1 planning figures.')
    parser.add_argument('--figures-dir', default=None)
    parser.add_argument('--output-dir', default=None)
    return parser

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    figures_dir = Path(args.figures_dir) if args.figures_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_PLOT_DIR

    metrics, _, _ = load_planning_visual_bundle(figures_dir=figures_dir)
    tables = build_figure_ready_tables(metrics)
    
    root_data = figures_dir if figures_dir else ROOT / 'data' / 'processed' / 'figures_tables'
    confidence_df = pd.read_csv(root_data / 'paper1_recommendation_confidence_summary.csv')
    benchmark_df = pd.read_csv(root_data / 'paper1_benchmark_claim_summary.csv')
    pathway_summary_path = ROOT / 'outputs' / 'planning' / 'pathway_summary.csv'
    pathway_summary_df = pd.read_csv(pathway_summary_path)
    pathway_summary_df = pathway_summary_df[
        pathway_summary_df['pathway'].astype(str).str.lower().isin(['pyrolysis', 'htc'])
    ].copy()
    portfolio_allocations_path = ROOT / 'outputs' / 'planning' / 'portfolio_allocations.csv'
    portfolio_allocations_df = pd.read_csv(portfolio_allocations_path)
    portfolio_allocations_df = portfolio_allocations_df[
        portfolio_allocations_df['pathway'].astype(str).str.lower().isin(['pyrolysis', 'htc'])
    ].copy()
    ablation_summary_path = (
        ROOT
        / 'outputs'
        / 'benchmark'
        / 'baseline'
        / 'targeted_planning_ablations'
        / 'targeted_planning_ablations_summary.csv'
    )
    if not ablation_summary_path.exists():
        ablation_summary_path = ROOT / 'outputs' / 'benchmark' / 'targeted_planning_ablations' / 'targeted_planning_ablations_summary.csv'
    ablation_summary_df = pd.read_csv(ablation_summary_path)

    # Build and Save individual figures
    fig1 = build_figure_allocation_stack(pathway_summary_df)
    fig2 = build_figure_score_comparison(tables['figure1_main'])
    fig3 = build_figure_evidence_composition(portfolio_allocations_df)
    fig4 = build_figure_confidence_decomposition(confidence_df)
    fig5 = build_figure_necessity_matrix(benchmark_df)
    fig6 = build_figure_mechanism_frontier(tables['figure1_main'])
    fig_boundary = build_figure_boundary_regime_map(ablation_summary_df)

    outputs = {
        'paper1_fig1_allocation_stack': save_plot_figure_set(fig1, 'paper1_fig1_allocation_stack', output_dir=output_dir),
        'paper1_fig2_score_leadership': save_plot_figure_set(fig2, 'paper1_fig2_score_leadership', output_dir=output_dir),
        'paper1_fig3_evidence_tier': save_plot_figure_set(fig3, 'paper1_fig3_evidence_tier', output_dir=output_dir),
        'paper1_fig4_confidence_profile': save_plot_figure_set(fig4, 'paper1_fig4_confidence_profile', output_dir=output_dir),
        'paper1_fig5_necessity_matrix': save_plot_figure_set(fig5, 'paper1_fig5_necessity_matrix', output_dir=output_dir),
        'paper1_fig6_mechanism_frontier': save_plot_figure_set(fig6, 'paper1_fig6_mechanism_frontier', output_dir=output_dir),
        'paper1_fig6_boundary_regime_map': save_plot_figure_set(fig_boundary, 'paper1_fig6_boundary_regime_map', output_dir=output_dir),
    }

    print(json.dumps({'output_dir': str(output_dir), 'figures': list(outputs.keys())}, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
