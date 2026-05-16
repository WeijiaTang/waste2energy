from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_ad_reference_layer_exports_traceable_reference_rows():
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "data-process" / "11_build_planning_mult_pathway_dataset.py"),
        ],
        cwd=ROOT,
        check=True,
    )

    observations_path = ROOT / "data" / "processed" / "figures_tables" / "paper1_ad_reference_observation_table.csv"
    summary_path = ROOT / "data" / "processed" / "figures_tables" / "paper1_ad_reference_summary_by_group.csv"
    observations = pd.read_csv(observations_path)
    summary = pd.read_csv(summary_path)

    assert len(observations) >= 80
    assert observations["planning_use"].eq("reference_only_not_primary_optimizer").all()
    assert observations["source_file"].astype(str).nunique() >= 5
    assert "facility-level cost" in observations["claim_boundary"].iloc[0]
    assert not summary.empty
    assert summary["planning_use"].eq("reference_only_not_primary_optimizer").all()
