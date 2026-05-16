from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_htc_planning_grid_expands_with_condition_provenance():
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "data-process" / "11_build_planning_mult_pathway_dataset.py"),
        ],
        cwd=ROOT,
        check=True,
    )

    prototypes = pd.read_csv(ROOT / "data" / "processed" / "unified_features" / "paper1_planning_pathway_prototypes.csv")
    htc = prototypes[prototypes["pathway"].eq("htc")].copy()

    assert len(htc) > 60
    assert htc[["process_temperature_c", "residence_time_min"]].drop_duplicates().shape[0] >= 8
    assert "condition_source_count" in htc.columns
    assert pd.to_numeric(htc["condition_source_count"], errors="coerce").fillna(0).gt(0).any()
