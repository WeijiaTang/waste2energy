# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import json

import pandas as pd

from waste2energy import highcase


def test_highcase_orchestrates_selected_phases_with_manifest(tmp_path, monkeypatch):
    calls: list[str] = []

    def fake_planning(**kwargs):
        calls.append("planning")
        planning_dir = kwargs["output_dir"]
        pd.DataFrame([{"allocated_feed_ton_per_year": 1.0}]).to_csv(
            f"{planning_dir}/portfolio_allocations.csv",
            index=False,
        )
        pd.DataFrame(
            [
                {
                    "summary_scope": "all_planning_rows",
                    "row_count": 1,
                    "independent_observation_count": 0,
                    "scenario_expanded_count": 1,
                }
            ]
        ).to_csv(f"{planning_dir}/planning_data_contract_summary.csv", index=False)
        pd.DataFrame([{"warning": "scenario_expanded_rows_must_not_be_counted_as_independent_evidence"}]).to_csv(
            f"{planning_dir}/planning_data_contract_warnings.csv",
            index=False,
        )
        pd.DataFrame([{"transferability_risk_label": "selected_share_conditionally_transferable"}]).to_csv(
            f"{planning_dir}/surrogate_transferability_summary.csv",
            index=False,
        )
        with open(f"{planning_dir}/run_config.json", "w", encoding="utf-8") as handle:
            json.dump({}, handle)
        with open(f"{planning_dir}/reproducibility_manifest.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "inputs": [{"exists": True, "sha256": "abc"}],
                    "outputs": [{"exists": True, "sha256": "def"}],
                },
                handle,
            )
        return {"outputs": {"run_config": f"{planning_dir}/run_config.json"}}

    monkeypatch.setattr(highcase, "run_planning_baseline", fake_planning)

    result = highcase.run_highcase(
        output_root=tmp_path / "highcase",
        profile_name="smoke",
        run_benchmark=False,
        run_targeted_ablations=False,
        run_scenario=False,
        run_operation=False,
    )

    assert calls == ["planning"]
    assert result["profile"] == "smoke"
    assert (tmp_path / "highcase" / "quality" / "quality_gate_summary.csv").exists()
    assert (tmp_path / "highcase" / "highcase_manifest.json").exists()

