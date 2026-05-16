from __future__ import annotations

import pandas as pd

from waste2energy.planning.solve import PlanningConfig, run_planning_baseline


def test_primary_optimizer_excludes_ad_by_default_but_can_include_policy_floor(tmp_path):
    default_dir = tmp_path / "default"
    run_planning_baseline(
        output_dir=str(default_dir),
        config=PlanningConfig(pareto_point_count=0, enable_pareto_export=False),
    )

    default_scored = pd.read_csv(default_dir / "scored_cases.csv")
    default_allocations = pd.read_csv(default_dir / "portfolio_allocations.csv")
    assert "ad" not in set(default_scored["pathway"].astype(str))
    assert "ad" not in set(default_allocations["pathway"].astype(str))
    assert set(default_scored["primary_optimization_pathway_scope"]) == {"htc,pyrolysis"}

    ad_floor_dir = tmp_path / "ad_floor"
    run_planning_baseline(
        output_dir=str(ad_floor_dir),
        config=PlanningConfig(
            pareto_point_count=0,
            enable_pareto_export=False,
            min_pathway_share=(("ad", 0.10),),
            enforce_min_distinct_subtypes=False,
        ),
    )

    ad_floor_scored = pd.read_csv(ad_floor_dir / "scored_cases.csv")
    ad_floor_allocations = pd.read_csv(ad_floor_dir / "portfolio_allocations.csv")
    assert "ad" in set(ad_floor_scored["pathway"].astype(str))
    assert "ad" in set(ad_floor_allocations["pathway"].astype(str))
