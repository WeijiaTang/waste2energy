from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _check(
    checks: list[dict[str, object]],
    *,
    name: str,
    passed: bool,
    detail: str,
    path: Path | None = None,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "detail": detail,
            "path": str(path) if path is not None else "",
        }
    )


def build_readiness_report(
    *,
    repo_root: Path = ROOT,
    require_bootstrap: bool = False,
    min_bootstrap_replicates: int = 1,
    require_targeted_ablations: bool = False,
    min_monte_carlo_replicates: int = 1,
    require_pdf: bool = False,
) -> dict[str, object]:
    checks: list[dict[str, object]] = []

    consistency_path = repo_root / "outputs" / "audit" / "planning_artifact_consistency_summary.csv"
    consistency = _read_csv(consistency_path)
    consistency_passed = (
        not consistency.empty
        and "consistency_status" in consistency.columns
        and consistency["consistency_status"].astype(str).eq("pass").all()
    )
    _check(
        checks,
        name="planning_artifact_consistency",
        passed=consistency_passed,
        detail=(
            "All manuscript-facing planning shares match portfolio_allocations."
            if consistency_passed
            else "Missing or failing planning artifact consistency rows."
        ),
        path=consistency_path,
    )

    benchmark_dir = repo_root / "outputs" / "benchmark" / "baseline"
    benchmark_run_config_path = benchmark_dir / "run_config.json"
    benchmark_run_config = _read_json(benchmark_run_config_path)
    bootstrap_count = int(benchmark_run_config.get("bootstrap_replicate_count", 0) or 0)
    bootstrap_stats_path = benchmark_dir / "benchmark_statistical_summary.csv"
    bootstrap_stats = _read_csv(bootstrap_stats_path)
    bootstrap_passed = bootstrap_count >= int(min_bootstrap_replicates) and not bootstrap_stats.empty
    _check(
        checks,
        name="benchmark_bootstrap_statistics",
        passed=(bootstrap_passed if require_bootstrap else True),
        detail=(
            f"Bootstrap replicate count={bootstrap_count}; statistical rows={len(bootstrap_stats)}."
            if require_bootstrap
            else "Bootstrap statistics not required by this readiness invocation."
        ),
        path=bootstrap_stats_path,
    )

    targeted_dirs = [
        benchmark_dir / "targeted_planning_ablations",
        repo_root / "outputs" / "benchmark" / "targeted_planning_ablations",
    ]
    targeted_dir = next((path for path in targeted_dirs if (path / "run_config.json").exists()), targeted_dirs[0])
    targeted_run_config_path = targeted_dir / "run_config.json"
    targeted_run_config = _read_json(targeted_run_config_path)
    monte_carlo_count = int(targeted_run_config.get("monte_carlo_replicate_count", 0) or 0)
    targeted_summary_path = targeted_dir / "targeted_planning_ablations_summary.csv"
    monte_carlo_summary_path = targeted_dir / "monte_carlo_uq_summary.csv"
    targeted_summary = _read_csv(targeted_summary_path)
    monte_carlo_summary = _read_csv(monte_carlo_summary_path)
    targeted_passed = (
        monte_carlo_count >= int(min_monte_carlo_replicates)
        and not targeted_summary.empty
        and not monte_carlo_summary.empty
    )
    _check(
        checks,
        name="targeted_ablation_monte_carlo",
        passed=(targeted_passed if require_targeted_ablations else True),
        detail=(
            f"Monte Carlo replicate count={monte_carlo_count}; targeted rows={len(targeted_summary)}; "
            f"Monte Carlo summary rows={len(monte_carlo_summary)}."
            if require_targeted_ablations
            else "Targeted ablations not required by this readiness invocation."
        ),
        path=targeted_dir,
    )

    pdf_path = repo_root / "waste2energy-paper" / "main.pdf"
    pdf_passed = pdf_path.exists() and pdf_path.stat().st_size > 0
    _check(
        checks,
        name="manuscript_pdf",
        passed=(pdf_passed if require_pdf else True),
        detail=(
            "Manuscript PDF exists."
            if pdf_passed
            else "Manuscript PDF missing or empty."
        ),
        path=pdf_path,
    )

    failures = [check for check in checks if not check["passed"]]
    return {
        "ready": not failures,
        "checks": checks,
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check reviewer-remediation submission artifact readiness.")
    parser.add_argument("--repo-root", default=str(ROOT), help="Repository root containing outputs/ and waste2energy-paper/.")
    parser.add_argument("--require-bootstrap", action="store_true", help="Require non-empty bootstrap benchmark statistics.")
    parser.add_argument("--min-bootstrap-replicates", type=int, default=1)
    parser.add_argument("--require-targeted-ablations", action="store_true", help="Require targeted ablations and Monte Carlo UQ outputs.")
    parser.add_argument("--min-monte-carlo-replicates", type=int, default=1)
    parser.add_argument("--require-pdf", action="store_true", help="Require waste2energy-paper/main.pdf.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_readiness_report(
        repo_root=Path(args.repo_root),
        require_bootstrap=args.require_bootstrap,
        min_bootstrap_replicates=args.min_bootstrap_replicates,
        require_targeted_ablations=args.require_targeted_ablations,
        min_monte_carlo_replicates=args.min_monte_carlo_replicates,
        require_pdf=args.require_pdf,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
