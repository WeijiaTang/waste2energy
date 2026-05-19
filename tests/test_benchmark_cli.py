# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import sys

from waste2energy import benchmark_cli


def test_benchmark_cli_only_targeted_skips_benchmark_suite(tmp_path, monkeypatch, capsys):
    calls: list[tuple[str, object]] = []

    def fake_benchmark_suite(**kwargs):
        calls.append(("benchmark", kwargs))
        return {"variant_count": 0}

    def fake_targeted_ablations(**kwargs):
        calls.append(("targeted", kwargs))
        return {"summary_csv": str(tmp_path / "summary.csv")}

    monkeypatch.setattr(benchmark_cli, "run_planning_benchmark_suite", fake_benchmark_suite)
    monkeypatch.setattr(benchmark_cli, "run_targeted_planning_ablations", fake_targeted_ablations)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "waste2energy-benchmark",
            "--only-targeted-ablations",
            "--output-dir",
            str(tmp_path / "baseline"),
            "--targeted-monte-carlo-replicates",
            "4",
        ],
    )

    assert benchmark_cli.main() == 0
    assert [name for name, _ in calls] == ["targeted"]
    targeted_kwargs = calls[0][1]
    assert targeted_kwargs["monte_carlo_replicates"] == 4
    assert targeted_kwargs["output_dir"] == str(tmp_path / "baseline" / "targeted_planning_ablations")
    assert "targeted_ablations" in capsys.readouterr().out


def test_benchmark_cli_rejects_conflicting_targeted_flags(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["waste2energy-benchmark", "--skip-targeted-ablations", "--only-targeted-ablations"],
    )

    assert benchmark_cli.main() == 2
    assert "cannot be used together" in capsys.readouterr().err
