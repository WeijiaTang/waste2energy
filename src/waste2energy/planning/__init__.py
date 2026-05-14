"""Planning-layer utilities for Waste2Energy."""

from .constraints import build_scenario_constraints
from .inputs import PlanningInputBundle, load_planning_input_bundle
from .solve import PlanningConfig, run_planning_baseline


def run_planning_benchmark_suite(*args, **kwargs):
    from ..benchmarking import run_planning_benchmark_suite as _run_planning_benchmark_suite

    return _run_planning_benchmark_suite(*args, **kwargs)

__all__ = [
    "PlanningConfig",
    "PlanningInputBundle",
    "build_scenario_constraints",
    "load_planning_input_bundle",
    "run_planning_benchmark_suite",
    "run_planning_baseline",
]
