"""Surrogate benchmarking utilities for Waste2Energy."""

from .evaluate import regression_metrics
from .train import (
    run_regression_baseline,
    run_regression_baseline_suite,
    run_xgboost_baseline,
    run_xgboost_baseline_suite,
)

__all__ = [
    "regression_metrics",
    "run_regression_baseline",
    "run_regression_baseline_suite",
    "run_xgboost_baseline",
    "run_xgboost_baseline_suite",
]
