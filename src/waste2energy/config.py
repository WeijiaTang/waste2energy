from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODEL_READY_DIR = ROOT / "data" / "processed" / "model_ready"
FIGURES_TABLES_DIR = ROOT / "data" / "processed" / "figures_tables"
OUTPUTS_ROOT = ROOT / "outputs"
SURROGATE_OUTPUTS_DIR = OUTPUTS_ROOT / "xgboost"
PLANNING_OUTPUTS_DIR = OUTPUTS_ROOT / "planning"
SCENARIO_OUTPUTS_DIR = OUTPUTS_ROOT / "scenarios"
OPERATION_OUTPUTS_DIR = OUTPUTS_ROOT / "operation"
OUTPUTS_DIR = SURROGATE_OUTPUTS_DIR
RANDOM_STATE = 42
