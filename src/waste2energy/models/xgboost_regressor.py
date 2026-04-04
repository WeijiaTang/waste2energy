from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from ..config import RANDOM_STATE


@dataclass(frozen=True)
class XGBoostConfig:
    n_estimators: int = 400
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    min_child_weight: float = 1.0
    objective: str = "reg:squarederror"
    tree_method: str = "hist"
    random_state: int = RANDOM_STATE
    n_jobs: int = -1


def build_model(config: XGBoostConfig | None = None):
    xgb = _import_xgboost()
    active_config = config or XGBoostConfig()
    return xgb.XGBRegressor(**asdict(active_config))


def train_model(
    x_frame: pd.DataFrame,
    y_series: pd.Series,
    sample_weight: pd.Series | None = None,
    config: XGBoostConfig | None = None,
):
    model = build_model(config=config)
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(x_frame, y_series, **fit_kwargs)
    return model


def build_feature_importance(model, feature_columns: list[str]) -> pd.DataFrame:
    importance = pd.DataFrame(
        {
            "feature_name": feature_columns,
            "importance_gain_proxy": model.feature_importances_,
        }
    )
    return importance.sort_values("importance_gain_proxy", ascending=False).reset_index(drop=True)


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)


def _import_xgboost():
    try:
        import xgboost as xgb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xgboost is not installed. Install project dependencies first, for example: "
            "`python -m pip install -e .` or `python -m pip install xgboost`."
        ) from exc
    return xgb
