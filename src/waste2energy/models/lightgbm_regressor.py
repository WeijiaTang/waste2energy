from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from ..config import RANDOM_STATE


@dataclass(frozen=True)
class LightGBMConfig:
    boosting_type: str = "gbdt"
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 20
    min_child_weight: float = 1e-3
    subsample: float = 0.9
    subsample_freq: int = 1
    colsample_bytree: float = 0.9
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    objective: str = "regression"
    importance_type: str = "split"
    random_state: int = RANDOM_STATE
    n_jobs: int = -1
    verbose: int = -1


def build_model(config: LightGBMConfig | None = None):
    lightgbm = _import_lightgbm()
    active_config = config or LightGBMConfig()
    return lightgbm.LGBMRegressor(**asdict(active_config))


def train_model(
    x_frame: pd.DataFrame,
    y_series: pd.Series,
    sample_weight: pd.Series | None = None,
    config: LightGBMConfig | None = None,
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
    model.booster_.save_model(str(path))


def _import_lightgbm():
    try:
        import lightgbm
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "lightgbm is not installed. Install project dependencies first, for example: "
            "`python -m pip install -e .` or `python -m pip install lightgbm`."
        ) from exc
    return lightgbm
