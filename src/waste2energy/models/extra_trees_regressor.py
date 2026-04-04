from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from ..config import RANDOM_STATE


@dataclass(frozen=True)
class ExtraTreesConfig:
    n_estimators: int = 500
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: float | str = "sqrt"
    bootstrap: bool = False
    random_state: int = RANDOM_STATE
    n_jobs: int = -1


def build_model(config: ExtraTreesConfig | None = None) -> ExtraTreesRegressor:
    active_config = config or ExtraTreesConfig()
    return ExtraTreesRegressor(**asdict(active_config))


def train_model(
    x_frame: pd.DataFrame,
    y_series: pd.Series,
    sample_weight: pd.Series | None = None,
    config: ExtraTreesConfig | None = None,
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
    joblib.dump(model, path)
