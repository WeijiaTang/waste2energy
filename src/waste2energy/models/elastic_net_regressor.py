from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import RANDOM_STATE


@dataclass(frozen=True)
class ElasticNetConfig:
    alpha: float = 0.05
    l1_ratio: float = 0.5
    fit_intercept: bool = True
    max_iter: int = 20000
    tol: float = 1e-4
    selection: str = "cyclic"
    random_state: int = RANDOM_STATE


def build_model(config: ElasticNetConfig | None = None) -> Pipeline:
    active_config = config or ElasticNetConfig()
    estimator = ElasticNet(**asdict(active_config))
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("elasticnet", estimator),
        ]
    )


def train_model(
    x_frame: pd.DataFrame,
    y_series: pd.Series,
    sample_weight: pd.Series | None = None,
    config: ElasticNetConfig | None = None,
):
    model = build_model(config=config)
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["elasticnet__sample_weight"] = sample_weight
    model.fit(x_frame, y_series, **fit_kwargs)
    return model


def build_feature_importance(model: Pipeline, feature_columns: list[str]) -> pd.DataFrame:
    coefficients = model.named_steps["elasticnet"].coef_
    importance = pd.DataFrame(
        {
            "feature_name": feature_columns,
            "importance_gain_proxy": abs(coefficients),
            "coefficient": coefficients,
        }
    )
    return importance.sort_values("importance_gain_proxy", ascending=False).reset_index(drop=True)


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
