from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

from .catboost_regressor import CatBoostConfig, build_model as build_catboost_model
from .random_forest_regressor import RandomForestConfig, build_model as build_random_forest_model
from .xgboost_regressor import XGBoostConfig, build_model as build_xgboost_model


@dataclass(frozen=True)
class StackingRegressorConfig:
    cv: int = 3
    n_jobs: int = -1
    passthrough: bool = False
    final_alpha: float = 0.25
    final_fit_intercept: bool = True
    xgboost_config: XGBoostConfig = field(
        default_factory=lambda: XGBoostConfig(n_estimators=250, max_depth=4, learning_rate=0.05)
    )
    catboost_config: CatBoostConfig = field(
        default_factory=lambda: CatBoostConfig(iterations=300, depth=6, learning_rate=0.05)
    )
    rf_config: RandomForestConfig = field(
        default_factory=lambda: RandomForestConfig(n_estimators=300, max_features="sqrt")
    )


def build_model(config: StackingRegressorConfig | None = None) -> StackingRegressor:
    active_config = config or StackingRegressorConfig()
    final_estimator = Ridge(
        alpha=active_config.final_alpha,
        fit_intercept=active_config.final_fit_intercept,
    )
    return StackingRegressor(
        estimators=[
            ("xgboost", build_xgboost_model(active_config.xgboost_config)),
            ("catboost", build_catboost_model(active_config.catboost_config)),
            ("rf", build_random_forest_model(active_config.rf_config)),
        ],
        final_estimator=final_estimator,
        cv=active_config.cv,
        n_jobs=active_config.n_jobs,
        passthrough=active_config.passthrough,
    )


def train_model(
    x_frame: pd.DataFrame,
    y_series: pd.Series,
    sample_weight: pd.Series | None = None,
    config: StackingRegressorConfig | None = None,
):
    model = build_model(config=config)
    if sample_weight is None:
        model.fit(x_frame, y_series)
        return model

    try:
        model.fit(x_frame, y_series, sample_weight=sample_weight)
    except (TypeError, ValueError) as exc:
        if "sample_weight" not in str(exc):
            raise
        model.fit(x_frame, y_series)
    return model


def build_feature_importance(model: StackingRegressor, feature_columns: list[str]) -> pd.DataFrame:
    importance_vectors: list[np.ndarray] = []

    named_estimators = getattr(model, "named_estimators_", {})
    for estimator in named_estimators.values():
        vector = _extract_importance_vector(estimator, len(feature_columns))
        if vector is None:
            continue
        magnitude = np.abs(vector.astype(float))
        total = float(magnitude.sum())
        if total > 0:
            magnitude = magnitude / total
        importance_vectors.append(magnitude)

    aggregate = (
        np.mean(np.vstack(importance_vectors), axis=0)
        if importance_vectors
        else np.zeros(len(feature_columns), dtype=float)
    )
    importance = pd.DataFrame(
        {
            "feature_name": feature_columns,
            "importance_gain_proxy": aggregate,
        }
    )
    return importance.sort_values("importance_gain_proxy", ascending=False).reset_index(drop=True)


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def _extract_importance_vector(estimator, feature_count: int) -> np.ndarray | None:
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "get_feature_importance"):
        values = np.asarray(estimator.get_feature_importance(), dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.ravel(np.asarray(estimator.coef_, dtype=float))
    elif hasattr(estimator, "named_steps") and estimator.named_steps:
        last_step = next(reversed(estimator.named_steps.values()))
        return _extract_importance_vector(last_step, feature_count)
    else:
        return None

    if values.size != feature_count:
        return None
    return values
