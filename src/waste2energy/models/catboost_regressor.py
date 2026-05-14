from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from ..config import RANDOM_STATE


@dataclass(frozen=True)
class CatBoostConfig:
    iterations: int = 500
    depth: int = 6
    learning_rate: float = 0.05
    l2_leaf_reg: float = 3.0
    subsample: float = 0.9
    loss_function: str = "RMSE"
    random_seed: int = RANDOM_STATE
    thread_count: int = -1
    verbose: bool = False
    allow_writing_files: bool = False


def build_model(config: CatBoostConfig | None = None):
    catboost = _import_catboost()
    active_config = config or CatBoostConfig()
    return catboost.CatBoostRegressor(**asdict(active_config))


def train_model(
    x_frame: pd.DataFrame,
    y_series: pd.Series,
    sample_weight: pd.Series | None = None,
    config: CatBoostConfig | None = None,
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
            "importance_gain_proxy": model.get_feature_importance(),
        }
    )
    return importance.sort_values("importance_gain_proxy", ascending=False).reset_index(drop=True)


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))


def _import_catboost():
    try:
        import catboost
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "catboost is not installed. Install project dependencies first, for example: "
            "`python -m pip install -e .` or `python -m pip install catboost`."
        ) from exc
    return catboost
