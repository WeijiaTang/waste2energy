"""Model wrappers for Waste2Energy baselines."""

from __future__ import annotations

from .elastic_net_regressor import (
    ElasticNetConfig,
    build_feature_importance as build_elastic_net_feature_importance,
    save_model as save_elastic_net_model,
    train_model as train_elastic_net_model,
)
from .gradient_boosting_regressor import (
    GradientBoostingConfig,
    build_feature_importance as build_gradient_boosting_feature_importance,
    save_model as save_gradient_boosting_model,
    train_model as train_gradient_boosting_model,
)
from .extra_trees_regressor import (
    ExtraTreesConfig,
    build_feature_importance as build_extra_trees_feature_importance,
    save_model as save_extra_trees_model,
    train_model as train_extra_trees_model,
)
from .random_forest_regressor import (
    RandomForestConfig,
    build_feature_importance as build_random_forest_feature_importance,
    save_model as save_random_forest_model,
    train_model as train_random_forest_model,
)
from .xgboost_regressor import (
    XGBoostConfig,
    build_feature_importance as build_xgboost_feature_importance,
    save_model as save_xgboost_model,
    train_model as train_xgboost_model,
)


MODEL_KEYS = ("xgboost", "rf", "extra_trees", "elastic_net", "gradient_boosting")


def get_model_ops(model_key: str) -> dict[str, object]:
    if model_key == "xgboost":
        default_config = XGBoostConfig()
        return {
            "model_key": "xgboost",
            "model_family": "xgboost_regressor",
            "model_file_name": "model.json",
            "default_config": default_config,
            "train_model": train_xgboost_model,
            "build_feature_importance": build_xgboost_feature_importance,
            "save_model": save_xgboost_model,
        }
    if model_key == "rf":
        default_config = RandomForestConfig()
        return {
            "model_key": "rf",
            "model_family": "random_forest_regressor",
            "model_file_name": "model.joblib",
            "default_config": default_config,
            "train_model": train_random_forest_model,
            "build_feature_importance": build_random_forest_feature_importance,
            "save_model": save_random_forest_model,
        }
    if model_key == "extra_trees":
        default_config = ExtraTreesConfig()
        return {
            "model_key": "extra_trees",
            "model_family": "extra_trees_regressor",
            "model_file_name": "model.joblib",
            "default_config": default_config,
            "train_model": train_extra_trees_model,
            "build_feature_importance": build_extra_trees_feature_importance,
            "save_model": save_extra_trees_model,
        }
    if model_key == "elastic_net":
        default_config = ElasticNetConfig()
        return {
            "model_key": "elastic_net",
            "model_family": "elastic_net_regressor",
            "model_file_name": "model.joblib",
            "default_config": default_config,
            "train_model": train_elastic_net_model,
            "build_feature_importance": build_elastic_net_feature_importance,
            "save_model": save_elastic_net_model,
        }
    if model_key == "gradient_boosting":
        default_config = GradientBoostingConfig()
        return {
            "model_key": "gradient_boosting",
            "model_family": "gradient_boosting_regressor",
            "model_file_name": "model.joblib",
            "default_config": default_config,
            "train_model": train_gradient_boosting_model,
            "build_feature_importance": build_gradient_boosting_feature_importance,
            "save_model": save_gradient_boosting_model,
        }
    allowed = ", ".join(MODEL_KEYS)
    raise ValueError(f"Unsupported model '{model_key}'. Choose from: {allowed}")


__all__ = ["MODEL_KEYS", "get_model_ops"]
