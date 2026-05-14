from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from waste2energy.models import MODEL_KEYS, get_model_ops
from waste2energy.models.stacking_regressor import StackingRegressorConfig, build_feature_importance, train_model


def test_stacking_is_registered_with_expected_artifact_metadata():
    assert "stacking" in MODEL_KEYS

    ops = get_model_ops("stacking")

    assert ops["model_key"] == "stacking"
    assert ops["model_family"] == "stacking_regressor"
    assert ops["model_file_name"] == "model.joblib"
    assert callable(ops["train_model"])
    assert callable(ops["build_feature_importance"])
    assert callable(ops["save_model"])


def test_stacking_can_fit_small_tabular_regression_problem():
    pytest.importorskip("xgboost")
    pytest.importorskip("catboost")

    rng = np.random.default_rng(0)
    x_frame = pd.DataFrame(rng.normal(size=(24, 4)), columns=["a", "b", "c", "d"])
    y_series = pd.Series(
        0.7 * x_frame["a"] - 0.2 * x_frame["b"] + 0.1 * x_frame["c"] + rng.normal(scale=0.05, size=24)
    )
    sample_weight = pd.Series(np.ones(len(x_frame)))

    model = train_model(
        x_frame,
        y_series,
        sample_weight=sample_weight,
        config=StackingRegressorConfig(cv=3),
    )

    predictions = model.predict(x_frame)
    importance = build_feature_importance(model, list(x_frame.columns))

    assert len(predictions) == len(x_frame)
    assert set(importance.columns) == {"feature_name", "importance_gain_proxy"}
    assert len(importance) == x_frame.shape[1]
