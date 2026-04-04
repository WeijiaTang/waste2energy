from __future__ import annotations

from math import sqrt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": sqrt(mse),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }
