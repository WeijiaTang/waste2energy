# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..config import resolve_surrogate_outputs_dir


SURROGATE_TARGETS = (
    "product_char_yield_pct",
    "product_char_hhv_mj_per_kg",
    "energy_recovery_pct",
    "carbon_retention_pct",
)

PATHWAY_DATASET_PREFERENCES: dict[str, tuple[str, ...]] = {
    "htc": ("paper1_htc_scope", "htc_direct"),
    "pyrolysis": ("pyrolysis_direct",),
}

SUPPORTED_SURROGATE_PATHWAYS = frozenset(PATHWAY_DATASET_PREFERENCES)


@dataclass(frozen=True)
class SurrogateArtifact:
    pathway: str
    target_column: str
    dataset_key: str
    model_key: str
    split_strategy: str
    model_path: Path
    run_config_path: Path
    metrics_path: Path
    feature_columns: tuple[str, ...]


class SurrogateEvaluator:
    def __init__(
        self,
        *,
        outputs_root: str | Path | None = None,
        preferred_split_strategy: str = "strict_group",
        fallback_split_strategy: str = "recommended",
        fallback_uncertainty_ratio: float = 0.10,
        allow_documented_fallback: bool = True,
    ) -> None:
        self.outputs_root = Path(outputs_root) if outputs_root else resolve_surrogate_outputs_dir()
        self.preferred_split_strategy = preferred_split_strategy
        self.fallback_split_strategy = fallback_split_strategy
        self.fallback_uncertainty_ratio = fallback_uncertainty_ratio
        self.allow_documented_fallback = allow_documented_fallback
        self._artifact_cache: dict[tuple[str, str], SurrogateArtifact | None] = {}
        self._model_cache: dict[Path, object] = {}

    def evaluate(self, frame: pd.DataFrame) -> pd.DataFrame:
        predictions = frame[["optimization_case_id", "pathway"]].copy()
        predictions["pathway"] = predictions["pathway"].astype(str).str.strip().str.lower()

        for target in SURROGATE_TARGETS:
            predictions[f"predicted_{target}"] = np.nan
            predictions[f"{target}_ci_lower"] = np.nan
            predictions[f"{target}_ci_upper"] = np.nan
            predictions[f"{target}_prediction_std"] = np.nan
            predictions[f"{target}_uncertainty_ratio"] = np.nan
            predictions[f"{target}_prediction_source"] = ""

        for pathway, pathway_frame in frame.groupby(frame["pathway"].astype(str).str.strip().str.lower(), dropna=False):
            pathway_index = pathway_frame.index
            if pathway in SUPPORTED_SURROGATE_PATHWAYS:
                for target in SURROGATE_TARGETS:
                    payload_frame = self._predict_target_batch(
                        pathway_frame,
                        pathway=pathway,
                        target_column=target,
                    )
                    for column in payload_frame.columns:
                        if column in {"optimization_case_id", "pathway"}:
                            continue
                        predictions.loc[pathway_index, column] = payload_frame[column].to_numpy()
            else:
                fallback_frame = self._build_documented_fallback_batch(pathway_frame, pathway=pathway)
                for column in fallback_frame.columns:
                    if column in {"optimization_case_id", "pathway"}:
                        continue
                    predictions.loc[pathway_index, column] = fallback_frame[column].to_numpy()

        return predictions

    def _predict_target_batch(
        self,
        frame: pd.DataFrame,
        *,
        pathway: str,
        target_column: str,
    ) -> pd.DataFrame:
        artifact = self._resolve_artifact(pathway=pathway, target_column=target_column)
        if artifact is None:
            if not self.allow_documented_fallback:
                raise FileNotFoundError(
                    f"No trained surrogate artifact found for pathway='{pathway}' target='{target_column}'."
                )
            return self._fallback_payload_batch(
                frame=frame,
                target_column=target_column,
                reason="missing_artifact",
            )

        model = self._load_model(artifact)
        prediction_std = self._estimate_prediction_std(artifact)
        feature_frame = pd.DataFrame(
            [self._materialize_features(row, artifact.feature_columns) for _, row in frame.iterrows()]
        )
        mean_values = np.asarray(model.predict(feature_frame), dtype=float)
        lower = mean_values - 1.96 * prediction_std
        upper = mean_values + 1.96 * prediction_std
        ratio = np.abs(upper - lower) / np.maximum(np.abs(mean_values), 1.0)
        return pd.DataFrame(
            {
                "optimization_case_id": frame["optimization_case_id"].to_numpy(),
                "pathway": frame["pathway"].astype(str).str.strip().str.lower().to_numpy(),
                f"predicted_{target_column}": mean_values,
                f"{target_column}_ci_lower": lower,
                f"{target_column}_ci_upper": upper,
                f"{target_column}_prediction_std": prediction_std,
                f"{target_column}_uncertainty_ratio": ratio,
                f"{target_column}_prediction_source": [
                    f"surrogate:{artifact.model_key}:{artifact.dataset_key}:{artifact.split_strategy}"
                ]
                * len(frame),
            }
        )

    def _resolve_artifact(self, *, pathway: str, target_column: str) -> SurrogateArtifact | None:
        cache_key = (pathway, target_column)
        if cache_key in self._artifact_cache:
            return self._artifact_cache[cache_key]

        summary_candidates = [
            self.outputs_root / "traditional_ml_suite_summary_strict_group.csv",
            self.outputs_root / self.preferred_split_strategy / "traditional_ml_suite_summary_strict_group.csv",
            self.outputs_root / "traditional_ml_suite_summary.csv",
            self.outputs_root / self.fallback_split_strategy / "traditional_ml_suite_summary.csv",
        ]
        datasets = PATHWAY_DATASET_PREFERENCES.get(pathway, ())
        selected: SurrogateArtifact | None = None

        for summary_path in summary_candidates:
            if not summary_path.exists():
                continue
            summary = pd.read_csv(summary_path)
            subset = summary[
                summary["dataset_key"].isin(datasets) & summary["target_column"].eq(target_column)
            ].copy()
            if subset.empty:
                continue
            subset["test_r2_rank"] = pd.to_numeric(subset.get("test_r2"), errors="coerce").fillna(-np.inf)
            best = subset.sort_values(["test_r2_rank", "model_key"], ascending=[False, True]).iloc[0]
            artifact = self._build_artifact_from_summary(best)
            if artifact is not None:
                selected = artifact
                break

        self._artifact_cache[cache_key] = selected
        return selected

    def _build_artifact_from_summary(self, row: pd.Series) -> SurrogateArtifact | None:
        model_key = str(row["model_key"])
        dataset_key = str(row["dataset_key"])
        target_column = str(row["target_column"])
        split_strategy = str(row.get("split_strategy", "recommended") or "recommended")

        if split_strategy == "recommended":
            base_dir = self.outputs_root
        else:
            base_dir = self.outputs_root / split_strategy

        if model_key == "xgboost":
            artifact_dir = base_dir / dataset_key / target_column
            model_path = artifact_dir / "model.json"
        else:
            artifact_dir = base_dir / model_key / dataset_key / target_column
            model_path = artifact_dir / "model.joblib"

        run_config_path = artifact_dir / "run_config.json"
        metrics_path = artifact_dir / "metrics.json"
        if not run_config_path.exists() or not model_path.exists():
            return None

        payload = json.loads(run_config_path.read_text(encoding="utf-8"))
        return SurrogateArtifact(
            pathway=self._infer_pathway_from_dataset(dataset_key),
            target_column=target_column,
            dataset_key=dataset_key,
            model_key=model_key,
            split_strategy=split_strategy,
            model_path=model_path,
            run_config_path=run_config_path,
            metrics_path=metrics_path,
            feature_columns=tuple(payload.get("feature_columns", [])),
        )

    def _load_model(self, artifact: SurrogateArtifact):
        if artifact.model_path in self._model_cache:
            return self._model_cache[artifact.model_path]

        if artifact.model_key == "xgboost":
            import xgboost as xgb

            model = xgb.XGBRegressor()
            model.load_model(artifact.model_path)
        else:
            model = joblib.load(artifact.model_path)
        self._model_cache[artifact.model_path] = model
        return model

    def _estimate_prediction_std(self, artifact: SurrogateArtifact) -> float:
        if artifact.metrics_path.exists():
            payload = json.loads(artifact.metrics_path.read_text(encoding="utf-8"))
            rmse_values = []
            for split_name in ("validation", "test", "train"):
                split_payload = payload.get(split_name, {})
                rmse = split_payload.get("rmse")
                if rmse is not None:
                    rmse_values.append(float(rmse))
            if rmse_values:
                return max(max(rmse_values), 1e-6)
        return self.fallback_uncertainty_ratio

    def _materialize_features(
        self,
        row: pd.Series,
        feature_columns: tuple[str, ...],
    ) -> dict[str, float]:
        record: dict[str, float] = {}
        manure_subtype = str(row.get("manure_subtype", "") or "").strip().lower()
        feedstock_group = str(row.get("feedstock_group", "") or "").strip().lower()
        row_origin = self._infer_row_origin(row)
        for feature_name in feature_columns:
            if feature_name in row.index:
                record[feature_name] = float(
                    pd.to_numeric(pd.Series([row[feature_name]]), errors="coerce").fillna(0.0).iloc[0]
                )
                continue
            if feature_name.startswith("feedstock_group_"):
                record[feature_name] = 1.0 if feedstock_group == feature_name.removeprefix("feedstock_group_") else 0.0
                continue
            if feature_name.startswith("manure_subtype_"):
                record[feature_name] = 1.0 if manure_subtype == feature_name.removeprefix("manure_subtype_") else 0.0
                continue
            if feature_name.startswith("row_origin_"):
                record[feature_name] = 1.0 if row_origin == feature_name.removeprefix("row_origin_") else 0.0
                continue
            record[feature_name] = 0.0
        return record

    def _infer_row_origin(self, row: pd.Series) -> str:
        source_kind = str(row.get("source_dataset_kind", "") or "").lower()
        if "observed" in source_kind:
            return "observed"
        if "synthetic" in source_kind or "candidate" in source_kind:
            return "synthetic"
        sample_id = str(row.get("sample_id", "") or "").lower()
        if "planning::" in sample_id:
            return "synthetic"
        return "observed"

    def _build_documented_fallback_batch(self, frame: pd.DataFrame, *, pathway: str) -> pd.DataFrame:
        payload = frame[["optimization_case_id", "pathway"]].copy()
        uncertainty_columns = []
        for target in SURROGATE_TARGETS:
            fallback = self._fallback_payload_batch(
                frame=frame,
                target_column=target,
                reason=f"static_direct:{pathway or 'unknown'}",
            )
            for column in fallback.columns:
                if column in {"optimization_case_id", "pathway"}:
                    continue
                payload[column] = fallback[column].to_numpy()
            uncertainty_columns.append(f"{target}_uncertainty_ratio")
        payload["combined_uncertainty_ratio"] = payload[uncertainty_columns].mean(axis=1).fillna(0.0)
        payload["surrogate_mode"] = "documented_static_fallback"
        return payload

    def _fallback_payload_batch(
        self,
        *,
        frame: pd.DataFrame,
        target_column: str,
        reason: str,
    ) -> pd.DataFrame:
        values = pd.to_numeric(frame.get(target_column), errors="coerce").fillna(0.0)
        std = np.maximum(np.abs(values.to_numpy()) * self.fallback_uncertainty_ratio, self.fallback_uncertainty_ratio)
        lower = values.to_numpy() - 1.96 * std
        upper = values.to_numpy() + 1.96 * std
        return pd.DataFrame(
            {
                "optimization_case_id": frame["optimization_case_id"].to_numpy(),
                "pathway": frame["pathway"].astype(str).str.strip().str.lower().to_numpy(),
                f"predicted_{target_column}": values.to_numpy(),
                f"{target_column}_ci_lower": lower,
                f"{target_column}_ci_upper": upper,
                f"{target_column}_prediction_std": std,
                f"{target_column}_uncertainty_ratio": np.abs(upper - lower) / np.maximum(np.abs(values.to_numpy()), 1.0),
                f"{target_column}_prediction_source": [reason] * len(frame),
            }
        )

    def _infer_pathway_from_dataset(self, dataset_key: str) -> str:
        if "pyrolysis" in dataset_key:
            return "pyrolysis"
        if "htc" in dataset_key:
            return "htc"
        return dataset_key


def build_surrogate_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    evaluator = SurrogateEvaluator()
    predictions = evaluator.evaluate(frame)
    uncertainty_columns = [f"{target}_uncertainty_ratio" for target in SURROGATE_TARGETS]
    predictions["combined_uncertainty_ratio"] = (
        predictions[[column for column in uncertainty_columns if column in predictions.columns]]
        .mean(axis=1)
        .fillna(0.0)
    )
    predictions["surrogate_mode"] = np.where(
        predictions["pathway"].isin(tuple(SUPPORTED_SURROGATE_PATHWAYS)),
        "trained_surrogate_or_fallback",
        "documented_static_fallback",
    )
    predictions = predictions.fillna(
        {
            f"{target}_prediction_source": "documented_static_fallback"
            for target in SURROGATE_TARGETS
        }
    )
    return predictions
