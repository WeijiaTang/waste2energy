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
        predictions["surrogate_prediction_status"] = ""
        predictions["surrogate_feature_imputation_flag"] = False
        predictions["surrogate_missing_feature_columns"] = ""
        predictions["surrogate_fallback_reason"] = ""

        for target in SURROGATE_TARGETS:
            predictions[f"predicted_{target}"] = np.nan
            predictions[f"{target}_ci_lower"] = np.nan
            predictions[f"{target}_ci_upper"] = np.nan
            predictions[f"{target}_prediction_std"] = np.nan
            predictions[f"{target}_uncertainty_ratio"] = np.nan
            predictions[f"{target}_prediction_source"] = ""

        grouped = frame.groupby(frame["pathway"].astype(str).str.strip().str.lower(), dropna=False)
        for pathway, pathway_frame in grouped:
            pathway_index = pathway_frame.index
            if pathway in SUPPORTED_SURROGATE_PATHWAYS:
                target_payloads: list[pd.DataFrame] = []
                for target in SURROGATE_TARGETS:
                    target_payloads.append(
                        self._predict_target_batch(
                            pathway_frame,
                            pathway=pathway,
                            target_column=target,
                        )
                    )
                merged_payload = target_payloads[0]
                for payload in target_payloads[1:]:
                    merge_columns = [
                        "surrogate_prediction_status",
                        "surrogate_feature_imputation_flag",
                        "surrogate_missing_feature_columns",
                        "surrogate_fallback_reason",
                    ]
                    payload = payload.drop(columns=[column for column in merge_columns if column in payload.columns])
                    merged_payload = merged_payload.merge(
                        payload,
                        on=["optimization_case_id", "pathway"],
                        how="left",
                        validate="one_to_one",
                    )
                for column in merged_payload.columns:
                    if column in {"optimization_case_id", "pathway"}:
                        continue
                    predictions.loc[pathway_index, column] = merged_payload[column].to_numpy()
            else:
                fallback_frame = self._build_documented_fallback_batch(
                    pathway_frame,
                    pathway=pathway,
                    reason=f"unsupported_pathway:{pathway or 'unknown'}",
                )
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
                prediction_status="documented_fallback_missing_artifact",
                missing_feature_columns=pd.Series([""] * len(frame), index=frame.index, dtype="object"),
                feature_imputation_flag=pd.Series([False] * len(frame), index=frame.index, dtype=bool),
            )

        materialized = self._materialize_features(frame, artifact.feature_columns)
        missing_feature_mask = materialized["missing_required_feature_columns"].astype(str).str.len().gt(0)
        if missing_feature_mask.any():
            if not self.allow_documented_fallback:
                details = materialized.loc[missing_feature_mask, "missing_required_feature_columns"].iloc[0]
                raise ValueError(
                    f"Missing required surrogate features for pathway='{pathway}' target='{target_column}': {details}"
                )
            return self._fallback_payload_batch(
                frame=frame,
                target_column=target_column,
                reason=f"missing_required_feature:{target_column}",
                prediction_status="documented_fallback_missing_required_feature",
                missing_feature_columns=materialized["missing_required_feature_columns"],
                feature_imputation_flag=materialized["feature_imputation_flag"],
            )

        model = self._load_model(artifact)
        prediction_std = self._estimate_prediction_std(artifact)
        mean_values = np.asarray(model.predict(materialized["feature_frame"]), dtype=float)
        lower = mean_values - 1.96 * prediction_std
        upper = mean_values + 1.96 * prediction_std
        ratio = np.abs(upper - lower) / np.maximum(np.abs(mean_values), 1.0)
        return pd.DataFrame(
            {
                "optimization_case_id": frame["optimization_case_id"].to_numpy(),
                "pathway": frame["pathway"].astype(str).str.strip().str.lower().to_numpy(),
                "surrogate_prediction_status": ["trained_surrogate_prediction"] * len(frame),
                "surrogate_feature_imputation_flag": materialized["feature_imputation_flag"].to_numpy(),
                "surrogate_missing_feature_columns": materialized["missing_required_feature_columns"].to_numpy(),
                "surrogate_fallback_reason": [""] * len(frame),
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

        selected_manifest_candidates = [
            self.outputs_root / f"selected_models_manifest_{self.preferred_split_strategy}.csv",
            self.outputs_root / "selected_models_manifest.csv",
            self.outputs_root / f"selected_models_manifest_{self.fallback_split_strategy}.csv",
        ]
        summary_candidates = [
            self.outputs_root / "traditional_ml_suite_summary_strict_group.csv",
            self.outputs_root / self.preferred_split_strategy / "traditional_ml_suite_summary_strict_group.csv",
            self.outputs_root / "traditional_ml_suite_summary.csv",
            self.outputs_root / self.fallback_split_strategy / "traditional_ml_suite_summary.csv",
        ]
        datasets = PATHWAY_DATASET_PREFERENCES.get(pathway, ())
        selected: SurrogateArtifact | None = None

        for manifest_path in selected_manifest_candidates:
            if not manifest_path.exists():
                continue
            manifest = pd.read_csv(manifest_path)
            subset = manifest[
                manifest["dataset_key"].isin(datasets) & manifest["target_column"].eq(target_column)
            ].copy()
            if subset.empty:
                continue
            subset = self._apply_dataset_preference_order(subset, datasets)
            if subset.empty:
                continue
            preferred_subset = subset[
                subset.get("artifact_role", pd.Series([""] * len(subset), index=subset.index)).astype(str)
                == "selected_model_refit"
            ]
            if preferred_subset.empty:
                preferred_subset = subset[
                    subset.get("selection_status", pd.Series([""] * len(subset), index=subset.index))
                    .astype(str)
                    .str.startswith("selected_on_validation")
                ]
            if not preferred_subset.empty:
                subset = preferred_subset
            best = subset.sort_values(["selected_validation_r2", "selected_model_key"], ascending=[False, True]).iloc[0]
            artifact = self._build_artifact_from_selected_manifest(best)
            if artifact is not None:
                selected = artifact
                break

        if selected is not None:
            self._artifact_cache[cache_key] = selected
            return selected

        for summary_path in summary_candidates:
            if not summary_path.exists():
                continue
            summary = pd.read_csv(summary_path)
            subset = summary[
                summary["dataset_key"].isin(datasets) & summary["target_column"].eq(target_column)
            ].copy()
            if subset.empty:
                continue
            subset = self._apply_dataset_preference_order(subset, datasets)
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

    def _build_artifact_from_selected_manifest(self, row: pd.Series) -> SurrogateArtifact | None:
        model_key = str(row.get("selected_model_key"))
        dataset_key = str(row.get("dataset_key"))
        target_column = str(row.get("target_column"))
        split_strategy = str(row.get("split_strategy", "recommended") or "recommended")
        model_path = Path(str(row.get("model_path", "")))
        run_config_path = Path(str(row.get("run_config_path", "")))
        metrics_path = Path(str(row.get("metrics_path", "")))

        if not model_path.exists() or not run_config_path.exists():
            payload = pd.Series(
                {
                    "model_key": model_key,
                    "dataset_key": dataset_key,
                    "target_column": target_column,
                    "split_strategy": split_strategy,
                }
            )
            return self._build_artifact_from_summary(payload)

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
        frame: pd.DataFrame,
        feature_columns: tuple[str, ...],
    ) -> dict[str, pd.DataFrame | pd.Series]:
        records: list[dict[str, float]] = []
        missing_columns: list[str] = []
        feature_imputation_flags: list[bool] = []

        for _, row in frame.iterrows():
            record: dict[str, float] = {}
            missing_for_row: list[str] = []
            manure_subtype = str(row.get("manure_subtype", "") or "").strip().lower()
            feedstock_group = str(row.get("feedstock_group", "") or "").strip().lower()
            row_origin = self._infer_row_origin(row)
            for feature_name in feature_columns:
                if feature_name in row.index:
                    numeric_value = pd.to_numeric(pd.Series([row[feature_name]]), errors="coerce").iloc[0]
                    if pd.isna(numeric_value):
                        missing_for_row.append(feature_name)
                        record[feature_name] = np.nan
                    else:
                        record[feature_name] = float(numeric_value)
                    continue
                if feature_name.startswith("feedstock_group_"):
                    record[feature_name] = (
                        1.0 if feedstock_group == feature_name.removeprefix("feedstock_group_") else 0.0
                    )
                    continue
                if feature_name.startswith("manure_subtype_"):
                    record[feature_name] = (
                        1.0 if manure_subtype == feature_name.removeprefix("manure_subtype_") else 0.0
                    )
                    continue
                if feature_name.startswith("row_origin_"):
                    record[feature_name] = 1.0 if row_origin == feature_name.removeprefix("row_origin_") else 0.0
                    continue
                missing_for_row.append(feature_name)
                record[feature_name] = np.nan
            records.append(record)
            missing_columns.append("|".join(sorted(set(missing_for_row))))
            feature_imputation_flags.append(bool(missing_for_row))

        return {
            "feature_frame": pd.DataFrame(records, index=frame.index),
            "missing_required_feature_columns": pd.Series(missing_columns, index=frame.index, dtype="object"),
            "feature_imputation_flag": pd.Series(feature_imputation_flags, index=frame.index, dtype=bool),
        }

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

    def _build_documented_fallback_batch(
        self,
        frame: pd.DataFrame,
        *,
        pathway: str,
        reason: str,
    ) -> pd.DataFrame:
        payload = frame[["optimization_case_id", "pathway"]].copy()
        payload["surrogate_prediction_status"] = "documented_static_fallback"
        payload["surrogate_feature_imputation_flag"] = False
        payload["surrogate_missing_feature_columns"] = ""
        payload["surrogate_fallback_reason"] = reason
        uncertainty_columns = []
        for target in SURROGATE_TARGETS:
            fallback = self._fallback_payload_batch(
                frame=frame,
                target_column=target,
                reason=reason,
                prediction_status="documented_static_fallback",
                missing_feature_columns=pd.Series([""] * len(frame), index=frame.index, dtype="object"),
                feature_imputation_flag=pd.Series([False] * len(frame), index=frame.index, dtype=bool),
            )
            for column in fallback.columns:
                if column in {"optimization_case_id", "pathway"}:
                    continue
                payload[column] = fallback[column].to_numpy()
            uncertainty_columns.append(f"{target}_uncertainty_ratio")
        payload["combined_uncertainty_ratio"] = payload[uncertainty_columns].mean(axis=1, skipna=True)
        payload["surrogate_mode"] = "documented_static_fallback"
        return payload

    def _fallback_payload_batch(
        self,
        *,
        frame: pd.DataFrame,
        target_column: str,
        reason: str,
        prediction_status: str,
        missing_feature_columns: pd.Series,
        feature_imputation_flag: pd.Series,
    ) -> pd.DataFrame:
        if target_column in frame.columns:
            values = pd.to_numeric(frame[target_column], errors="coerce")
        else:
            values = pd.Series(np.nan, index=frame.index, dtype=float)
        value_array = values.to_numpy(dtype=float, na_value=np.nan)
        std = np.where(
            np.isnan(value_array),
            np.nan,
            np.maximum(np.abs(value_array) * self.fallback_uncertainty_ratio, self.fallback_uncertainty_ratio),
        )
        lower = value_array - 1.96 * std
        upper = value_array + 1.96 * std
        ratio = np.where(
            np.isnan(value_array),
            np.nan,
            np.abs(upper - lower) / np.maximum(np.abs(value_array), 1.0),
        )
        status = np.where(
            values.isna(),
            "documented_fallback_missing_target_value",
            prediction_status,
        )
        fallback_reason = np.where(values.isna(), f"{reason}|missing_target_value", reason)
        prediction_source = np.where(
            values.isna(),
            f"{reason}|target_missing",
            reason,
        )
        return pd.DataFrame(
            {
                "optimization_case_id": frame["optimization_case_id"].to_numpy(),
                "pathway": frame["pathway"].astype(str).str.strip().str.lower().to_numpy(),
                "surrogate_prediction_status": status,
                "surrogate_feature_imputation_flag": feature_imputation_flag.to_numpy(),
                "surrogate_missing_feature_columns": missing_feature_columns.to_numpy(),
                "surrogate_fallback_reason": fallback_reason,
                f"predicted_{target_column}": values.to_numpy(),
                f"{target_column}_ci_lower": lower,
                f"{target_column}_ci_upper": upper,
                f"{target_column}_prediction_std": std,
                f"{target_column}_uncertainty_ratio": ratio,
                f"{target_column}_prediction_source": prediction_source,
            }
        )

    def _infer_pathway_from_dataset(self, dataset_key: str) -> str:
        if "pyrolysis" in dataset_key:
            return "pyrolysis"
        if "htc" in dataset_key:
            return "htc"
        return dataset_key

    def _apply_dataset_preference_order(
        self,
        frame: pd.DataFrame,
        datasets: tuple[str, ...],
    ) -> pd.DataFrame:
        if frame.empty or not datasets:
            return frame
        ordered = frame.copy()
        for preferred_dataset in datasets:
            subset = ordered[ordered["dataset_key"].astype(str) == preferred_dataset].copy()
            if not subset.empty:
                return subset
        return pd.DataFrame(columns=frame.columns)


def build_surrogate_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    evaluator = SurrogateEvaluator()
    predictions = evaluator.evaluate(frame)
    uncertainty_columns = [f"{target}_uncertainty_ratio" for target in SURROGATE_TARGETS]
    predictions["combined_uncertainty_ratio"] = predictions[
        [column for column in uncertainty_columns if column in predictions.columns]
    ].mean(axis=1, skipna=True)
    predictions["surrogate_mode"] = np.where(
        predictions["pathway"].isin(tuple(SUPPORTED_SURROGATE_PATHWAYS)),
        np.where(
            predictions["surrogate_prediction_status"].astype(str).str.startswith("trained_surrogate"),
            "trained_surrogate",
            "trained_surrogate_with_documented_fallback",
        ),
        "documented_static_fallback",
    )
    predictions = predictions.fillna(
        {
            f"{target}_prediction_source": "documented_static_fallback"
            for target in SURROGATE_TARGETS
        }
    )
    return predictions
