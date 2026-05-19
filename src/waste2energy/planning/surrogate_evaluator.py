# Ref: docs/spec/task.md (Task-ID: WTE-SPEC-2026-04-07-PLANNING-REFINE)

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..config import BENCHMARK_OUTPUTS_DIR, resolve_surrogate_outputs_dir


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

PATHWAY_MODEL_PRIORITIES: dict[str, tuple[str, ...]] = {
    "htc": (
        "catboost",
        "lightgbm",
        "stacking",
        "xgboost",
        "extra_trees",
        "rf",
        "gradient_boosting",
        "elastic_net",
    ),
}

SUPPORTED_SURROGATE_PATHWAYS = frozenset(PATHWAY_DATASET_PREFERENCES)

MODEL_LOAD_FALLBACK_EXCEPTIONS = (ImportError, ModuleNotFoundError, FileNotFoundError, OSError, ValueError)


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
    calibration_predictions_path: Path | None
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
        pathway_model_priorities: dict[str, tuple[str, ...]] | None = None,
        minimum_artifact_test_r2: float | None = None,
    ) -> None:
        self.outputs_root = Path(outputs_root) if outputs_root else resolve_surrogate_outputs_dir()
        self.preferred_split_strategy = preferred_split_strategy
        self.fallback_split_strategy = fallback_split_strategy
        self.fallback_uncertainty_ratio = fallback_uncertainty_ratio
        self.allow_documented_fallback = allow_documented_fallback
        self.minimum_artifact_test_r2 = minimum_artifact_test_r2
        self.pathway_model_priorities = {
            key: tuple(value) for key, value in (pathway_model_priorities or PATHWAY_MODEL_PRIORITIES).items()
        }
        self._artifact_cache: dict[tuple[str, str], SurrogateArtifact | None] = {}
        self._model_cache: dict[Path, object] = {}

    def evaluate(self, frame: pd.DataFrame) -> pd.DataFrame:
        predictions = frame[["optimization_case_id", "pathway"]].copy()
        predictions["pathway"] = predictions["pathway"].astype(str).str.strip().str.lower()
        predictions["surrogate_prediction_status"] = ""
        predictions["surrogate_feature_imputation_flag"] = False
        predictions["surrogate_missing_feature_columns"] = ""
        predictions["surrogate_imputed_feature_columns"] = ""
        predictions["surrogate_fallback_reason"] = ""

        for target in SURROGATE_TARGETS:
            predictions[f"predicted_{target}"] = np.nan
            predictions[f"{target}_ci_lower"] = np.nan
            predictions[f"{target}_ci_upper"] = np.nan
            predictions[f"{target}_prediction_std"] = np.nan
            predictions[f"{target}_uncertainty_ratio"] = np.nan
            predictions[f"{target}_uncertainty_method"] = ""
            predictions[f"{target}_uncertainty_calibration_count"] = np.nan
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
                diagnostic_columns = [
                    "surrogate_prediction_status",
                    "surrogate_feature_imputation_flag",
                    "surrogate_missing_feature_columns",
                    "surrogate_imputed_feature_columns",
                    "surrogate_fallback_reason",
                ]
                for payload in target_payloads[1:]:
                    payload = payload.drop(columns=[column for column in diagnostic_columns if column in payload.columns])
                    merged_payload = merged_payload.merge(
                        payload,
                        on=["optimization_case_id", "pathway"],
                        how="left",
                        validate="one_to_one",
                    )
                diagnostics = self._aggregate_target_diagnostics(target_payloads)
                merged_payload = merged_payload.drop(
                    columns=[column for column in diagnostic_columns if column in merged_payload.columns]
                ).merge(
                    diagnostics,
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
                imputed_feature_columns=materialized["imputed_feature_columns"],
            )

        try:
            model = self._load_model(artifact)
            mean_values = np.asarray(model.predict(materialized["feature_frame"]), dtype=float)
            half_width, prediction_std, uncertainty_method, calibration_count = (
                self._estimate_prediction_interval(artifact)
            )
            lower = mean_values - half_width
            upper = mean_values + half_width
            ratio = np.abs(upper - lower) / np.maximum(np.abs(mean_values), 1.0)
            return pd.DataFrame(
                {
                    "optimization_case_id": frame["optimization_case_id"].to_numpy(),
                    "pathway": frame["pathway"].astype(str).str.strip().str.lower().to_numpy(),
                    "surrogate_prediction_status": ["trained_surrogate_prediction"] * len(frame),
                    "surrogate_feature_imputation_flag": materialized["feature_imputation_flag"].to_numpy(),
                    "surrogate_missing_feature_columns": materialized["missing_required_feature_columns"].to_numpy(),
                    "surrogate_imputed_feature_columns": materialized["imputed_feature_columns"].to_numpy(),
                    "surrogate_fallback_reason": [""] * len(frame),
                    f"predicted_{target_column}": mean_values,
                    f"{target_column}_ci_lower": lower,
                    f"{target_column}_ci_upper": upper,
                    f"{target_column}_prediction_std": prediction_std,
                    f"{target_column}_uncertainty_ratio": ratio,
                    f"{target_column}_uncertainty_method": [uncertainty_method] * len(frame),
                    f"{target_column}_uncertainty_calibration_count": [calibration_count] * len(frame),
                    f"{target_column}_prediction_source": [
                        f"surrogate:{artifact.model_key}:{artifact.dataset_key}:{artifact.split_strategy}"
                    ]
                    * len(frame),
                }
            )
        except MODEL_LOAD_FALLBACK_EXCEPTIONS as exc:
            if not self.allow_documented_fallback:
                raise
            reason = (
                f"model_load_failure:{artifact.model_key}:{artifact.dataset_key}:"
                f"{exc.__class__.__name__}"
            )
            return self._fallback_payload_batch(
                frame=frame,
                target_column=target_column,
                reason=reason,
                prediction_status="documented_fallback_model_load_failure",
                missing_feature_columns=materialized["missing_required_feature_columns"],
                feature_imputation_flag=materialized["feature_imputation_flag"],
                imputed_feature_columns=materialized["imputed_feature_columns"],
            )

    def _aggregate_target_diagnostics(self, payloads: list[pd.DataFrame]) -> pd.DataFrame:
        base = payloads[0][["optimization_case_id", "pathway"]].copy()
        rows: list[dict[str, object]] = []
        for row_position in range(len(base)):
            statuses = [
                str(payload.iloc[row_position].get("surrogate_prediction_status", "") or "")
                for payload in payloads
            ]
            trained_count = sum(status == "trained_surrogate_prediction" for status in statuses)
            if trained_count == len(statuses):
                aggregate_status = "trained_surrogate_prediction"
            elif trained_count > 0 or any(status == "trained_surrogate_with_documented_fallback" for status in statuses):
                aggregate_status = "trained_surrogate_with_documented_fallback"
            else:
                aggregate_status = next((status for status in statuses if status), "documented_static_fallback")

            rows.append(
                {
                    "optimization_case_id": base.iloc[row_position]["optimization_case_id"],
                    "pathway": base.iloc[row_position]["pathway"],
                    "surrogate_prediction_status": aggregate_status,
                    "surrogate_feature_imputation_flag": any(
                        bool(payload.iloc[row_position].get("surrogate_feature_imputation_flag", False))
                        for payload in payloads
                    ),
                    "surrogate_missing_feature_columns": self._join_diagnostic_values(
                        payload.iloc[row_position].get("surrogate_missing_feature_columns", "")
                        for payload in payloads
                    ),
                    "surrogate_imputed_feature_columns": self._join_diagnostic_values(
                        payload.iloc[row_position].get("surrogate_imputed_feature_columns", "")
                        for payload in payloads
                    ),
                    "surrogate_fallback_reason": self._join_diagnostic_values(
                        payload.iloc[row_position].get("surrogate_fallback_reason", "")
                        for payload in payloads
                    ),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _join_diagnostic_values(values: object) -> str:
        parts: list[str] = []
        for value in values:
            for part in str(value or "").split("|"):
                cleaned = part.strip()
                if cleaned and cleaned.lower() != "nan" and cleaned not in parts:
                    parts.append(cleaned)
        return "|".join(parts)

    def _resolve_artifact(self, *, pathway: str, target_column: str) -> SurrogateArtifact | None:
        cache_key = (pathway, target_column)
        if cache_key in self._artifact_cache:
            return self._artifact_cache[cache_key]

        datasets = PATHWAY_DATASET_PREFERENCES.get(pathway, ())
        selected: SurrogateArtifact | None = None

        for source_root, manifest_path in self._selected_manifest_candidates(pathway):
            if not manifest_path.exists():
                continue
            manifest = pd.read_csv(manifest_path)
            subset = manifest[
                manifest["dataset_key"].isin(datasets) & manifest["target_column"].eq(target_column)
            ].copy()
            if subset.empty:
                continue
            subset = self._rank_candidate_rows(
                subset,
                pathway=pathway,
                datasets=datasets,
                metric_columns=("selected_validation_r2", "selected_test_r2"),
            )
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
            best = subset.iloc[0]
            artifact = self._build_artifact_from_selected_manifest(best, source_root=source_root)
            if artifact is not None:
                selected = artifact
                break

        if selected is not None:
            self._artifact_cache[cache_key] = selected
            return selected

        for source_root, summary_path in self._summary_candidates(pathway):
            if not summary_path.exists():
                continue
            summary = pd.read_csv(summary_path)
            subset = summary[
                summary["dataset_key"].isin(datasets) & summary["target_column"].eq(target_column)
            ].copy()
            if subset.empty:
                continue
            subset = self._rank_candidate_rows(
                subset,
                pathway=pathway,
                datasets=datasets,
                metric_columns=("validation_r2", "test_r2"),
            )
            if subset.empty:
                continue
            best = subset.iloc[0]
            artifact = self._build_artifact_from_summary(best, source_root=source_root)
            if artifact is not None:
                selected = artifact
                break

        self._artifact_cache[cache_key] = selected
        return selected

    def _build_artifact_from_selected_manifest(self, row: pd.Series, *, source_root: Path) -> SurrogateArtifact | None:
        model_key = str(row.get("selected_model_key"))
        dataset_key = str(row.get("dataset_key"))
        target_column = str(row.get("target_column"))
        split_strategy = str(row.get("split_strategy", "recommended") or "recommended")
        model_path = self._resolve_existing_path(source_root, row.get("model_path", ""))
        run_config_path = self._resolve_existing_path(source_root, row.get("run_config_path", ""))
        metrics_path = self._resolve_existing_path(source_root, row.get("metrics_path", ""))

        if not model_path.exists() or not run_config_path.exists():
            payload = pd.Series(
                {
                    "model_key": model_key,
                    "dataset_key": dataset_key,
                    "target_column": target_column,
                    "split_strategy": split_strategy,
                }
            )
            return self._build_artifact_from_summary(payload, source_root=source_root)

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
            calibration_predictions_path=self._resolve_optional_existing_path(
                source_root,
                row.get("benchmark_predictions_path"),
                row.get("predictions_path"),
            ),
            feature_columns=tuple(payload.get("feature_columns", [])),
        )

    def _build_artifact_from_summary(self, row: pd.Series, *, source_root: Path) -> SurrogateArtifact | None:
        model_key = str(row["model_key"])
        dataset_key = str(row["dataset_key"])
        target_column = str(row["target_column"])
        split_strategy = str(row.get("split_strategy", "recommended") or "recommended")

        artifact_dir, model_path = self._artifact_location_from_root(
            source_root=source_root,
            split_strategy=split_strategy,
            model_key=model_key,
            dataset_key=dataset_key,
            target_column=target_column,
        )

        run_config_path = artifact_dir / "run_config.json"
        metrics_path = artifact_dir / "metrics.json"
        predictions_path = artifact_dir / "predictions.csv"
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
            calibration_predictions_path=predictions_path if predictions_path.exists() else None,
            feature_columns=tuple(payload.get("feature_columns", [])),
        )

    def _load_model(self, artifact: SurrogateArtifact):
        if artifact.model_path in self._model_cache:
            return self._model_cache[artifact.model_path]

        if artifact.model_key == "xgboost":
            import xgboost as xgb

            model = xgb.XGBRegressor()
            model.load_model(artifact.model_path)
        elif artifact.model_key == "catboost":
            import catboost

            model = catboost.CatBoostRegressor()
            model.load_model(str(artifact.model_path))
        elif artifact.model_key == "lightgbm":
            import lightgbm as lgb

            model = lgb.Booster(model_file=str(artifact.model_path))
        else:
            model = joblib.load(artifact.model_path)
        self._model_cache[artifact.model_path] = model
        return model

    def _artifact_location_from_root(
        self,
        *,
        source_root: Path,
        split_strategy: str,
        model_key: str,
        dataset_key: str,
        target_column: str,
    ) -> tuple[Path, Path]:
        candidate_roots = [source_root]
        if split_strategy != "recommended":
            candidate_roots.insert(0, source_root / split_strategy)
        for root in candidate_roots:
            artifact_dir, model_path = _artifact_location_for_model(
                root=root,
                model_key=model_key,
                dataset_key=dataset_key,
                target_column=target_column,
            )
            if model_path.exists() or (artifact_dir / "run_config.json").exists():
                return artifact_dir, model_path
        return _artifact_location_for_model(
            root=candidate_roots[0],
            model_key=model_key,
            dataset_key=dataset_key,
            target_column=target_column,
        )

    def _selected_manifest_candidates(self, pathway: str) -> list[tuple[Path, Path]]:
        candidates: list[tuple[Path, Path]] = []
        for root in self._candidate_source_roots(pathway):
            if pathway == "htc":
                candidates.extend(
                    [
                        (root, root / "selected_models_manifest_benchmark_leave_study_out.csv"),
                        (root, root / "selected_models_manifest_leave_study_out.csv"),
                    ]
                )
            candidates.extend(
                [
                    (root, root / f"selected_models_manifest_{self.preferred_split_strategy}.csv"),
                    (root, root / "selected_models_manifest.csv"),
                    (root, root / f"selected_models_manifest_{self.fallback_split_strategy}.csv"),
                ]
            )
        return _deduplicate_candidate_paths(candidates)

    def _summary_candidates(self, pathway: str) -> list[tuple[Path, Path]]:
        candidates: list[tuple[Path, Path]] = []
        for root in self._candidate_source_roots(pathway):
            if pathway == "htc":
                candidates.append((root, root / "traditional_ml_suite_summary_leave_study_out.csv"))
            candidates.extend(
                [
                    (root, root / f"traditional_ml_suite_summary_{self.preferred_split_strategy}.csv"),
                    (root, root / self.preferred_split_strategy / f"traditional_ml_suite_summary_{self.preferred_split_strategy}.csv"),
                    (root, root / "traditional_ml_suite_summary.csv"),
                    (root, root / self.fallback_split_strategy / "traditional_ml_suite_summary.csv"),
                ]
            )
        return _deduplicate_candidate_paths(candidates)

    def _candidate_source_roots(self, pathway: str) -> list[Path]:
        roots: list[Path] = []
        if pathway == "htc":
            roots.append(BENCHMARK_OUTPUTS_DIR / "htc_model_compare_lso")
        roots.append(self.outputs_root)
        return roots

    def _rank_candidate_rows(
        self,
        subset: pd.DataFrame,
        *,
        pathway: str,
        datasets: tuple[str, ...],
        metric_columns: tuple[str, ...],
    ) -> pd.DataFrame:
        working = self._apply_dataset_preference_order(subset, datasets)
        if working.empty:
            return working
        working = working.copy()
        model_column = "selected_model_key" if "selected_model_key" in working.columns else "model_key"
        working = working[
            working[model_column].astype(str).map(self._model_runtime_available).fillna(False)
        ].copy()
        if working.empty:
            return working
        working = self._apply_minimum_test_r2_gate(working)
        if working.empty:
            return working
        working["_model_priority_rank"] = working[model_column].astype(str).map(
            lambda value: self._model_priority_rank(pathway=pathway, model_key=value)
        )
        sort_columns = ["_model_priority_rank"]
        ascending = [True]
        for metric_column in metric_columns:
            if metric_column in working.columns:
                metric_values = pd.to_numeric(working[metric_column], errors="coerce")
            else:
                metric_values = pd.Series(np.nan, index=working.index, dtype=float)
            working[f"_{metric_column}_sort"] = metric_values.fillna(-np.inf)
            sort_columns.append(f"_{metric_column}_sort")
            ascending.append(False)
        sort_columns.append(model_column)
        ascending.append(True)
        return working.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)

    def _apply_minimum_test_r2_gate(self, working: pd.DataFrame) -> pd.DataFrame:
        if self.minimum_artifact_test_r2 is None:
            return working
        test_metric_column = None
        for candidate in ("selected_test_r2", "test_r2", "reporting_test_r2"):
            if candidate in working.columns:
                test_metric_column = candidate
                break
        if test_metric_column is None:
            return working
        test_r2 = pd.to_numeric(working[test_metric_column], errors="coerce")
        return working[test_r2.ge(float(self.minimum_artifact_test_r2)).fillna(False)].copy()

    def _model_priority_rank(self, *, pathway: str, model_key: str) -> int:
        priorities = self.pathway_model_priorities.get(pathway, ())
        try:
            return priorities.index(model_key)
        except ValueError:
            return len(priorities)

    def _model_runtime_available(self, model_key: str) -> bool:
        try:
            if model_key == "xgboost":
                import xgboost  # noqa: F401
            elif model_key == "catboost":
                import catboost  # noqa: F401
            elif model_key == "lightgbm":
                import lightgbm  # noqa: F401
            return True
        except Exception:
            return model_key in {"stacking", "rf", "extra_trees", "gradient_boosting", "elastic_net"}

    def _resolve_existing_path(self, source_root: Path, raw_path: object) -> Path:
        path = Path(str(raw_path or ""))
        if path.exists():
            return path
        if path.is_absolute():
            return path
        candidate = source_root / path
        if candidate.exists():
            return candidate
        return path

    def _resolve_optional_existing_path(self, source_root: Path, *raw_paths: object) -> Path | None:
        for raw_path in raw_paths:
            if raw_path is None:
                continue
            path = self._resolve_existing_path(source_root, raw_path)
            if path.exists():
                return path
        return None

    def _estimate_prediction_interval(
        self,
        artifact: SurrogateArtifact,
    ) -> tuple[float, float, str, int]:
        if artifact.calibration_predictions_path and artifact.calibration_predictions_path.exists():
            calibration = self._read_calibration_residuals(artifact.calibration_predictions_path)
            if calibration is not None:
                half_width, count = calibration
                half_width = max(float(half_width), 1e-6)
                return half_width, half_width / 1.96, "split_conformal_abs_residual", count

        prediction_std = max(float(self._estimate_prediction_std(artifact)), 1e-6)
        return 1.96 * prediction_std, prediction_std, "rmse_proxy", 0

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

    def _read_calibration_residuals(self, predictions_path: Path) -> tuple[float, int] | None:
        try:
            predictions = pd.read_csv(predictions_path)
        except (OSError, pd.errors.EmptyDataError):
            return None

        if predictions.empty or not {"y_true", "y_pred"}.issubset(predictions.columns):
            return None

        working = predictions.copy()
        if "split" in working.columns:
            for split_name in ("validation", "test", "train", "refit_train"):
                selected = working[working["split"].astype(str) == split_name].copy()
                if not selected.empty:
                    working = selected
                    break

        residuals = (
            pd.to_numeric(working["y_true"], errors="coerce")
            - pd.to_numeric(working["y_pred"], errors="coerce")
        ).abs().dropna()
        if residuals.empty:
            return None

        values = np.sort(residuals.to_numpy(dtype=float))
        count = int(len(values))
        quantile_rank = int(np.ceil((count + 1) * 0.95)) - 1
        quantile_rank = min(max(quantile_rank, 0), count - 1)
        return float(values[quantile_rank]), count

    def _materialize_features(
        self,
        frame: pd.DataFrame,
        feature_columns: tuple[str, ...],
    ) -> dict[str, pd.DataFrame | pd.Series]:
        records: list[dict[str, float]] = []
        missing_columns: list[str] = []
        imputed_columns: list[str] = []
        feature_imputation_flags: list[bool] = []

        for _, row in frame.iterrows():
            record: dict[str, float] = {}
            missing_for_row: list[str] = []
            imputed_for_row: list[str] = []
            manure_subtype = str(row.get("manure_subtype", "") or "").strip().lower()
            feedstock_group = str(row.get("feedstock_group", "") or "").strip().lower()
            row_origin = self._infer_row_origin(row)
            for feature_name in feature_columns:
                if feature_name in row.index:
                    numeric_value = pd.to_numeric(pd.Series([row[feature_name]]), errors="coerce").iloc[0]
                    if pd.isna(numeric_value):
                        derived_value = self._derive_feature_value(row, feature_name)
                        if derived_value is None:
                            missing_for_row.append(feature_name)
                            record[feature_name] = np.nan
                        else:
                            record[feature_name] = float(derived_value)
                            imputed_for_row.append(feature_name)
                    else:
                        record[feature_name] = float(numeric_value)
                    continue
                derived_value = self._derive_feature_value(row, feature_name)
                if derived_value is not None:
                    record[feature_name] = float(derived_value)
                    imputed_for_row.append(feature_name)
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
            imputed_columns.append("|".join(sorted(set(imputed_for_row))))
            feature_imputation_flags.append(bool(missing_for_row or imputed_for_row))

        return {
            "feature_frame": pd.DataFrame(records, index=frame.index),
            "missing_required_feature_columns": pd.Series(missing_columns, index=frame.index, dtype="object"),
            "imputed_feature_columns": pd.Series(imputed_columns, index=frame.index, dtype="object"),
            "feature_imputation_flag": pd.Series(feature_imputation_flags, index=frame.index, dtype=bool),
        }

    def _derive_feature_value(self, row: pd.Series, feature_name: str) -> float | None:
        if feature_name != "feedstock_hhv_mj_per_kg":
            return None
        required = {
            "feedstock_carbon_pct": "carbon",
            "feedstock_hydrogen_pct": "hydrogen",
            "feedstock_nitrogen_pct": "nitrogen",
            "feedstock_oxygen_pct": "oxygen",
            "feedstock_ash_pct": "ash",
        }
        values: dict[str, float] = {}
        for column, key in required.items():
            if column not in row.index:
                return None
            value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if pd.isna(value):
                return None
            values[key] = float(value)
        # Channiwala-Parikh/Dulong-style HHV estimate from ultimate analysis
        # mass percentages. Sulfur is unavailable in the planning input and is
        # therefore treated as zero. The derived value is used only to
        # materialize a trained-surrogate feature and is flagged as imputed.
        hhv = (
            0.3491 * values["carbon"]
            + 1.1783 * values["hydrogen"]
            - 0.1034 * values["oxygen"]
            - 0.0151 * values["nitrogen"]
            - 0.0211 * values["ash"]
        )
        return float(hhv) if np.isfinite(hhv) and hhv > 0.0 else None

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
        payload["surrogate_imputed_feature_columns"] = ""
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
        imputed_feature_columns: pd.Series | None = None,
    ) -> pd.DataFrame:
        if imputed_feature_columns is None:
            imputed_feature_columns = pd.Series([""] * len(frame), index=frame.index, dtype="object")
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
                "surrogate_imputed_feature_columns": imputed_feature_columns.to_numpy(),
                "surrogate_fallback_reason": fallback_reason,
                f"predicted_{target_column}": values.to_numpy(),
                f"{target_column}_ci_lower": lower,
                f"{target_column}_ci_upper": upper,
                f"{target_column}_prediction_std": std,
                f"{target_column}_uncertainty_ratio": ratio,
                f"{target_column}_uncertainty_method": np.where(
                    values.isna(),
                    "",
                    "fallback_ratio_proxy",
                ),
                f"{target_column}_uncertainty_calibration_count": np.where(
                    values.isna(),
                    np.nan,
                    0,
                ),
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


def build_surrogate_predictions(
    frame: pd.DataFrame,
    *,
    pathway_model_priorities: dict[str, tuple[str, ...]] | None = None,
    minimum_artifact_test_r2: float | None = None,
) -> pd.DataFrame:
    evaluator = SurrogateEvaluator(
        pathway_model_priorities=pathway_model_priorities,
        minimum_artifact_test_r2=minimum_artifact_test_r2,
    )
    predictions = evaluator.evaluate(frame)
    uncertainty_columns = [f"{target}_uncertainty_ratio" for target in SURROGATE_TARGETS]
    predictions["combined_uncertainty_ratio"] = predictions[
        [column for column in uncertainty_columns if column in predictions.columns]
    ].mean(axis=1, skipna=True)
    predictions["surrogate_mode"] = np.where(
        predictions["pathway"].isin(tuple(SUPPORTED_SURROGATE_PATHWAYS)),
        np.select(
            [
                predictions["surrogate_prediction_status"].astype(str).eq("trained_surrogate_prediction"),
                predictions["surrogate_prediction_status"].astype(str).eq("trained_surrogate_with_documented_fallback"),
            ],
            [
                "trained_surrogate",
                "trained_surrogate_with_documented_fallback",
            ],
            default="trained_surrogate_with_documented_fallback",
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


def _optional_path(*values: object) -> Path | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        return Path(text)
    return None


def _deduplicate_candidate_paths(
    values: list[tuple[Path, Path]],
) -> list[tuple[Path, Path]]:
    deduplicated: list[tuple[Path, Path]] = []
    seen: set[tuple[str, str]] = set()
    for source_root, path in values:
        key = (str(source_root), str(path))
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append((source_root, path))
    return deduplicated


def _artifact_location_for_model(
    *,
    root: Path,
    model_key: str,
    dataset_key: str,
    target_column: str,
) -> tuple[Path, Path]:
    if model_key == "xgboost":
        artifact_dir = root / dataset_key / target_column
        return artifact_dir, artifact_dir / "model.json"
    if model_key == "catboost":
        artifact_dir = root / model_key / dataset_key / target_column
        return artifact_dir, artifact_dir / "model.cbm"
    if model_key == "lightgbm":
        artifact_dir = root / model_key / dataset_key / target_column
        return artifact_dir, artifact_dir / "model.txt"
    artifact_dir = root / model_key / dataset_key / target_column
    return artifact_dir, artifact_dir / "model.joblib"
