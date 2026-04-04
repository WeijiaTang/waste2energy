from __future__ import annotations

import argparse
import json
import sys

from .data import DATASET_KEYS, TARGET_COLUMNS
from .models import MODEL_KEYS
from .surrogates import run_regression_baseline, run_regression_baseline_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train traditional machine-learning baselines for Waste2Energy datasets."
    )
    parser.add_argument(
        "--model",
        choices=MODEL_KEYS,
        default="xgboost",
        help="Model family to train. Default: xgboost",
    )
    parser.add_argument("--dataset", choices=DATASET_KEYS, help="Dataset key to train on.")
    parser.add_argument("--target", choices=TARGET_COLUMNS, help="Regression target column.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional explicit output directory. Default: outputs/xgboost/<dataset>/<target>",
    )
    parser.add_argument(
        "--split-strategy",
        choices=(
            "recommended",
            "strict_group",
            "leave_source_repo_out",
            "leave_study_out",
        ),
        default="recommended",
        help="Dataset split strategy. Default: recommended",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Print supported dataset keys and exit.",
    )
    parser.add_argument(
        "--list-targets",
        action="store_true",
        help="Print supported target columns and exit.",
    )
    parser.add_argument(
        "--run-suite",
        action="store_true",
        help="Run the full baseline suite for all datasets and targets, or for the provided filters.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="When used with --run-suite, run all supported model families.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print supported model families and exit.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_datasets:
        print("\n".join(DATASET_KEYS))
        return 0
    if args.list_targets:
        print("\n".join(TARGET_COLUMNS))
        return 0
    if args.list_models:
        print("\n".join(MODEL_KEYS))
        return 0
    if args.run_suite:
        selected_models = None if args.all_models else [args.model]
        selected_datasets = [args.dataset] if args.dataset else None
        selected_targets = [args.target] if args.target else None
        try:
            result = run_regression_baseline_suite(
                model_keys=selected_models,
                dataset_keys=selected_datasets,
                target_columns=selected_targets,
                output_root=args.output_dir or None,
                split_strategy=args.split_strategy,
            )
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(json.dumps(result, indent=2))
        return 0
    if not args.dataset or not args.target:
        parser.error("--dataset and --target are required unless a list option is used.")

    try:
        result = run_regression_baseline(
            model_key=args.model,
            dataset_key=args.dataset,
            target_column=args.target,
            output_dir=args.output_dir or None,
            split_strategy=args.split_strategy,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
