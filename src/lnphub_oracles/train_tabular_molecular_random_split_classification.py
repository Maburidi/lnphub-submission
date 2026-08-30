from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .io_utils import ensure_dir, now_utc, write_json
from .train_frozen_classification import (
    CLASS_LABELS,
    CLASS_NAMES,
    fit_classifier,
    format_metrics,
    make_classifier,
    predict_proba_fixed,
    safe_name,
)
from .train_tabular_molecular_classification import (
    METHODS,
    _save_eval_tables,
    build_molecular_features,
    load_molecular_subset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train random-split molecular-only 3-class ECFP4/RDKit classifiers.")
    parser.add_argument("--train-subset", required=True, type=Path)
    parser.add_argument("--validation-subset", required=True, type=Path)
    parser.add_argument("--test-subset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-col", default="delivery_class_3")
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    train, y_train, train_smiles = load_molecular_subset(args.train_subset, args.target_col, "train")
    val, y_val, val_smiles = load_molecular_subset(args.validation_subset, args.target_col, "validation")
    test, y_test, test_smiles = load_molecular_subset(args.test_subset, args.target_col, "test")

    manifest_rows = []
    all_metric_rows = []
    feature_cache = {}

    for key in args.methods:
        if key not in METHODS:
            raise ValueError(f"Unknown method: {key}. Valid methods: {sorted(METHODS)}")
        method_name, molecular_type, estimator_kind = METHODS[key]
        random_method_name = f"{method_name} Random Split"
        print("\n" + "=" * 80, flush=True)
        print(random_method_name, flush=True)
        print("=" * 80, flush=True)

        if molecular_type not in feature_cache:
            print(f"Building {molecular_type} molecular-only features...", flush=True)
            feature_cache[molecular_type] = {
                "train": build_molecular_features(train_smiles, molecular_type),
                "validation": build_molecular_features(val_smiles, molecular_type),
                "test": build_molecular_features(test_smiles, molecular_type),
            }
        X_train, feature_cols = feature_cache[molecular_type]["train"]
        X_val, val_feature_cols = feature_cache[molecular_type]["validation"]
        X_test, test_feature_cols = feature_cache[molecular_type]["test"]
        if feature_cols != val_feature_cols or feature_cols != test_feature_cols:
            raise ValueError(f"Feature mismatch for {random_method_name}")

        ok_train = np.isfinite(X_train).all(axis=1)
        ok_val = np.isfinite(X_val).all(axis=1)
        ok_test = np.isfinite(X_test).all(axis=1)
        X_train_fit, y_train_fit, train_fit = X_train[ok_train], y_train[ok_train], train.loc[ok_train].reset_index(drop=True)
        X_val_fit, y_val_fit, val_fit = X_val[ok_val], y_val[ok_val], val.loc[ok_val].reset_index(drop=True)
        X_test_fit, y_test_fit, test_fit = X_test[ok_test], y_test[ok_test], test.loc[ok_test].reset_index(drop=True)
        print(f"Feature matrix: train={X_train_fit.shape[0]:,} x {X_train_fit.shape[1]:,}; validation={X_val_fit.shape[0]:,}; test={X_test_fit.shape[0]:,}", flush=True)
        print("Train classes:", dict(pd.Series(y_train_fit).value_counts().sort_index()), flush=True)
        print("Validation classes:", dict(pd.Series(y_val_fit).value_counts().sort_index()), flush=True)
        print("Test classes:", dict(pd.Series(y_test_fit).value_counts().sort_index()), flush=True)

        model = make_classifier(estimator_kind, args.seed)
        print(f"{random_method_name}: fitting on random training split...", flush=True)
        fit_classifier(model, estimator_kind, X_train_fit, y_train_fit, X_val_fit, y_val_fit)
        print(f"{random_method_name}: fit complete", flush=True)

        val_proba = predict_proba_fixed(model, X_val_fit)
        val_pred = np.argmax(val_proba, axis=1)
        test_proba = predict_proba_fixed(model, X_test_fit)
        test_pred = np.argmax(test_proba, axis=1)

        out_dir = ensure_dir(Path(args.output_dir) / safe_name(random_method_name))
        val_metrics = _save_eval_tables(out_dir, val_fit, y_val_fit, val_pred, val_proba, "validation")
        test_metrics = _save_eval_tables(out_dir, test_fit, y_test_fit, test_pred, test_proba, "test")
        val_metrics.update({"method": random_method_name})
        test_metrics.update({"method": random_method_name})
        print(f"{random_method_name} | validation: {format_metrics(val_metrics)}", flush=True)
        print(f"{random_method_name} | test: {format_metrics(test_metrics)}", flush=True)

        joblib.dump(model, out_dir / "model.joblib")
        pd.DataFrame([val_metrics, test_metrics]).to_csv(out_dir / "metrics.csv", index=False)
        write_json(out_dir / "metrics.json", {"validation": val_metrics, "test": test_metrics})
        write_json(out_dir / "feature_schema.json", {
            "feature_columns": feature_cols,
            "molecular_feature_columns": feature_cols,
            "multimodal_feature_columns": [],
            "raw_multimodal_columns": [],
            "n_features": len(feature_cols),
            "n_molecular_features": len(feature_cols),
            "n_multimodal_features": 0,
        })
        write_json(out_dir / "metadata.json", {
            "method": random_method_name,
            "feature_set": "molecular_only",
            "molecular_feature_type": molecular_type,
            "classifier": estimator_kind,
            "target": args.target_col,
            "class_names": CLASS_NAMES,
            "train_path": str(args.train_subset),
            "validation_path": str(args.validation_subset),
            "test_path": str(args.test_subset),
            "split_strategy": "stratified random 80/10/10 split after shuffling",
            "test_used_for_training": False,
            "trained_at_utc": now_utc(),
            "output_dir": str(out_dir),
        })
        print(f"{random_method_name}: saved outputs -> {out_dir}", flush=True)

        manifest_rows.append({
            "method": random_method_name,
            "feature_set": "molecular_only",
            "status": "trained",
            "output_dir": str(out_dir),
            "validation_accuracy": val_metrics.get("accuracy"),
            "validation_balanced_accuracy": val_metrics.get("balanced_accuracy"),
            "validation_macro_f1": val_metrics.get("macro_f1"),
            "validation_roc_auc_ovr_macro": val_metrics.get("roc_auc_ovr_macro"),
            "validation_average_precision_macro": val_metrics.get("average_precision_macro"),
            "test_accuracy": test_metrics.get("accuracy"),
            "test_balanced_accuracy": test_metrics.get("balanced_accuracy"),
            "test_macro_f1": test_metrics.get("macro_f1"),
            "test_roc_auc_ovr_macro": test_metrics.get("roc_auc_ovr_macro"),
            "test_average_precision_macro": test_metrics.get("average_precision_macro"),
        })
        all_metric_rows.extend([val_metrics, test_metrics])

    manifest = pd.DataFrame(manifest_rows)
    all_metrics = pd.DataFrame(all_metric_rows)
    manifest_path = Path(args.output_dir) / "tabular_molecular_random_split_manifest.csv"
    all_metrics_path = Path(args.output_dir) / "tabular_molecular_random_split_all_metrics.csv"
    manifest.to_csv(manifest_path, index=False)
    all_metrics.to_csv(all_metrics_path, index=False)
    print("\nSaved molecular-only random-split manifest:", manifest_path, flush=True)
    print("Saved molecular-only random-split metrics:", all_metrics_path, flush=True)
    print(manifest.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
