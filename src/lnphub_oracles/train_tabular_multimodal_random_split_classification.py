from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .io_utils import ensure_dir, now_utc, read_table, write_json
from .train_frozen_classification import (
    CLASS_LABELS,
    CLASS_NAMES,
    classification_metrics,
    fit_classifier,
    format_metrics,
    id_columns,
    make_classifier,
    predict_proba_fixed,
    safe_name,
)
from .train_tabular_multimodal_classification import (
    DEFAULT_MULTIMODAL_COLS,
    METHODS,
    build_features,
    load_subset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train random-split 3-class ECFP4/RDKit multimodal classifiers."
    )
    parser.add_argument("--train-subset", required=True, type=Path)
    parser.add_argument("--validation-subset", required=True, type=Path)
    parser.add_argument("--test-subset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-col", default="delivery_class_3")
    parser.add_argument("--multimodal-cols", nargs="+", default=DEFAULT_MULTIMODAL_COLS)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def evaluate_split(model, X, y, frame, split_name: str) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    proba = predict_proba_fixed(model, X)
    pred = np.argmax(proba, axis=1)
    metrics = classification_metrics(y, pred, proba)
    metrics.update({"split": split_name})

    pred_df = frame[id_columns(frame)].copy()
    pred_df["split"] = split_name
    pred_df["y_true"] = y
    pred_df["y_pred"] = pred
    for cls in CLASS_LABELS:
        pred_df[f"prob_{CLASS_NAMES[cls]}"] = proba[:, cls]

    cm = confusion_matrix(y, pred, labels=CLASS_LABELS)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{CLASS_NAMES[c]}" for c in CLASS_LABELS],
        columns=[f"pred_{CLASS_NAMES[c]}" for c in CLASS_LABELS],
    )
    per_class = precision_recall_fscore_support(y, pred, labels=CLASS_LABELS, zero_division=0)
    per_class_df = pd.DataFrame({
        "class": [CLASS_NAMES[c] for c in CLASS_LABELS],
        "precision": per_class[0],
        "recall": per_class[1],
        "f1": per_class[2],
        "support": per_class[3],
    })
    return metrics, pred_df, cm_df, per_class_df


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    train, y_train, train_smiles = load_subset(args.train_subset, args.target_col, args.multimodal_cols, "train")
    val, y_val, val_smiles = load_subset(args.validation_subset, args.target_col, args.multimodal_cols, "validation")
    test, y_test, test_smiles = load_subset(args.test_subset, args.target_col, args.multimodal_cols, "test")

    manifest_rows = []
    all_metric_rows = []
    feature_cache = {}

    for key in args.methods:
        if key not in METHODS:
            raise ValueError(f"Unknown method: {key}. Valid methods: {sorted(METHODS)}")

        method_name, molecular_type, estimator_kind = METHODS[key]
        random_method_name = f"{method_name} Random Split"
        print("", flush=True)
        print("=" * 80, flush=True)
        print(random_method_name, flush=True)
        print("=" * 80, flush=True)

        if molecular_type not in feature_cache:
            print(f"Building {molecular_type} + multimodal features...", flush=True)
            feature_cache[molecular_type] = {
                "train": build_features(train, train_smiles, molecular_type, args.multimodal_cols),
                "validation": build_features(val, val_smiles, molecular_type, args.multimodal_cols),
                "test": build_features(test, test_smiles, molecular_type, args.multimodal_cols),
            }

        X_train, feature_cols, mol_cols, multi_cols = feature_cache[molecular_type]["train"]
        X_val, val_feature_cols, _, _ = feature_cache[molecular_type]["validation"]
        X_test, test_feature_cols, _, _ = feature_cache[molecular_type]["test"]
        if feature_cols != val_feature_cols or feature_cols != test_feature_cols:
            raise ValueError(f"Feature mismatch for {random_method_name}")

        ok_train = np.isfinite(X_train).all(axis=1)
        ok_val = np.isfinite(X_val).all(axis=1)
        ok_test = np.isfinite(X_test).all(axis=1)
        X_train_fit, y_train_fit, train_fit = X_train[ok_train], y_train[ok_train], train.loc[ok_train].reset_index(drop=True)
        X_val_fit, y_val_fit, val_fit = X_val[ok_val], y_val[ok_val], val.loc[ok_val].reset_index(drop=True)
        X_test_fit, y_test_fit, test_fit = X_test[ok_test], y_test[ok_test], test.loc[ok_test].reset_index(drop=True)

        print(
            f"Feature matrix: train={X_train_fit.shape[0]:,} x {X_train_fit.shape[1]:,}; "
            f"validation={X_val_fit.shape[0]:,}; test={X_test_fit.shape[0]:,}",
            flush=True,
        )
        print(f"Molecular features: {len(mol_cols):,}; formulation/physchem features: {len(multi_cols):,}", flush=True)
        print("Train classes:", dict(pd.Series(y_train_fit).value_counts().sort_index()), flush=True)
        print("Validation classes:", dict(pd.Series(y_val_fit).value_counts().sort_index()), flush=True)
        print("Test classes:", dict(pd.Series(y_test_fit).value_counts().sort_index()), flush=True)

        model = make_classifier(estimator_kind, args.seed)
        print(f"{random_method_name}: fitting on random training split...", flush=True)
        fit_classifier(model, estimator_kind, X_train_fit, y_train_fit, X_val_fit, y_val_fit)
        print(f"{random_method_name}: fit complete", flush=True)

        val_metrics, val_pred, val_cm, val_per_class = evaluate_split(model, X_val_fit, y_val_fit, val_fit, "validation")
        test_metrics, test_pred, test_cm, test_per_class = evaluate_split(model, X_test_fit, y_test_fit, test_fit, "test")
        val_metrics.update({"method": random_method_name})
        test_metrics.update({"method": random_method_name})

        print(f"{random_method_name} | validation: {format_metrics(val_metrics)}", flush=True)
        print(f"{random_method_name} | test: {format_metrics(test_metrics)}", flush=True)

        out_dir = ensure_dir(Path(args.output_dir) / safe_name(random_method_name))
        joblib.dump(model, out_dir / "model.joblib")
        val_pred.to_csv(out_dir / "validation_predictions.csv", index=False)
        test_pred.to_csv(out_dir / "test_predictions.csv", index=False)
        val_cm.to_csv(out_dir / "validation_confusion_matrix.csv")
        test_cm.to_csv(out_dir / "test_confusion_matrix.csv")
        val_per_class.to_csv(out_dir / "validation_per_class_metrics.csv", index=False)
        test_per_class.to_csv(out_dir / "test_per_class_metrics.csv", index=False)
        pd.DataFrame([val_metrics, test_metrics]).to_csv(out_dir / "metrics.csv", index=False)
        write_json(out_dir / "metrics.json", {"validation": val_metrics, "test": test_metrics})
        write_json(out_dir / "feature_schema.json", {
            "feature_columns": feature_cols,
            "molecular_feature_columns": mol_cols,
            "multimodal_feature_columns": multi_cols,
            "raw_multimodal_columns": args.multimodal_cols,
            "n_features": len(feature_cols),
            "n_molecular_features": len(mol_cols),
            "n_multimodal_features": len(multi_cols),
        })
        write_json(out_dir / "metadata.json", {
            "method": random_method_name,
            "molecular_feature_type": molecular_type,
            "classifier": estimator_kind,
            "target": args.target_col,
            "class_names": CLASS_NAMES,
            "train_path": str(args.train_subset),
            "validation_path": str(args.validation_subset),
            "test_path": str(args.test_subset),
            "split_strategy": "stratified random train/validation/test split after shuffling",
            "test_used_for_training": False,
            "trained_at_utc": now_utc(),
            "output_dir": str(out_dir),
        })
        print(f"{random_method_name}: saved outputs -> {out_dir}", flush=True)

        row = {
            "method": random_method_name,
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
        }
        manifest_rows.append(row)
        all_metric_rows.extend([val_metrics, test_metrics])

    manifest = pd.DataFrame(manifest_rows)
    all_metrics = pd.DataFrame(all_metric_rows)
    manifest_path = Path(args.output_dir) / "tabular_multimodal_random_split_manifest.csv"
    all_metrics_path = Path(args.output_dir) / "tabular_multimodal_random_split_all_metrics.csv"
    manifest.to_csv(manifest_path, index=False)
    all_metrics.to_csv(all_metrics_path, index=False)

    print("", flush=True)
    print("Saved random-split multimodal manifest:", manifest_path, flush=True)
    print("Saved random-split multimodal metrics:", all_metrics_path, flush=True)
    print("", flush=True)
    print(manifest.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
