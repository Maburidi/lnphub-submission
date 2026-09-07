from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .features import make_ecfp4, make_rdkit_descriptors, smiles_series, valid_smiles_mask
from .io_utils import ensure_dir, now_utc, read_table, write_json
from .train_frozen_classification import (
    CLASS_LABELS,
    CLASS_NAMES,
    class_names_for,
    classification_metrics,
    fit_classifier,
    format_metrics,
    id_columns,
    infer_class_labels,
    make_classifier,
    predict_proba_fixed,
    safe_name,
)


METHODS = {
    "ecfp4_mlp": ("ECFP4 + MLP", "ecfp4", "mlp"),
    "ecfp4_xgboost": ("ECFP4 + XGBoost", "ecfp4", "xgboost"),
    "ecfp4_random_forest": ("ECFP4 + Random Forest", "ecfp4", "random_forest"),
    "rdkit_descriptors_mlp": ("RDKit Descriptors + MLP", "rdkit_descriptors", "mlp"),
    "rdkit_descriptors_xgboost": ("RDKit Descriptors + XGBoost", "rdkit_descriptors", "xgboost"),
    "rdkit_descriptors_random_forest": ("RDKit Descriptors + Random Forest", "rdkit_descriptors", "random_forest"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train molecular-only 3-class ECFP4/RDKit descriptor classifiers.")
    parser.add_argument("--trainval-subset", required=True, type=Path)
    parser.add_argument("--test-subset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-col", default="delivery_class_3")
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--cv-description", default="stratified random 5-fold CV on balanced trainval data")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def load_molecular_subset(path: Path, target_col: str, split_name: str) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    frame = read_table(path).copy()
    if target_col not in frame.columns:
        raise ValueError(f"{split_name} subset is missing target column: {target_col}")
    y = pd.to_numeric(frame[target_col], errors="coerce").to_numpy(dtype=float)
    smiles = smiles_series(frame)
    ok = np.isfinite(y) & valid_smiles_mask(smiles)
    dropped = int((~ok).sum())
    if dropped:
        print(f"{split_name}: dropping {dropped:,} rows with missing class or invalid SMILES", flush=True)
    frame = frame.loc[ok].reset_index(drop=True)
    y = pd.to_numeric(frame[target_col], errors="coerce").to_numpy(dtype=int)
    smiles = smiles_series(frame).reset_index(drop=True)
    print(f"{split_name}: {len(frame):,} usable rows", flush=True)
    print(f"{split_name} class counts: {dict(pd.Series(y).value_counts().sort_index())}", flush=True)
    return frame, y, smiles


def build_molecular_features(smiles: pd.Series, molecular_type: str) -> tuple[np.ndarray, list[str]]:
    if molecular_type == "ecfp4":
        return make_ecfp4(smiles)
    if molecular_type == "rdkit_descriptors":
        return make_rdkit_descriptors(smiles)
    raise ValueError(f"Unknown molecular feature type: {molecular_type}")


def _save_eval_tables(out_dir: Path, frame: pd.DataFrame, y_true, pred, proba, split_name: str) -> dict[str, float]:
    class_labels = infer_class_labels(y_true, pred, proba)
    class_names = class_names_for(class_labels)
    metrics = classification_metrics(y_true, pred, proba, class_labels=class_labels)
    metrics.update({"split": split_name})
    pred_df = frame[id_columns(frame)].copy()
    pred_df["split"] = split_name
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = pred
    for i, cls in enumerate(class_labels):
        pred_df[f"prob_{class_names[cls]}"] = proba[:, i]
    pred_df.to_csv(out_dir / f"{split_name}_predictions.csv", index=False)

    cm = confusion_matrix(y_true, pred, labels=class_labels)
    pd.DataFrame(
        cm,
        index=[f"true_{class_names[c]}" for c in class_labels],
        columns=[f"pred_{class_names[c]}" for c in class_labels],
    ).to_csv(out_dir / f"{split_name}_confusion_matrix.csv")
    per_class = precision_recall_fscore_support(y_true, pred, labels=class_labels, zero_division=0)
    pd.DataFrame({
        "class": [class_names[c] for c in class_labels],
        "precision": per_class[0],
        "recall": per_class[1],
        "f1": per_class[2],
        "support": per_class[3],
    }).to_csv(out_dir / f"{split_name}_per_class_metrics.csv", index=False)
    return metrics


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    trainval, y, smiles = load_molecular_subset(args.trainval_subset, args.target_col, "trainval")
    test, y_test, test_smiles = load_molecular_subset(args.test_subset, args.target_col, "test")
    if "cv_fold" not in trainval.columns:
        raise ValueError("trainval subset is missing cv_fold.")

    manifest_rows = []
    all_metric_rows = []
    feature_cache = {}
    test_feature_cache = {}

    for key in args.methods:
        if key not in METHODS:
            raise ValueError(f"Unknown method: {key}. Valid methods: {sorted(METHODS)}")
        method_name, molecular_type, estimator_kind = METHODS[key]
        print("\n" + "=" * 80, flush=True)
        print(method_name, flush=True)
        print("=" * 80, flush=True)

        if molecular_type not in feature_cache:
            print(f"Building {molecular_type} trainval features...", flush=True)
            feature_cache[molecular_type] = build_molecular_features(smiles, molecular_type)
        X, feature_cols = feature_cache[molecular_type]
        if molecular_type not in test_feature_cache:
            print(f"Building {molecular_type} test features...", flush=True)
            test_feature_cache[molecular_type] = build_molecular_features(test_smiles, molecular_type)
        X_test, test_feature_cols = test_feature_cache[molecular_type]
        if feature_cols != test_feature_cols:
            raise ValueError(f"Feature mismatch for {method_name}")

        ok = np.isfinite(X).all(axis=1)
        test_ok = np.isfinite(X_test).all(axis=1)
        X_fit, y_fit, frame_fit = X[ok], y[ok], trainval.loc[ok].reset_index(drop=True)
        X_eval, y_eval, test_eval = X_test[test_ok], y_test[test_ok], test.loc[test_ok].reset_index(drop=True)
        print(f"Feature matrix: trainval={X_fit.shape[0]:,} x {X_fit.shape[1]:,}; test={X_eval.shape[0]:,} x {X_eval.shape[1]:,}", flush=True)

        out_dir = ensure_dir(Path(args.output_dir) / safe_name(method_name))
        cv_predictions = []
        metric_rows = []
        folds = sorted(int(f) for f in frame_fit["cv_fold"].dropna().unique() if int(f) >= 0)
        for fold in folds:
            val_idx = np.where(frame_fit["cv_fold"].astype(int).to_numpy() == fold)[0]
            train_idx = np.setdiff1d(np.arange(len(frame_fit)), val_idx)
            validation_studies = ", ".join(sorted(frame_fit.iloc[val_idx]["study_id"].astype(str).unique()))
            print(f"\n{method_name} | fold {fold}: train={len(train_idx):,}, val={len(val_idx):,}; validation studies={validation_studies}", flush=True)
            model = make_classifier(estimator_kind, args.seed + fold)
            fit_classifier(model, estimator_kind, X_fit[train_idx], y_fit[train_idx], X_fit[val_idx], y_fit[val_idx])
            proba = predict_proba_fixed(model, X_fit[val_idx])
            pred = np.argmax(proba, axis=1)
            metrics = classification_metrics(y_fit[val_idx], pred, proba)
            metrics.update({"method": method_name, "split": f"fold_{fold}", "fold": fold, "validation_study": validation_studies})
            metric_rows.append(metrics)
            all_metric_rows.append(metrics)
            print(f"{method_name} | fold_{fold}: {format_metrics(metrics)}", flush=True)

            part = frame_fit.iloc[val_idx][id_columns(frame_fit)].copy()
            part["split"] = f"fold_{fold}"
            part["cv_fold"] = fold
            part["y_true"] = y_fit[val_idx]
            part["y_pred"] = pred
            for cls in CLASS_LABELS:
                part[f"prob_{CLASS_NAMES[cls]}"] = proba[:, cls]
            cv_predictions.append(part)

        fold_metrics = pd.DataFrame(metric_rows)
        cv_mean = fold_metrics.drop(columns=["method", "split", "fold", "validation_study"], errors="ignore").mean(numeric_only=True).to_dict()
        cv_mean.update({"method": method_name, "split": "cv_mean", "fold": -1, "validation_study": args.cv_description})
        cv_std = fold_metrics.drop(columns=["method", "split", "fold", "validation_study"], errors="ignore").std(numeric_only=True).to_dict()
        cv_std.update({"method": method_name, "split": "cv_std", "fold": -1, "validation_study": args.cv_description})
        metric_rows.extend([cv_mean, cv_std])
        all_metric_rows.extend([cv_mean, cv_std])
        print(f"\n{method_name} | cv_mean: {format_metrics(cv_mean)}", flush=True)

        final_model = make_classifier(estimator_kind, args.seed)
        print(f"{method_name}: fitting final model on all trainval rows...", flush=True)
        fit_classifier(final_model, estimator_kind, X_fit, y_fit)
        print(f"{method_name}: final model fit complete", flush=True)
        test_proba = predict_proba_fixed(final_model, X_eval)
        test_pred = np.argmax(test_proba, axis=1)
        test_metrics = _save_eval_tables(out_dir, test_eval, y_eval, test_pred, test_proba, "test")
        test_metrics.update({"method": method_name, "split": "test", "fold": -1, "validation_study": ", ".join(sorted(test_eval["study_id"].astype(str).unique()))})
        metric_rows.append(test_metrics)
        all_metric_rows.append(test_metrics)
        print(f"{method_name} | test: {format_metrics(test_metrics)}", flush=True)

        joblib.dump(final_model, out_dir / "model.joblib")
        pd.concat(cv_predictions, ignore_index=True).to_csv(out_dir / "cv_predictions.csv", index=False)
        pd.DataFrame(metric_rows).to_csv(out_dir / "metrics.csv", index=False)
        write_json(out_dir / "metrics.json", {row["split"]: row for row in metric_rows})
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
            "method": method_name,
            "feature_set": "molecular_only",
            "molecular_feature_type": molecular_type,
            "classifier": estimator_kind,
            "target": args.target_col,
            "class_names": CLASS_NAMES,
            "trainval_path": str(args.trainval_subset),
            "test_path": str(args.test_subset),
            "cv": args.cv_description,
            "test_used_for_training": False,
            "trained_at_utc": now_utc(),
            "output_dir": str(out_dir),
        })
        print(f"{method_name}: saved outputs -> {out_dir}", flush=True)

        manifest_rows.append({
            "method": method_name,
            "feature_set": "molecular_only",
            "status": "trained",
            "output_dir": str(out_dir),
            "cv_accuracy": cv_mean.get("accuracy"),
            "cv_balanced_accuracy": cv_mean.get("balanced_accuracy"),
            "cv_macro_f1": cv_mean.get("macro_f1"),
            "cv_roc_auc_ovr_macro": cv_mean.get("roc_auc_ovr_macro"),
            "cv_average_precision_macro": cv_mean.get("average_precision_macro"),
            "test_accuracy": test_metrics.get("accuracy"),
            "test_balanced_accuracy": test_metrics.get("balanced_accuracy"),
            "test_macro_f1": test_metrics.get("macro_f1"),
            "test_roc_auc_ovr_macro": test_metrics.get("roc_auc_ovr_macro"),
            "test_average_precision_macro": test_metrics.get("average_precision_macro"),
        })

    manifest = pd.DataFrame(manifest_rows)
    all_metrics = pd.DataFrame(all_metric_rows)
    manifest_path = Path(args.output_dir) / "tabular_molecular_classification_manifest.csv"
    all_metrics_path = Path(args.output_dir) / "tabular_molecular_all_metrics.csv"
    manifest.to_csv(manifest_path, index=False)
    all_metrics.to_csv(all_metrics_path, index=False)
    print("\nSaved molecular-only classification manifest:", manifest_path, flush=True)
    print("Saved all molecular-only metrics:", all_metrics_path, flush=True)
    print(manifest.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
