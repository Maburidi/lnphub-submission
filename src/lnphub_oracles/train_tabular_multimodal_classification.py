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
    classification_metrics,
    fit_classifier,
    format_metrics,
    id_columns,
    make_classifier,
    predict_proba_fixed,
    safe_name,
)


METHODS = {
    "ecfp4_multimodal_mlp": ("ECFP4 + Formulation/PhysChem + MLP", "ecfp4", "mlp"),
    "ecfp4_multimodal_xgboost": ("ECFP4 + Formulation/PhysChem + XGBoost", "ecfp4", "xgboost"),
    "ecfp4_multimodal_random_forest": ("ECFP4 + Formulation/PhysChem + Random Forest", "ecfp4", "random_forest"),
    "rdkit_descriptors_multimodal_mlp": ("RDKit Descriptors + Formulation/PhysChem + MLP", "rdkit_descriptors", "mlp"),
    "rdkit_descriptors_multimodal_xgboost": ("RDKit Descriptors + Formulation/PhysChem + XGBoost", "rdkit_descriptors", "xgboost"),
    "rdkit_descriptors_multimodal_random_forest": ("RDKit Descriptors + Formulation/PhysChem + Random Forest", "rdkit_descriptors", "random_forest"),
}

DEFAULT_MULTIMODAL_COLS = [
    "il_mol_ratio",
    "helper_lipid_mol_ratio",
    "cholesterol_mol_ratio",
    "peg_lipid_mol_ratio",
    "particle_size_nm",
    "pdi",
]
FORMULATION_ONLY_COLS = [
    "il_mol_ratio",
    "helper_lipid_mol_ratio",
    "cholesterol_mol_ratio",
    "peg_lipid_mol_ratio",
]


def feature_set_label(cols: list[str]) -> str:
    if cols == FORMULATION_ONLY_COLS:
        return "Formulation"
    if cols == DEFAULT_MULTIMODAL_COLS:
        return "Formulation/PhysChem"
    return "Selected Covariates"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 3-class ECFP4/RDKit descriptor classifiers with formulation and physicochemical covariates."
    )
    parser.add_argument("--trainval-subset", required=True, type=Path)
    parser.add_argument("--test-subset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-col", default="delivery_class_3")
    parser.add_argument("--multimodal-cols", nargs="+", default=DEFAULT_MULTIMODAL_COLS)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--cv-description", default="leave-one-study-out CV on balanced trainval data")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def numeric_multimodal_matrix(frame: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, list[str]]:
    missing = [col for col in cols if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing multimodal feature columns: {missing}")
    values = frame[cols].apply(pd.to_numeric, errors="coerce")
    feature_names = [f"multimodal_{col}" for col in cols]
    return values.to_numpy(dtype=np.float32), feature_names


def load_subset(path: Path, target_col: str, multimodal_cols: list[str], split_name: str) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    frame = read_table(path).copy()
    if target_col not in frame.columns:
        raise ValueError(f"{split_name} subset is missing target column: {target_col}")
    y = pd.to_numeric(frame[target_col], errors="coerce").to_numpy(dtype=float)
    smiles = smiles_series(frame)
    ok = np.isfinite(y) & valid_smiles_mask(smiles)

    for col in multimodal_cols:
        if col not in frame.columns:
            raise ValueError(f"{split_name} subset is missing multimodal column: {col}")
        ok &= pd.to_numeric(frame[col], errors="coerce").notna().to_numpy()

    dropped = int((~ok).sum())
    if dropped:
        print(f"{split_name}: dropping {dropped:,} rows with missing class, invalid SMILES, or missing multimodal features", flush=True)

    frame = frame.loc[ok].reset_index(drop=True)
    y = pd.to_numeric(frame[target_col], errors="coerce").to_numpy(dtype=int)
    smiles = smiles_series(frame).reset_index(drop=True)

    print(f"{split_name}: {len(frame):,} usable rows", flush=True)
    print(f"{split_name} class counts: {dict(pd.Series(y).value_counts().sort_index())}", flush=True)
    return frame, y, smiles


def build_features(
    frame: pd.DataFrame,
    smiles: pd.Series,
    molecular_type: str,
    multimodal_cols: list[str],
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    if molecular_type == "ecfp4":
        X_mol, mol_cols = make_ecfp4(smiles)
    elif molecular_type == "rdkit_descriptors":
        X_mol, mol_cols = make_rdkit_descriptors(smiles)
    else:
        raise ValueError(f"Unknown molecular feature type: {molecular_type}")

    X_multi, multi_cols = numeric_multimodal_matrix(frame, multimodal_cols)
    X = np.concatenate([X_mol, X_multi], axis=1).astype(np.float32)
    return X, mol_cols + multi_cols, mol_cols, multi_cols


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    trainval, y, smiles = load_subset(args.trainval_subset, args.target_col, args.multimodal_cols, "trainval")
    test, y_test, test_smiles = load_subset(args.test_subset, args.target_col, args.multimodal_cols, "untouched_test")

    if "cv_fold" not in trainval.columns:
        raise ValueError("trainval subset is missing cv_fold. Run the notebook split cell first.")

    manifest_rows = []
    all_metric_rows = []
    feature_cache: dict[str, tuple[np.ndarray, list[str], list[str], list[str]]] = {}
    test_feature_cache: dict[str, tuple[np.ndarray, list[str], list[str], list[str]]] = {}

    for key in args.methods:
        if key not in METHODS:
            raise ValueError(f"Unknown method: {key}. Valid methods: {sorted(METHODS)}")

        method_name, molecular_type, estimator_kind = METHODS[key]
        method_name = method_name.replace("Formulation/PhysChem", feature_set_label(args.multimodal_cols))
        print("", flush=True)
        print("=" * 80, flush=True)
        print(method_name, flush=True)
        print("=" * 80, flush=True)

        if molecular_type not in feature_cache:
            print(f"Building {molecular_type} + multimodal trainval features...", flush=True)
            feature_cache[molecular_type] = build_features(trainval, smiles, molecular_type, args.multimodal_cols)
        X, feature_cols, mol_cols, multi_cols = feature_cache[molecular_type]

        if molecular_type not in test_feature_cache:
            print(f"Building {molecular_type} + multimodal untouched-test features...", flush=True)
            test_feature_cache[molecular_type] = build_features(test, test_smiles, molecular_type, args.multimodal_cols)
        X_test, test_feature_cols, _, _ = test_feature_cache[molecular_type]
        if feature_cols != test_feature_cols:
            raise ValueError(f"Feature mismatch for {method_name}")

        ok = np.isfinite(X).all(axis=1)
        test_ok = np.isfinite(X_test).all(axis=1)
        if int((~ok).sum()) or int((~test_ok).sum()):
            print(
                f"{method_name}: dropping non-finite feature rows trainval={(~ok).sum():,}, test={(~test_ok).sum():,}",
                flush=True,
            )
        X_fit, y_fit, frame_fit = X[ok], y[ok], trainval.loc[ok].reset_index(drop=True)
        X_eval, y_eval, test_eval = X_test[test_ok], y_test[test_ok], test.loc[test_ok].reset_index(drop=True)

        print(
            f"Feature matrix: trainval={X_fit.shape[0]:,} x {X_fit.shape[1]:,}; "
            f"test={X_eval.shape[0]:,} x {X_eval.shape[1]:,}",
            flush=True,
        )
        print(f"Molecular features: {len(mol_cols):,}; appended numeric features: {len(multi_cols):,}", flush=True)

        out_dir = ensure_dir(Path(args.output_dir) / safe_name(method_name))
        cv_predictions = []
        metric_rows = []
        folds = sorted(int(f) for f in frame_fit["cv_fold"].dropna().unique() if int(f) >= 0)

        for fold in folds:
            val_idx = np.where(frame_fit["cv_fold"].astype(int).to_numpy() == fold)[0]
            train_idx = np.setdiff1d(np.arange(len(frame_fit)), val_idx)
            validation_study = ", ".join(sorted(frame_fit.iloc[val_idx]["study_id"].astype(str).unique()))
            print("", flush=True)
            print(f"{method_name} | fold {fold}: validation study = {validation_study}", flush=True)
            print(f"{method_name} | fold {fold}: train={len(train_idx):,}, val={len(val_idx):,}", flush=True)
            print("  train classes:", dict(pd.Series(y_fit[train_idx]).value_counts().sort_index()), flush=True)
            print("  val classes:", dict(pd.Series(y_fit[val_idx]).value_counts().sort_index()), flush=True)
            print(f"{method_name} | fold {fold}: fitting...", flush=True)

            model = make_classifier(estimator_kind, args.seed + fold)
            fit_classifier(model, estimator_kind, X_fit[train_idx], y_fit[train_idx], X_fit[val_idx], y_fit[val_idx])
            proba = predict_proba_fixed(model, X_fit[val_idx])
            pred = np.argmax(proba, axis=1)
            metrics = classification_metrics(y_fit[val_idx], pred, proba)
            metrics.update({"method": method_name, "split": f"fold_{fold}", "fold": fold, "validation_study": validation_study})
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
        print("", flush=True)
        print(f"{method_name} | cv_mean: {format_metrics(cv_mean)}", flush=True)

        print(f"{method_name}: fitting final classifier on all trainval rows...", flush=True)
        final_model = make_classifier(estimator_kind, args.seed)
        fit_classifier(final_model, estimator_kind, X_fit, y_fit)
        print(f"{method_name}: final model fit complete", flush=True)

        test_proba = predict_proba_fixed(final_model, X_eval)
        test_pred = np.argmax(test_proba, axis=1)
        test_metrics = classification_metrics(y_eval, test_pred, test_proba)
        test_metrics.update({"method": method_name, "split": "untouched_test", "fold": -1, "validation_study": ", ".join(sorted(test_eval["study_id"].astype(str).unique()))})
        metric_rows.append(test_metrics)
        all_metric_rows.append(test_metrics)
        print(f"{method_name} | untouched_test: {format_metrics(test_metrics)}", flush=True)

        test_pred_df = test_eval[id_columns(test_eval)].copy()
        test_pred_df["split"] = "untouched_test"
        test_pred_df["y_true"] = y_eval
        test_pred_df["y_pred"] = test_pred
        for cls in CLASS_LABELS:
            test_pred_df[f"prob_{CLASS_NAMES[cls]}"] = test_proba[:, cls]

        cm = confusion_matrix(y_eval, test_pred, labels=CLASS_LABELS)
        cm_df = pd.DataFrame(cm, index=[f"true_{CLASS_NAMES[c]}" for c in CLASS_LABELS], columns=[f"pred_{CLASS_NAMES[c]}" for c in CLASS_LABELS])
        per_class = precision_recall_fscore_support(y_eval, test_pred, labels=CLASS_LABELS, zero_division=0)
        per_class_df = pd.DataFrame({
            "class": [CLASS_NAMES[c] for c in CLASS_LABELS],
            "precision": per_class[0],
            "recall": per_class[1],
            "f1": per_class[2],
            "support": per_class[3],
        })

        joblib.dump(final_model, out_dir / "model.joblib")
        pd.concat(cv_predictions, ignore_index=True).to_csv(out_dir / "cv_predictions.csv", index=False)
        test_pred_df.to_csv(out_dir / "untouched_test_predictions.csv", index=False)
        pd.DataFrame(metric_rows).to_csv(out_dir / "metrics.csv", index=False)
        cm_df.to_csv(out_dir / "untouched_test_confusion_matrix.csv")
        per_class_df.to_csv(out_dir / "untouched_test_per_class_metrics.csv", index=False)
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
            "method": method_name,
            "molecular_feature_type": molecular_type,
            "classifier": estimator_kind,
            "target": args.target_col,
            "class_names": CLASS_NAMES,
            "trainval_path": str(args.trainval_subset),
            "untouched_test_path": str(args.test_subset),
            "cv": args.cv_description,
            "untouched_test_used_for_training": False,
            "trained_at_utc": now_utc(),
            "output_dir": str(out_dir),
        })
        print(f"{method_name}: saved outputs -> {out_dir}", flush=True)

        manifest_rows.append({
            "method": method_name,
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
    manifest_path = Path(args.output_dir) / "tabular_multimodal_classification_manifest.csv"
    all_metrics_path = Path(args.output_dir) / "tabular_multimodal_all_metrics.csv"
    manifest.to_csv(manifest_path, index=False)
    all_metrics.to_csv(all_metrics_path, index=False)

    print("", flush=True)
    print("Saved multimodal tabular classification manifest:", manifest_path, flush=True)
    print("Saved all multimodal tabular metrics:", all_metrics_path, flush=True)
    print("", flush=True)
    print(manifest.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
