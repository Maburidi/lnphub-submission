from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None

from .io_utils import ensure_dir, now_utc, read_table, write_json


DEFAULT_ENCODERS = ["AGILE", "LiON", "ChemBERTa_MTR", "MoLFormer", "MolCLR_GIN", "Uni_Mol"]
PRETTY = {
    "ChemBERTa_MTR": "ChemBERTa-MTR",
    "MolCLR_GIN": "MolCLR-GIN",
    "Uni_Mol": "Uni-Mol",
}
ID_COLS = {
    "record_id",
    "study_id",
    "lnp_id",
    "il_id",
    "il_smiles",
    "canonical_il_smiles",
    "delivery",
    "smiles_index",
    "oracle_split",
    "cv_fold",
    "delivery_class_3",
    "delivery_class_label",
}
CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = {0: "low", 1: "medium", 2: "high"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frozen-embedding 3-class LNPHub classifiers.")
    parser.add_argument("--trainval-subset", required=True, type=Path)
    parser.add_argument("--test-subset", required=True, type=Path)
    parser.add_argument("--encoded-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-col", default="delivery_class_3")
    parser.add_argument("--id-col", default="record_id")
    parser.add_argument("--encoders", nargs="+", default=DEFAULT_ENCODERS)
    parser.add_argument("--estimator", default="mlp", choices=["mlp", "xgboost", "random_forest"])
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def encoder_file(encoded_dir: Path, encoder: str) -> Path:
    return encoded_dir / f"{encoder}_encoder_vectors.parquet"


def safe_name(name: str) -> str:
    return name.replace(" + ", "_plus_").replace(" ", "_").replace("/", "_").replace("-", "_")


def vector_columns(frame: pd.DataFrame) -> list[str]:
    cols = []
    for col in frame.columns:
        if col in ID_COLS:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.notna().sum() > 0:
            cols.append(col)
    if not cols:
        raise ValueError("No numeric vector columns found.")
    return cols


def align_vectors_to_subset(
    encoded: pd.DataFrame,
    subset: pd.DataFrame,
    *,
    id_col: str,
    target_col: str,
) -> pd.DataFrame:
    if id_col not in encoded.columns:
        raise ValueError(f"Encoded file is missing id column: {id_col}")
    if id_col not in subset.columns:
        raise ValueError(f"Subset file is missing id column: {id_col}")
    if target_col not in subset.columns:
        raise ValueError(f"Subset file is missing target column: {target_col}")

    encoded = encoded.drop_duplicates(subset=[id_col]).copy()
    keep_cols = [id_col, target_col, "cv_fold", "oracle_split", "delivery_class_label"]
    keep_cols += [c for c in ["study_id", "lnp_id", "il_id", "il_smiles", "canonical_il_smiles", "delivery"] if c in subset.columns]
    labels = subset[[c for c in keep_cols if c in subset.columns]].copy()
    labels["_order"] = np.arange(len(labels))

    vector_cols = vector_columns(encoded)
    merged = labels.merge(encoded[[id_col] + vector_cols], on=id_col, how="left", validate="many_to_one")
    missing = merged[vector_cols].isna().all(axis=1).sum()
    if missing:
        print(f"Warning: {missing:,} subset rows had no matching frozen vector and will be dropped.", flush=True)
    merged = merged.dropna(subset=vector_cols, how="all").sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return merged


def make_classifier(kind: str, seed: int):
    if kind == "mlp":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(512, 128),
                alpha=1e-4,
                batch_size=128,
                early_stopping=True,
                max_iter=500,
                random_state=seed,
                verbose=True,
            )),
        ])
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
            verbose=1,
        )
    if kind == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed.")
        return XGBClassifier(
            n_estimators=700,
            max_depth=4,
            learning_rate=0.035,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=3,
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            verbosity=1,
        )
    raise ValueError(f"Unknown classifier kind: {kind}")


def fit_classifier(model, kind: str, X_train, y_train, X_val=None, y_val=None):
    if kind == "xgboost" and X_val is not None and y_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
    else:
        model.fit(X_train, y_train)
    return model


def predict_proba_fixed(model, X: np.ndarray) -> np.ndarray:
    proba = np.asarray(model.predict_proba(X), dtype=float)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = getattr(model.named_steps.get("model"), "classes_", None)
    if classes is None:
        return proba
    fixed = np.zeros((len(X), len(CLASS_LABELS)), dtype=float)
    for j, cls in enumerate(classes):
        cls = int(cls)
        if cls in CLASS_LABELS:
            fixed[:, CLASS_LABELS.index(cls)] = proba[:, j]
    return fixed


def classification_metrics(y_true, y_pred, proba) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    try:
        out["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, proba, labels=CLASS_LABELS, multi_class="ovr", average="macro"))
    except Exception:
        out["roc_auc_ovr_macro"] = np.nan
    try:
        y_bin = label_binarize(y_true, classes=CLASS_LABELS)
        out["average_precision_macro"] = float(average_precision_score(y_bin, proba, average="macro"))
    except Exception:
        out["average_precision_macro"] = np.nan
    try:
        out["log_loss"] = float(log_loss(y_true, proba, labels=CLASS_LABELS))
    except Exception:
        out["log_loss"] = np.nan
    return out


def format_metrics(metrics: dict[str, float]) -> str:
    return (
        f"n={metrics.get('n', 0):,.0f} | "
        f"acc={metrics.get('accuracy', np.nan):.3f} | "
        f"bal_acc={metrics.get('balanced_accuracy', np.nan):.3f} | "
        f"macroF1={metrics.get('macro_f1', np.nan):.3f} | "
        f"AUROC={metrics.get('roc_auc_ovr_macro', np.nan):.3f} | "
        f"AUPRC={metrics.get('average_precision_macro', np.nan):.3f} | "
        f"MCC={metrics.get('mcc', np.nan):.3f}"
    )


def id_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in ["record_id", "study_id", "lnp_id", "il_id", "il_smiles", "canonical_il_smiles", "delivery", "delivery_class_label"] if c in frame.columns]


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    trainval_subset = read_table(args.trainval_subset)
    test_subset = read_table(args.test_subset)
    for name, frame in [("trainval", trainval_subset), ("untouched_test", test_subset)]:
        frame[args.target_col] = pd.to_numeric(frame[args.target_col], errors="coerce")
        before = len(frame)
        frame.dropna(subset=[args.target_col], inplace=True)
        frame[args.target_col] = frame[args.target_col].astype(int)
        print(f"{name}: {before:,} -> {len(frame):,} rows with class labels", flush=True)

    manifest_rows = []
    all_metric_rows = []

    for encoder in args.encoders:
        path = encoder_file(args.encoded_dir, encoder)
        display_name = PRETTY.get(encoder, encoder)
        method_name = f"{display_name}-Frozen"
        print("", flush=True)
        print("=" * 80, flush=True)
        print(method_name, flush=True)
        print("=" * 80, flush=True)

        if not path.exists():
            print(f"Skipped: missing encoded vectors -> {path}", flush=True)
            manifest_rows.append({"method": method_name, "encoder": encoder, "status": "missing_vectors", "needed_file": str(path)})
            continue

        encoded = read_table(path)
        trainval = align_vectors_to_subset(encoded, trainval_subset, id_col=args.id_col, target_col=args.target_col)
        test = align_vectors_to_subset(encoded, test_subset, id_col=args.id_col, target_col=args.target_col)
        vec_cols = vector_columns(encoded)

        X = trainval[vec_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        y = trainval[args.target_col].to_numpy(dtype=int)
        X_test = test[vec_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        y_test = test[args.target_col].to_numpy(dtype=int)

        ok = np.isfinite(X).all(axis=1)
        test_ok = np.isfinite(X_test).all(axis=1)
        X, y, trainval = X[ok], y[ok], trainval.loc[ok].reset_index(drop=True)
        X_test, y_test, test = X_test[test_ok], y_test[test_ok], test.loc[test_ok].reset_index(drop=True)

        print(f"Vector file: {path}", flush=True)
        print(f"Feature matrix: trainval={X.shape[0]:,} x {X.shape[1]:,}; test={X_test.shape[0]:,} x {X_test.shape[1]:,}", flush=True)
        print("Trainval class counts:", dict(pd.Series(y).value_counts().sort_index()), flush=True)
        print("Test class counts:", dict(pd.Series(y_test).value_counts().sort_index()), flush=True)

        out_dir = ensure_dir(Path(args.output_dir) / safe_name(method_name))
        cv_predictions = []
        metric_rows = []
        folds = sorted(int(f) for f in trainval["cv_fold"].dropna().unique() if int(f) >= 0)

        for fold in folds:
            val_idx = np.where(trainval["cv_fold"].astype(int).to_numpy() == fold)[0]
            train_idx = np.setdiff1d(np.arange(len(trainval)), val_idx)
            validation_study = ", ".join(sorted(trainval.iloc[val_idx]["study_id"].astype(str).unique()))
            print("", flush=True)
            print(f"{method_name} | fold {fold}: validation study = {validation_study}", flush=True)
            print(f"{method_name} | fold {fold}: train={len(train_idx):,}, val={len(val_idx):,}", flush=True)
            print("  train classes:", dict(pd.Series(y[train_idx]).value_counts().sort_index()), flush=True)
            print("  val classes:", dict(pd.Series(y[val_idx]).value_counts().sort_index()), flush=True)
            print(f"{method_name} | fold {fold}: fitting...", flush=True)

            model = make_classifier(args.estimator, args.seed + fold)
            fit_classifier(model, args.estimator, X[train_idx], y[train_idx], X[val_idx], y[val_idx])
            proba = predict_proba_fixed(model, X[val_idx])
            pred = np.argmax(proba, axis=1)
            metrics = classification_metrics(y[val_idx], pred, proba)
            metrics.update({"method": method_name, "encoder": encoder, "split": f"fold_{fold}", "fold": fold, "validation_study": validation_study})
            metric_rows.append(metrics)
            all_metric_rows.append(metrics)
            print(f"{method_name} | fold_{fold}: {format_metrics(metrics)}", flush=True)

            part = trainval.iloc[val_idx][id_columns(trainval)].copy()
            part["split"] = f"fold_{fold}"
            part["cv_fold"] = fold
            part["y_true"] = y[val_idx]
            part["y_pred"] = pred
            for cls in CLASS_LABELS:
                part[f"prob_{CLASS_NAMES[cls]}"] = proba[:, cls]
            cv_predictions.append(part)

        fold_metrics = pd.DataFrame(metric_rows)
        cv_mean = fold_metrics.drop(columns=["method", "encoder", "split", "fold", "validation_study"], errors="ignore").mean(numeric_only=True).to_dict()
        cv_mean.update({"method": method_name, "encoder": encoder, "split": "cv_mean", "fold": -1, "validation_study": "leave_one_study_out"})
        cv_std = fold_metrics.drop(columns=["method", "encoder", "split", "fold", "validation_study"], errors="ignore").std(numeric_only=True).to_dict()
        cv_std.update({"method": method_name, "encoder": encoder, "split": "cv_std", "fold": -1, "validation_study": "leave_one_study_out"})
        metric_rows.extend([cv_mean, cv_std])
        all_metric_rows.extend([cv_mean, cv_std])
        print("", flush=True)
        print(f"{method_name} | cv_mean: {format_metrics(cv_mean)}", flush=True)

        print(f"{method_name}: fitting final frozen-vector classifier on all trainval rows...", flush=True)
        final_model = make_classifier(args.estimator, args.seed)
        fit_classifier(final_model, args.estimator, X, y)
        print(f"{method_name}: final model fit complete", flush=True)

        test_proba = predict_proba_fixed(final_model, X_test)
        test_pred = np.argmax(test_proba, axis=1)
        test_metrics = classification_metrics(y_test, test_pred, test_proba)
        test_metrics.update({"method": method_name, "encoder": encoder, "split": "untouched_test", "fold": -1, "validation_study": ", ".join(sorted(test["study_id"].astype(str).unique()))})
        metric_rows.append(test_metrics)
        all_metric_rows.append(test_metrics)
        print(f"{method_name} | untouched_test: {format_metrics(test_metrics)}", flush=True)

        test_pred_df = test[id_columns(test)].copy()
        test_pred_df["split"] = "untouched_test"
        test_pred_df["y_true"] = y_test
        test_pred_df["y_pred"] = test_pred
        for cls in CLASS_LABELS:
            test_pred_df[f"prob_{CLASS_NAMES[cls]}"] = test_proba[:, cls]

        cm = confusion_matrix(y_test, test_pred, labels=CLASS_LABELS)
        cm_df = pd.DataFrame(cm, index=[f"true_{CLASS_NAMES[c]}" for c in CLASS_LABELS], columns=[f"pred_{CLASS_NAMES[c]}" for c in CLASS_LABELS])
        per_class = precision_recall_fscore_support(y_test, test_pred, labels=CLASS_LABELS, zero_division=0)
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
        write_json(out_dir / "feature_schema.json", {"encoder": encoder, "feature_columns": vec_cols, "n_features": len(vec_cols)})
        write_json(out_dir / "metadata.json", {
            "method": method_name,
            "encoder": encoder,
            "encoder_trainable": False,
            "classifier": args.estimator,
            "target": args.target_col,
            "class_names": CLASS_NAMES,
            "trainval_path": str(args.trainval_subset),
            "untouched_test_path": str(args.test_subset),
            "encoded_vectors": str(path),
            "cv": "leave-one-study-out CV on balanced trainval data",
            "untouched_test_used_for_training": False,
            "trained_at_utc": now_utc(),
            "output_dir": str(out_dir),
        })
        print(f"{method_name}: saved outputs -> {out_dir}", flush=True)

        manifest_rows.append({
            "method": method_name,
            "encoder": encoder,
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
    manifest_path = Path(args.output_dir) / "frozen_embedding_classification_manifest.csv"
    all_metrics_path = Path(args.output_dir) / "frozen_embedding_all_metrics.csv"
    manifest.to_csv(manifest_path, index=False)
    all_metrics.to_csv(all_metrics_path, index=False)

    print("", flush=True)
    print("Saved frozen-embedding classification manifest:", manifest_path, flush=True)
    print("Saved all frozen-embedding metrics:", all_metrics_path, flush=True)
    print("", flush=True)
    print(manifest.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
