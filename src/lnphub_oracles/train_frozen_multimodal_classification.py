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
    DEFAULT_ENCODERS,
    PRETTY,
    classification_metrics,
    fit_classifier,
    format_metrics,
    id_columns,
    make_classifier,
    predict_proba_fixed,
    safe_name,
    vector_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frozen-pretrained 3-class classifiers with optional extra data layers.")
    parser.add_argument("--trainval-subset", type=Path)
    parser.add_argument("--train-subset", type=Path)
    parser.add_argument("--validation-subset", type=Path)
    parser.add_argument("--test-subset", required=True, type=Path)
    parser.add_argument("--encoded-dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-col", default="delivery_class_3")
    parser.add_argument("--id-col", default="record_id")
    parser.add_argument("--encoders", nargs="+", default=DEFAULT_ENCODERS)
    parser.add_argument("--tabular-cols", nargs="*", default=[])
    parser.add_argument("--estimator", default="mlp", choices=["mlp", "xgboost", "random_forest"])
    parser.add_argument(
        "--require-complete-coverage",
        action="store_true",
        help="Fail instead of dropping subset records that do not have a matching frozen vector.",
    )
    parser.add_argument("--cv-description", default="random stratified 5-fold CV on balanced trainval data")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random_mode = args.train_subset is not None or args.validation_subset is not None
    cv_mode = args.trainval_subset is not None
    if random_mode == cv_mode:
        raise ValueError("Use either --trainval-subset for CV mode, or --train-subset plus --validation-subset for random-split mode.")
    if random_mode and (args.train_subset is None or args.validation_subset is None):
        raise ValueError("Random-split mode requires both --train-subset and --validation-subset.")
    return args


def encoder_file(encoded_dir: Path, encoder: str) -> Path:
    return encoded_dir / f"{encoder}_encoder_vectors.parquet"


def load_encoder_vectors(encoded_dirs: list[Path], encoder: str, id_col: str) -> tuple[pd.DataFrame | None, list[Path]]:
    frames = []
    paths = []
    for encoded_dir in encoded_dirs:
        path = encoder_file(encoded_dir, encoder)
        if not path.exists():
            continue
        frame = read_table(path)
        if id_col not in frame.columns:
            raise ValueError(f"{path} is missing id column: {id_col}")
        frames.append(frame)
        paths.append(path)
    if not frames:
        return None, []
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)
    return out, paths


def clean_subset(frame: pd.DataFrame, target_col: str, split_name: str) -> pd.DataFrame:
    frame = frame.copy()
    if target_col not in frame.columns:
        raise ValueError(f"{split_name} subset is missing target column: {target_col}")
    frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce")
    before = len(frame)
    frame = frame[frame[target_col].isin(CLASS_LABELS)].copy()
    frame[target_col] = frame[target_col].astype(int)
    print(f"{split_name}: {before:,} -> {len(frame):,} rows with usable class labels", flush=True)
    print(f"{split_name} class counts: {dict(frame[target_col].value_counts().sort_index())}", flush=True)
    return frame.reset_index(drop=True)


def align_features(
    encoded: pd.DataFrame,
    subset: pd.DataFrame,
    *,
    id_col: str,
    target_col: str,
    tabular_cols: list[str],
    require_complete_coverage: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], list[str]]:
    if id_col not in subset.columns:
        raise ValueError(f"Subset is missing id column: {id_col}")
    missing_tabular = [c for c in tabular_cols if c not in subset.columns]
    if missing_tabular:
        raise ValueError(f"Subset is missing tabular column(s): {missing_tabular}")

    vec_cols = vector_columns(encoded)
    keep = [id_col, target_col, "cv_fold", "oracle_split", "delivery_class_label"]
    keep += [c for c in ["study_id", "lnp_id", "il_id", "il_smiles", "canonical_il_smiles", "delivery"] if c in subset.columns]
    keep += [c for c in tabular_cols if c in subset.columns]
    labels = subset[[c for c in keep if c in subset.columns]].copy()
    labels["_order"] = np.arange(len(labels))

    merged = labels.merge(encoded[[id_col] + vec_cols], on=id_col, how="left", validate="many_to_one")
    missing = int(merged[vec_cols].isna().all(axis=1).sum())
    if missing:
        if require_complete_coverage:
            examples = merged.loc[merged[vec_cols].isna().all(axis=1), id_col].head(10).tolist()
            raise ValueError(
                f"Frozen-vector coverage is incomplete: {missing:,}/{len(subset):,} records are missing "
                f"for this subset. Example {id_col} values: {examples}"
            )
        print(f"Warning: dropping {missing:,} rows with no matching frozen vector.", flush=True)
    merged = merged.dropna(subset=vec_cols, how="all").sort_values("_order").drop(columns="_order").reset_index(drop=True)

    X_vec_df = merged[vec_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    feature_cols = list(vec_cols)
    if tabular_cols:
        X_tab_df = merged[tabular_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        X_vec = X_vec_df.to_numpy(dtype=np.float32)
        X_tab = X_tab_df.to_numpy(dtype=np.float32)
        X = np.hstack([X_vec, X_tab])
        feature_cols += list(tabular_cols)
    else:
        X = X_vec_df.to_numpy(dtype=np.float32)
    y = merged[target_col].to_numpy(dtype=int)
    rows_all_missing = np.isnan(X).all(axis=1)
    dropped = int(rows_all_missing.sum())
    if dropped:
        print(f"Warning: dropping {dropped:,} rows with all features missing after vector/tabular merge.", flush=True)
    X = X[~rows_all_missing]
    y = y[~rows_all_missing]
    merged = merged.loc[~rows_all_missing].reset_index(drop=True)
    nonfinite_fraction = float((~np.isfinite(X)).mean()) if X.size else 0.0
    if nonfinite_fraction:
        print(f"Non-finite feature entries retained for imputation: {nonfinite_fraction:.3%}", flush=True)
    return merged, X, y, vec_cols, feature_cols


def evaluate_split(model, X, y, frame, split_name: str):
    proba = predict_proba_fixed(model, X)
    row_sums = proba.sum(axis=1, keepdims=True)
    proba = np.divide(proba, row_sums, out=np.full_like(proba, 1.0 / proba.shape[1]), where=row_sums > 0)
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


def save_common_outputs(
    *,
    out_dir: Path,
    model,
    metric_rows: list[dict],
    vector_cols: list[str],
    feature_cols: list[str],
    tabular_cols: list[str],
    encoder: str,
    encoded_paths: list[Path],
    args,
    method_name: str,
    split_strategy: str,
    extra_tables: dict[str, pd.DataFrame],
):
    joblib.dump(model, out_dir / "model.joblib")
    for name, table in extra_tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=("confusion_matrix" in name))
    pd.DataFrame(metric_rows).to_csv(out_dir / "metrics.csv", index=False)
    write_json(out_dir / "metrics.json", {row["split"]: row for row in metric_rows})
    write_json(out_dir / "feature_schema.json", {
        "encoder": encoder,
        "frozen_embedding_columns": vector_cols,
        "tabular_columns": tabular_cols,
        "feature_columns": feature_cols,
        "n_frozen_embedding_features": len(vector_cols),
        "n_tabular_features": len(tabular_cols),
        "n_features": len(feature_cols),
    })
    write_json(out_dir / "metadata.json", {
        "method": method_name,
        "encoder": encoder,
        "encoder_trainable": False,
        "classifier": args.estimator,
        "target": args.target_col,
        "class_names": CLASS_NAMES,
        "trainval_path": str(args.trainval_subset) if args.trainval_subset else None,
        "train_path": str(args.train_subset) if args.train_subset else None,
        "validation_path": str(args.validation_subset) if args.validation_subset else None,
        "test_path": str(args.test_subset),
        "encoded_vector_files": [str(p) for p in encoded_paths],
        "split_strategy": split_strategy,
        "test_used_for_training": False,
        "trained_at_utc": now_utc(),
        "output_dir": str(out_dir),
    })


def run_random(args) -> pd.DataFrame:
    train_subset = clean_subset(read_table(args.train_subset), args.target_col, "train")
    validation_subset = clean_subset(read_table(args.validation_subset), args.target_col, "validation")
    test_subset = clean_subset(read_table(args.test_subset), args.target_col, "test")
    manifest_rows = []

    for encoder in args.encoders:
        display_name = PRETTY.get(encoder, encoder)
        method_name = f"{display_name}-Frozen"
        random_method_name = f"{method_name} Random Split"
        print("\n" + "=" * 80, flush=True)
        print(random_method_name, flush=True)
        print("=" * 80, flush=True)

        encoded, paths = load_encoder_vectors(args.encoded_dirs, encoder, args.id_col)
        if encoded is None:
            print(f"Skipped: no vector file found for {encoder} in {[str(p) for p in args.encoded_dirs]}", flush=True)
            manifest_rows.append({"method": random_method_name, "encoder": encoder, "status": "missing_vectors"})
            continue

        train, X_train, y_train, vector_cols, feature_cols = align_features(
            encoded, train_subset, id_col=args.id_col, target_col=args.target_col,
            tabular_cols=args.tabular_cols, require_complete_coverage=args.require_complete_coverage,
        )
        validation, X_val, y_val, _, val_feature_cols = align_features(
            encoded, validation_subset, id_col=args.id_col, target_col=args.target_col,
            tabular_cols=args.tabular_cols, require_complete_coverage=args.require_complete_coverage,
        )
        test, X_test, y_test, _, test_feature_cols = align_features(
            encoded, test_subset, id_col=args.id_col, target_col=args.target_col,
            tabular_cols=args.tabular_cols, require_complete_coverage=args.require_complete_coverage,
        )
        if feature_cols != val_feature_cols or feature_cols != test_feature_cols:
            raise ValueError(f"Feature mismatch for {random_method_name}")

        print(f"Vector files: {', '.join(str(p) for p in paths)}", flush=True)
        print(
            f"Feature matrix: train={X_train.shape[0]:,} x {X_train.shape[1]:,}; "
            f"validation={X_val.shape[0]:,}; test={X_test.shape[0]:,}",
            flush=True,
        )
        print(f"Frozen embedding features: {len(vector_cols):,}; appended extra-layer features: {len(args.tabular_cols):,}", flush=True)
        print("Train classes:", dict(pd.Series(y_train).value_counts().sort_index()), flush=True)
        print("Validation classes:", dict(pd.Series(y_val).value_counts().sort_index()), flush=True)
        print("Test classes:", dict(pd.Series(y_test).value_counts().sort_index()), flush=True)

        model = make_classifier(args.estimator, args.seed)
        print(f"{random_method_name}: fitting...", flush=True)
        fit_classifier(model, args.estimator, X_train, y_train, X_val, y_val)
        print(f"{random_method_name}: fit complete", flush=True)

        val_metrics, val_pred, val_cm, val_per_class = evaluate_split(model, X_val, y_val, validation, "validation")
        test_metrics, test_pred, test_cm, test_per_class = evaluate_split(model, X_test, y_test, test, "test")
        val_metrics.update({"method": random_method_name, "encoder": encoder})
        test_metrics.update({"method": random_method_name, "encoder": encoder})
        print(f"{random_method_name} | validation: {format_metrics(val_metrics)}", flush=True)
        print(f"{random_method_name} | test: {format_metrics(test_metrics)}", flush=True)

        out_dir = ensure_dir(args.output_dir / safe_name(random_method_name))
        save_common_outputs(
            out_dir=out_dir,
            model=model,
            metric_rows=[val_metrics, test_metrics],
            vector_cols=vector_cols,
            feature_cols=feature_cols,
            tabular_cols=args.tabular_cols,
            encoder=encoder,
            encoded_paths=paths,
            args=args,
            method_name=random_method_name,
            split_strategy="stratified random 80/10/10 split after shuffling",
            extra_tables={
                "validation_predictions": val_pred,
                "test_predictions": test_pred,
                "validation_confusion_matrix": val_cm,
                "test_confusion_matrix": test_cm,
                "validation_per_class_metrics": val_per_class,
                "test_per_class_metrics": test_per_class,
            },
        )
        print(f"{random_method_name}: saved outputs -> {out_dir}", flush=True)
        manifest_rows.append(manifest_row(random_method_name, encoder, out_dir, "trained", val_metrics, test_metrics, "validation"))

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(args.output_dir / "frozen_embedding_random_split_manifest.csv", index=False)
    print("\nSaved frozen random-split manifest:", args.output_dir / "frozen_embedding_random_split_manifest.csv", flush=True)
    print(manifest.to_string(index=False), flush=True)
    return manifest


def run_cv(args) -> pd.DataFrame:
    trainval_subset = clean_subset(read_table(args.trainval_subset), args.target_col, "trainval")
    test_subset = clean_subset(read_table(args.test_subset), args.target_col, "test")
    if "cv_fold" not in trainval_subset.columns:
        raise ValueError("trainval subset is missing cv_fold.")
    folds = sorted(int(f) for f in trainval_subset["cv_fold"].dropna().unique() if int(f) >= 0)
    if not folds:
        raise ValueError("No usable CV folds found.")
    manifest_rows = []
    all_metric_rows = []

    for encoder in args.encoders:
        display_name = PRETTY.get(encoder, encoder)
        method_name = f"{display_name}-Frozen"
        print("\n" + "=" * 80, flush=True)
        print(method_name, flush=True)
        print("=" * 80, flush=True)

        encoded, paths = load_encoder_vectors(args.encoded_dirs, encoder, args.id_col)
        if encoded is None:
            print(f"Skipped: no vector file found for {encoder} in {[str(p) for p in args.encoded_dirs]}", flush=True)
            manifest_rows.append({"method": method_name, "encoder": encoder, "status": "missing_vectors"})
            continue

        trainval, X, y, vector_cols, feature_cols = align_features(
            encoded, trainval_subset, id_col=args.id_col, target_col=args.target_col,
            tabular_cols=args.tabular_cols, require_complete_coverage=args.require_complete_coverage,
        )
        test, X_test, y_test, _, test_feature_cols = align_features(
            encoded, test_subset, id_col=args.id_col, target_col=args.target_col,
            tabular_cols=args.tabular_cols, require_complete_coverage=args.require_complete_coverage,
        )
        if feature_cols != test_feature_cols:
            raise ValueError(f"Feature mismatch for {method_name}")

        print(f"Vector files: {', '.join(str(p) for p in paths)}", flush=True)
        print(f"Feature matrix: trainval={X.shape[0]:,} x {X.shape[1]:,}; test={X_test.shape[0]:,}", flush=True)
        print(f"Frozen embedding features: {len(vector_cols):,}; appended extra-layer features: {len(args.tabular_cols):,}", flush=True)
        print("Trainval classes:", dict(pd.Series(y).value_counts().sort_index()), flush=True)
        print("Test classes:", dict(pd.Series(y_test).value_counts().sort_index()), flush=True)

        out_dir = ensure_dir(args.output_dir / safe_name(method_name))
        cv_predictions = []
        metric_rows = []
        for fold in folds:
            val_idx = np.where(trainval["cv_fold"].astype(int).to_numpy() == fold)[0]
            train_idx = np.setdiff1d(np.arange(len(trainval)), val_idx)
            validation_studies = ", ".join(sorted(trainval.iloc[val_idx]["study_id"].astype(str).unique()))
            print(f"\n{method_name} | fold {fold}: train={len(train_idx):,}, val={len(val_idx):,}; validation studies={validation_studies}", flush=True)
            print("  train classes:", dict(pd.Series(y[train_idx]).value_counts().sort_index()), flush=True)
            print("  val classes:", dict(pd.Series(y[val_idx]).value_counts().sort_index()), flush=True)

            model = make_classifier(args.estimator, args.seed + fold)
            print(f"{method_name} | fold {fold}: fitting...", flush=True)
            fit_classifier(model, args.estimator, X[train_idx], y[train_idx], X[val_idx], y[val_idx])
            metrics, pred_df, _, _ = evaluate_split(model, X[val_idx], y[val_idx], trainval.iloc[val_idx].reset_index(drop=True), f"fold_{fold}")
            metrics.update({"method": method_name, "encoder": encoder, "fold": fold, "validation_study": validation_studies})
            metric_rows.append(metrics)
            all_metric_rows.append(metrics)
            print(f"{method_name} | fold_{fold}: {format_metrics(metrics)}", flush=True)
            pred_df["cv_fold"] = fold
            cv_predictions.append(pred_df)

        fold_metrics = pd.DataFrame(metric_rows)
        cv_mean = fold_metrics.drop(columns=["method", "encoder", "split", "fold", "validation_study"], errors="ignore").mean(numeric_only=True).to_dict()
        cv_mean.update({"method": method_name, "encoder": encoder, "split": "cv_mean", "fold": -1, "validation_study": args.cv_description})
        cv_std = fold_metrics.drop(columns=["method", "encoder", "split", "fold", "validation_study"], errors="ignore").std(numeric_only=True).to_dict()
        cv_std.update({"method": method_name, "encoder": encoder, "split": "cv_std", "fold": -1, "validation_study": args.cv_description})
        metric_rows.extend([cv_mean, cv_std])
        all_metric_rows.extend([cv_mean, cv_std])
        print(f"\n{method_name} | cv_mean: {format_metrics(cv_mean)}", flush=True)

        print(f"{method_name}: fitting final frozen-vector classifier on all trainval rows...", flush=True)
        final_model = make_classifier(args.estimator, args.seed)
        fit_classifier(final_model, args.estimator, X, y)
        print(f"{method_name}: final model fit complete", flush=True)

        test_metrics, test_pred, test_cm, test_per_class = evaluate_split(final_model, X_test, y_test, test, "test")
        test_metrics.update({"method": method_name, "encoder": encoder, "fold": -1, "validation_study": ", ".join(sorted(test["study_id"].astype(str).unique()))})
        metric_rows.append(test_metrics)
        all_metric_rows.append(test_metrics)
        print(f"{method_name} | test: {format_metrics(test_metrics)}", flush=True)

        save_common_outputs(
            out_dir=out_dir,
            model=final_model,
            metric_rows=metric_rows,
            vector_cols=vector_cols,
            feature_cols=feature_cols,
            tabular_cols=args.tabular_cols,
            encoder=encoder,
            encoded_paths=paths,
            args=args,
            method_name=method_name,
            split_strategy=args.cv_description,
            extra_tables={
                "cv_predictions": pd.concat(cv_predictions, ignore_index=True),
                "test_predictions": test_pred,
                "test_confusion_matrix": test_cm,
                "test_per_class_metrics": test_per_class,
            },
        )
        print(f"{method_name}: saved outputs -> {out_dir}", flush=True)
        manifest_rows.append(manifest_row(method_name, encoder, out_dir, "trained", cv_mean, test_metrics, "cv"))

    manifest = pd.DataFrame(manifest_rows)
    all_metrics = pd.DataFrame(all_metric_rows)
    manifest.to_csv(args.output_dir / "frozen_embedding_cv_manifest.csv", index=False)
    all_metrics.to_csv(args.output_dir / "frozen_embedding_all_metrics.csv", index=False)
    print("\nSaved frozen CV manifest:", args.output_dir / "frozen_embedding_cv_manifest.csv", flush=True)
    print("Saved all frozen CV metrics:", args.output_dir / "frozen_embedding_all_metrics.csv", flush=True)
    print(manifest.to_string(index=False), flush=True)
    return manifest


def manifest_row(method_name: str, encoder: str, out_dir: Path, status: str, primary_metrics: dict, test_metrics: dict, primary_prefix: str) -> dict:
    row = {
        "method": method_name,
        "encoder": encoder,
        "status": status,
        "output_dir": str(out_dir),
    }
    for metric in ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc_ovr_macro", "average_precision_macro", "mcc", "log_loss"]:
        row[f"{primary_prefix}_{metric}"] = primary_metrics.get(metric)
        row[f"test_{metric}"] = test_metrics.get(metric)
    return row


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    print("Encoders:", ", ".join(args.encoders), flush=True)
    print("Encoded dirs:", ", ".join(str(p) for p in args.encoded_dirs), flush=True)
    print("Estimator/head:", args.estimator, flush=True)
    print("Require complete vector coverage:", args.require_complete_coverage, flush=True)
    print("Extra-layer columns:", len(args.tabular_cols), flush=True)
    if args.tabular_cols:
        print(", ".join(args.tabular_cols), flush=True)

    if args.trainval_subset is not None:
        run_cv(args)
    else:
        run_random(args)


if __name__ == "__main__":
    main()
