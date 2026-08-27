from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .io_utils import ensure_dir, method_dir, now_utc, write_json
from .metrics import regression_metrics
from .splits import cv_fold_indices


def make_estimator(kind: str, seed: int):
    kind = kind.lower()
    if kind == "mlp":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(hidden_layer_sizes=(512, 128), alpha=1e-4, batch_size=128, early_stopping=True, max_iter=500, random_state=seed)),
        ])
    if kind == "random_forest":
        return RandomForestRegressor(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    if kind == "xgboost":
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(
                n_estimators=700,
                max_depth=4,
                learning_rate=0.035,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=seed,
                n_jobs=-1,
            )
        except Exception:
            from sklearn.ensemble import HistGradientBoostingRegressor
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingRegressor(max_iter=700, learning_rate=0.035, random_state=seed)),
            ])
    raise ValueError(f"Unknown sklearn estimator kind: {kind}")


def train_sklearn_oracle(
    *,
    method_name: str,
    X: np.ndarray,
    y: np.ndarray,
    frame: pd.DataFrame,
    feature_columns: list[str],
    estimator_kind: str,
    output_dir: str | Path,
    target_col: str = "delivery",
    seed: int = 7,
    metadata: dict | None = None,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    test_frame: pd.DataFrame | None = None,
    fold_col: str = "cv_fold",
) -> dict:
    out_dir = method_dir(output_dir, method_name)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y = X[ok], y[ok]
    frame = frame.loc[ok].reset_index(drop=True)

    cv_predictions = []
    fold_metrics = {}
    for fold, train_idx, val_idx in cv_fold_indices(frame, fold_col=fold_col):
        model = make_estimator(estimator_kind, seed + fold)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[val_idx])
        split = f"fold_{fold}"
        fold_metrics[split] = regression_metrics(y[val_idx], pred)
        part = frame.iloc[val_idx][[c for c in ["record_id", "study_id", "lnp_id", "il_id", "il_smiles"] if c in frame.columns]].copy()
        part["split"] = split
        part["cv_fold"] = fold
        part["y_true"] = y[val_idx]
        part["y_pred"] = pred
        cv_predictions.append(part)

    final_model = make_estimator(estimator_kind, seed)
    final_model.fit(X, y)

    test_metrics = None
    if X_test is not None and y_test is not None and test_frame is not None:
        X_test = np.asarray(X_test, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.float32)
        test_ok = np.isfinite(y_test) & np.isfinite(X_test).all(axis=1)
        X_test = X_test[test_ok]
        y_test = y_test[test_ok]
        test_frame = test_frame.loc[test_ok].reset_index(drop=True)
        pred = final_model.predict(X_test)
        test_metrics = regression_metrics(y_test, pred)
        part = test_frame[[c for c in ["record_id", "study_id", "lnp_id", "il_id", "il_smiles"] if c in test_frame.columns]].copy()
        part["split"] = "untouched_test"
        part["y_true"] = y_test
        part["y_pred"] = pred
        part.to_csv(out_dir / "untouched_test_predictions.csv", index=False)

    all_metrics = dict(fold_metrics)
    if test_metrics is not None:
        all_metrics["untouched_test"] = test_metrics

    metric_rows = [{"split": k, **v} for k, v in all_metrics.items()]
    fold_df = pd.DataFrame([v for k, v in fold_metrics.items()])
    for stat_name, values in [("cv_mean", fold_df.mean(numeric_only=True)), ("cv_std", fold_df.std(numeric_only=True))]:
        metric_rows.append({"split": stat_name, **values.to_dict()})

    joblib.dump(final_model, out_dir / "model.joblib")
    pd.concat(cv_predictions, ignore_index=True).to_csv(out_dir / "cv_predictions.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(out_dir / "metrics.csv", index=False)
    write_json(out_dir / "metrics.json", all_metrics)
    write_json(out_dir / "feature_schema.json", {"feature_columns": feature_columns, "n_features": len(feature_columns)})

    meta = {
        "method_name": method_name,
        "estimator_kind": estimator_kind,
        "target_col": target_col,
        "n_trainval_rows": int(len(y)),
        "n_test_rows": int(len(y_test)) if y_test is not None else 0,
        "n_features": int(X.shape[1]),
        "cv": "5-fold cross-validation on trainval split",
        "untouched_test_used_for_training": False,
        "trained_at_utc": now_utc(),
        "model_file": str(out_dir / "model.joblib"),
        "metadata": metadata or {},
    }
    write_json(out_dir / "metadata.json", meta)
    out = {"method": method_name, "output_dir": str(out_dir)}
    cv_mean = next((row for row in metric_rows if row["split"] == "cv_mean"), {})
    for k, v in cv_mean.items():
        if k != "split":
            out[f"cv_mean_{k}"] = v
    if test_metrics is not None:
        out.update({f"untouched_test_{k}": v for k, v in test_metrics.items()})
    return out


def save_manifest(rows: list[dict], output_dir: str | Path, filename: str = "training_manifest.csv") -> pd.DataFrame:
    ensure_dir(output_dir)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(Path(output_dir) / filename, index=False)
    return manifest
