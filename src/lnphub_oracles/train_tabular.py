from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .features import make_ecfp4, make_rdkit_descriptors, smiles_series, valid_smiles_mask
from .io_utils import read_table
from .sklearn_train import save_manifest, train_sklearn_oracle


METHODS = {
    "ecfp4_mlp": ("ECFP4 + MLP", "ecfp4", "mlp"),
    "ecfp4_xgboost": ("ECFP4 + XGBoost", "ecfp4", "xgboost"),
    "ecfp4_random_forest": ("ECFP4 + Random Forest", "ecfp4", "random_forest"),
    "rdkit_descriptors_mlp": ("RDKit Descriptors + MLP", "rdkit_descriptors", "mlp"),
    "rdkit_descriptors_xgboost": ("RDKit Descriptors + XGBoost", "rdkit_descriptors", "xgboost"),
    "rdkit_descriptors_random_forest": ("RDKit Descriptors + Random Forest", "rdkit_descriptors", "random_forest"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ECFP4/RDKit-descriptor LNPHub oracles.")
    parser.add_argument("--subset", required=True, type=Path)
    parser.add_argument("--test-subset", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-col", default="delivery")
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = read_table(args.subset)
    y = pd.to_numeric(frame[args.target_col], errors="coerce").to_numpy(dtype=np.float32)
    smiles = smiles_series(frame)
    ok = np.isfinite(y) & valid_smiles_mask(smiles)
    frame = frame.loc[ok].reset_index(drop=True)
    y = y[ok]
    smiles = smiles.loc[ok].reset_index(drop=True)
    test_frame = None
    test_y = None
    test_smiles = None
    if args.test_subset:
        test_frame = read_table(args.test_subset)
        test_y = pd.to_numeric(test_frame[args.target_col], errors="coerce").to_numpy(dtype=np.float32)
        test_smiles = smiles_series(test_frame)
        test_ok = np.isfinite(test_y) & valid_smiles_mask(test_smiles)
        test_frame = test_frame.loc[test_ok].reset_index(drop=True)
        test_y = test_y[test_ok]
        test_smiles = test_smiles.loc[test_ok].reset_index(drop=True)

    print(f"Rows used: {len(frame):,}")
    if test_frame is not None:
        print(f"Untouched test rows: {len(test_frame):,}")
    cache: dict[str, tuple[np.ndarray, list[str]]] = {}
    test_cache: dict[str, tuple[np.ndarray, list[str]]] = {}
    rows = []
    for key in args.methods:
        if key not in METHODS:
            raise ValueError(f"Unknown method: {key}. Valid methods: {sorted(METHODS)}")
        method_name, feature_type, estimator_kind = METHODS[key]
        print(f"\n=== {method_name} ===")
        if feature_type not in cache:
            cache[feature_type] = make_ecfp4(smiles) if feature_type == "ecfp4" else make_rdkit_descriptors(smiles)
        X, feature_columns = cache[feature_type]
        X_test = None
        if test_smiles is not None:
            if feature_type not in test_cache:
                test_cache[feature_type] = make_ecfp4(test_smiles) if feature_type == "ecfp4" else make_rdkit_descriptors(test_smiles)
            X_test, test_feature_columns = test_cache[feature_type]
            if test_feature_columns != feature_columns:
                raise ValueError(f"Feature mismatch for {method_name}")
        rows.append(train_sklearn_oracle(
            method_name=method_name,
            X=X,
            y=y,
            frame=frame,
            feature_columns=feature_columns,
            estimator_kind=estimator_kind,
            output_dir=args.output_dir,
            target_col=args.target_col,
            seed=args.seed,
            metadata={"input_trainval_subset": str(args.subset), "input_untouched_test_subset": str(args.test_subset) if args.test_subset else None, "feature_type": feature_type},
            X_test=X_test,
            y_test=test_y,
            test_frame=test_frame,
        ))
    manifest = save_manifest(rows, args.output_dir, "tabular_training_manifest.csv")
    print("\nSaved tabular manifest:")
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
