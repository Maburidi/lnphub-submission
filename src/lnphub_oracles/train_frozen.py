from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .io_utils import ID_COLS, read_table, vector_columns
from .sklearn_train import save_manifest, train_sklearn_oracle


DEFAULT_ENCODERS = ["AGILE", "KPGT", "LiON", "ChemBERTa_MTR", "MoLFormer", "MolCLR_GIN", "Uni_Mol"]
PRETTY = {
    "ChemBERTa_MTR": "ChemBERTa-MTR",
    "MolCLR_GIN": "MolCLR-GIN",
    "Uni_Mol": "Uni-Mol",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frozen-embedding LNPHub oracles.")
    parser.add_argument("--encoded-dir", required=True, type=Path)
    parser.add_argument("--test-encoded-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-col", default="delivery")
    parser.add_argument("--encoders", nargs="+", default=DEFAULT_ENCODERS)
    parser.add_argument("--estimator", default="mlp", choices=["mlp", "xgboost", "random_forest"])
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def encoder_file(encoded_dir: Path, encoder: str) -> Path:
    return encoded_dir / f"{encoder}_encoder_vectors.parquet"


def main() -> None:
    args = parse_args()
    rows = []
    for encoder in args.encoders:
        path = encoder_file(args.encoded_dir, encoder)
        if not path.exists():
            print(f"\n=== {encoder}-Frozen ===")
            print(f"Skipped: missing {path}")
            continue
        frame = read_table(path)
        vec_cols = vector_columns(frame)
        X = frame[vec_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        y = pd.to_numeric(frame[args.target_col], errors="coerce").to_numpy(dtype=np.float32)
        test_frame = None
        X_test = None
        y_test = None
        test_path = encoder_file(args.test_encoded_dir, encoder) if args.test_encoded_dir else None
        if test_path and test_path.exists():
            test_frame = read_table(test_path)
            missing_vec = [c for c in vec_cols if c not in test_frame.columns]
            if missing_vec:
                raise ValueError(f"{encoder}: test encoded file is missing vector columns.")
            X_test = test_frame[vec_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
            y_test = pd.to_numeric(test_frame[args.target_col], errors="coerce").to_numpy(dtype=np.float32)
        display_name = PRETTY.get(encoder, encoder)
        method_name = f"{display_name}-Frozen"
        print(f"\n=== {method_name} ===")
        rows.append(train_sklearn_oracle(
            method_name=method_name,
            X=X,
            y=y,
            frame=frame,
            feature_columns=vec_cols,
            estimator_kind=args.estimator,
            output_dir=args.output_dir,
            target_col=args.target_col,
            seed=args.seed,
            metadata={"input_trainval_vectors": str(path), "input_untouched_test_vectors": str(test_path) if test_path else None, "encoder": encoder, "encoder_trainable": False},
            X_test=X_test,
            y_test=y_test,
            test_frame=test_frame,
        ))
    manifest = save_manifest(rows, args.output_dir, "frozen_training_manifest.csv")
    print("\nSaved frozen-embedding manifest:")
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
