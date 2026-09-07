from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .io_utils import ensure_dir, read_table, write_json
from .splits import assign_untouched_test_and_cv_folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create untouched test set and 5-fold CV trainval splits.")
    parser.add_argument("--subset", required=True, type=Path)
    parser.add_argument("--encoded-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split-name", default="in_vitro_expression")
    parser.add_argument("--id-col", default="record_id")
    parser.add_argument("--group-col", default="study_id")
    parser.add_argument("--target-col", default="delivery")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--test-groups", nargs="*", default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    enc_trainval_dir = ensure_dir(out_dir / "encoded_trainval")
    enc_test_dir = ensure_dir(out_dir / "encoded_untouched_test")

    frame = read_table(args.subset).reset_index(drop=True)
    required = [args.id_col, args.group_col, args.target_col]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    assignment = assign_untouched_test_and_cv_folds(
        frame,
        seed=args.seed,
        group_col=args.group_col,
        test_size=args.test_size,
        n_folds=args.n_folds,
        test_groups=args.test_groups,
    )
    assignment[args.id_col] = frame[args.id_col].astype(str).to_numpy()
    assignment[args.group_col] = frame[args.group_col].astype(str).to_numpy()

    full = frame.copy()
    full["oracle_split"] = assignment["split"].to_numpy()
    full["cv_fold"] = assignment["cv_fold"].to_numpy()
    trainval = full[full["oracle_split"].eq("trainval")].sample(frac=1, random_state=args.seed + 101).reset_index(drop=True)
    test = full[full["oracle_split"].eq("untouched_test")].sample(frac=1, random_state=args.seed + 202).reset_index(drop=True)

    prefix = f"lnphub_oracle_{args.split_name}"
    trainval_path = out_dir / f"{prefix}_trainval_5fold.csv"
    test_path = out_dir / f"{prefix}_untouched_test.csv"
    assignment_path = out_dir / f"{prefix}_split_assignments.csv"
    trainval.to_csv(trainval_path, index=False)
    test.to_csv(test_path, index=False)
    assignment.to_csv(assignment_path, index=False)

    train_ids = set(trainval[args.id_col].astype(str))
    test_ids = set(test[args.id_col].astype(str))
    encoded_rows = []
    for path in sorted(args.encoded_dir.glob("*_encoder_vectors.parquet")):
        enc = pd.read_parquet(path)
        if args.id_col not in enc.columns:
            print(f"Skipped {path.name}: missing {args.id_col}")
            continue
        enc2 = enc.copy()
        enc2[args.id_col] = enc2[args.id_col].astype(str)
        enc2 = enc2.merge(
            assignment[[args.id_col, "split", "cv_fold"]].rename(columns={"split": "oracle_split"}),
            on=args.id_col,
            how="inner",
        )
        enc_ids = enc2[args.id_col].astype(str)
        enc_train = enc2.loc[enc_ids.isin(train_ids)].sample(frac=1, random_state=args.seed + 301).reset_index(drop=True)
        enc_test = enc2.loc[enc_ids.isin(test_ids)].sample(frac=1, random_state=args.seed + 401).reset_index(drop=True)
        train_path = enc_trainval_dir / path.name
        test_vec_path = enc_test_dir / path.name
        enc_train.to_parquet(train_path, index=False)
        enc_test.to_parquet(test_vec_path, index=False)
        encoded_rows.append({
            "encoder_file": path.name,
            "input_rows": len(enc),
            "trainval_rows": len(enc_train),
            "untouched_test_rows": len(enc_test),
            "trainval_file": str(train_path),
            "untouched_test_file": str(test_vec_path),
        })

    encoded_summary = pd.DataFrame(encoded_rows)
    encoded_summary_path = out_dir / f"{prefix}_encoded_split_summary.csv"
    encoded_summary.to_csv(encoded_summary_path, index=False)

    split_summary = {
        "input_subset": str(args.subset),
        "input_encoded_dir": str(args.encoded_dir),
        "split_name": args.split_name,
        "id_col": args.id_col,
        "group_col": args.group_col,
        "target_col": args.target_col,
        "test_size": args.test_size,
        "test_groups": args.test_groups or [],
        "n_folds": args.n_folds,
        "seed": args.seed,
        "rows_shuffled_after_split": True,
        "rows_total": len(full),
        "rows_trainval": len(trainval),
        "rows_untouched_test": len(test),
        "groups_trainval": trainval[args.group_col].nunique(dropna=True),
        "groups_untouched_test": test[args.group_col].nunique(dropna=True),
        "trainval_file": str(trainval_path),
        "untouched_test_file": str(test_path),
        "assignment_file": str(assignment_path),
        "encoded_trainval_dir": str(enc_trainval_dir),
        "encoded_untouched_test_dir": str(enc_test_dir),
    }
    write_json(out_dir / f"{prefix}_split_metadata.json", split_summary)

    print("Saved trainval:", trainval_path)
    print("Saved untouched test:", test_path)
    print("Saved assignments:", assignment_path)
    print("Saved encoded split summary:", encoded_summary_path)
    print("\nSplit summary:")
    print(pd.DataFrame([split_summary]).to_string(index=False))
    print("\nEncoded files:")
    print(encoded_summary.to_string(index=False))
    print("\nCV fold sizes:")
    print(trainval["cv_fold"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
