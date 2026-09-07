#!/usr/bin/env python3
"""Collect multimodal random 5-fold CV metrics into comparison tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from lnphub_oracles.cv_comparison import (
    build_formatted_summary_table,
    build_summary_table,
    build_test_results_table,
    collect_cv_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect fold-level, cv_mean, and cv_std rows from the three "
            "multimodal random_5fold_CV result folders."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("."),
        help="Repository/results root. Default: current working directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/comparisons/multimodal_random_5fold_cv"),
        help="Directory where comparison tables will be written.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=3,
        help="Decimal places for the formatted mean +/- std table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = results_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    long_df = collect_cv_results(results_root)
    test_df = build_test_results_table(long_df)
    summary_df = build_summary_table(long_df)
    formatted_df = build_formatted_summary_table(summary_df, precision=args.precision)

    long_path = output_dir / "cv_results_long.csv"
    test_path = output_dir / "test_results_all.csv"
    summary_path = output_dir / "cv_results_summary_mean_std.csv"
    formatted_path = output_dir / "cv_results_summary_formatted.csv"

    long_df.to_csv(long_path, index=False)
    test_df.to_csv(test_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    formatted_df.to_csv(formatted_path, index=False)

    print(f"Wrote fold-level table: {long_path}")
    print(f"Wrote held-out test table: {test_path}")
    print(f"Wrote mean/std table: {summary_path}")
    print(f"Wrote formatted table: {formatted_path}")
    print(
        f"Collected {len(long_df)} rows from "
        f"{summary_df[['experiment', 'category', 'model']].drop_duplicates().shape[0]} "
        "experiment/category/model combinations."
    )


if __name__ == "__main__":
    main()
