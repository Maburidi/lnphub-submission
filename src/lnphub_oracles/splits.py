from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, KFold, train_test_split


def make_splits(frame: pd.DataFrame, seed: int = 7, group_col: str = "study_id") -> dict[str, np.ndarray]:
    n = len(frame)
    idx = np.arange(n)
    if group_col in frame.columns and frame[group_col].nunique(dropna=True) >= 3:
        groups = frame[group_col].astype(str).fillna("missing").to_numpy()
        first = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        train_val_idx, test_idx = next(first.split(idx, groups=groups))
        groups_tv = groups[train_val_idx]
        if len(np.unique(groups_tv)) >= 3:
            second = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=seed + 1)
            train_rel, val_rel = next(second.split(train_val_idx, groups=groups_tv))
            train_idx = train_val_idx[train_rel]
            val_idx = train_val_idx[val_rel]
        else:
            train_idx, val_idx = train_test_split(train_val_idx, test_size=0.1765, random_state=seed)
    else:
        train_val_idx, test_idx = train_test_split(idx, test_size=0.15, random_state=seed)
        train_idx, val_idx = train_test_split(train_val_idx, test_size=0.1765, random_state=seed + 1)
    return {"train": np.asarray(train_idx), "val": np.asarray(val_idx), "test": np.asarray(test_idx)}


def assign_untouched_test_and_cv_folds(
    frame: pd.DataFrame,
    *,
    seed: int = 7,
    group_col: str = "study_id",
    test_size: float = 0.15,
    n_folds: int = 5,
) -> pd.DataFrame:
    """Assign one untouched test split and CV folds within the remaining rows."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")

    out = pd.DataFrame({"row_index": np.arange(len(frame), dtype=int)})
    out["split"] = "trainval"
    out["cv_fold"] = -1

    if group_col in frame.columns and frame[group_col].nunique(dropna=True) >= n_folds + 1:
        groups = frame[group_col].astype(str).fillna("missing").to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        trainval_idx, test_idx = next(splitter.split(np.arange(len(frame)), groups=groups))
        out.loc[test_idx, "split"] = "untouched_test"

        trainval_groups = pd.Series(groups[trainval_idx], index=trainval_idx)
        group_sizes = trainval_groups.value_counts().sample(frac=1, random_state=seed).sort_values(ascending=False)
        fold_load = {i: 0 for i in range(n_folds)}
        group_to_fold = {}
        for group, size in group_sizes.items():
            fold = min(fold_load, key=fold_load.get)
            group_to_fold[group] = fold
            fold_load[fold] += int(size)
        out.loc[trainval_idx, "cv_fold"] = trainval_groups.map(group_to_fold).astype(int).to_numpy()
    else:
        idx = np.arange(len(frame))
        trainval_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed, shuffle=True)
        out.loc[test_idx, "split"] = "untouched_test"
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (_, val_rel) in enumerate(kf.split(trainval_idx)):
            out.loc[trainval_idx[val_rel], "cv_fold"] = fold

    return out


def cv_fold_indices(frame: pd.DataFrame, fold_col: str = "cv_fold") -> list[tuple[int, np.ndarray, np.ndarray]]:
    if fold_col not in frame.columns:
        raise ValueError(f"Missing CV fold column: {fold_col}")
    folds = sorted(int(f) for f in pd.Series(frame[fold_col]).dropna().unique() if int(f) >= 0)
    out = []
    all_idx = np.arange(len(frame))
    for fold in folds:
        val_idx = np.where(frame[fold_col].astype(int).to_numpy() == fold)[0]
        train_idx = np.setdiff1d(all_idx, val_idx)
        if len(train_idx) and len(val_idx):
            out.append((fold, train_idx, val_idx))
    if not out:
        raise ValueError("No usable CV folds found.")
    return out
