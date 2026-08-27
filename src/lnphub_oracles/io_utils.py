from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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
}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if not isinstance(value, (list, dict, tuple, np.ndarray)):
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False))


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported table format: {path}")


def numeric_target(frame: pd.DataFrame, target_col: str) -> pd.Series:
    if target_col not in frame.columns:
        raise ValueError(f"Missing target column: {target_col}")
    y = pd.to_numeric(frame[target_col], errors="coerce")
    if y.notna().sum() == 0:
        raise ValueError(f"Target column has no numeric values: {target_col}")
    return y


def vector_columns(frame: pd.DataFrame) -> list[str]:
    cols = [c for c in frame.columns if c not in ID_COLS]
    if not cols:
        raise ValueError("No vector columns found.")
    return cols


def method_dir(output_dir: str | Path, method_name: str) -> Path:
    safe = (
        method_name.strip()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("/", "_")
        .replace(":", "_")
        .replace("-", "_")
    )
    return ensure_dir(Path(output_dir) / safe)
