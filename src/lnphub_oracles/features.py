from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors


def smiles_series(frame: pd.DataFrame) -> pd.Series:
    for col in ["canonical_il_smiles", "il_smiles", "SMILES", "smiles"]:
        if col in frame.columns:
            return frame[col].astype(str)
    raise ValueError("No SMILES column found.")


def valid_smiles_mask(smiles: pd.Series) -> np.ndarray:
    return smiles.map(lambda x: Chem.MolFromSmiles(str(x)) is not None).to_numpy(dtype=bool)


def make_ecfp4(smiles: pd.Series, n_bits: int = 2048) -> tuple[np.ndarray, list[str]]:
    vectors = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles.astype(str)):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            vectors[i, :] = np.nan
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        vectors[i, :] = arr.astype(np.float32)
    return vectors, [f"ecfp4_{i:04d}" for i in range(n_bits)]


def make_rdkit_descriptors(smiles: pd.Series) -> tuple[np.ndarray, list[str]]:
    names = [name for name, _ in Descriptors.descList]
    funcs = [fn for _, fn in Descriptors.descList]
    rows = []
    for smi in smiles.astype(str):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append([np.nan] * len(funcs))
            continue
        vals = []
        for fn in funcs:
            try:
                vals.append(float(fn(mol)))
            except Exception:
                vals.append(np.nan)
        rows.append(vals)
    X = np.asarray(rows, dtype=np.float32)
    med = np.nanmedian(X, axis=0)
    med[~np.isfinite(med)] = 0
    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(med, inds[1])
    return X, [f"rdkit_{name}" for name in names]
