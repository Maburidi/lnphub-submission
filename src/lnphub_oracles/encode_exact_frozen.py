from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ENCODER_NAMES = {
    "agile": ("AGILE", "agile"),
    "chemberta_mtr": ("ChemBERTa_MTR", "chemberta_mtr"),
    "molformer": ("MoLFormer", "molformer"),
    "molclr_gin": ("MolCLR_GIN", "molclr_gin"),
    "uni_mol": ("Uni_Mol", "unimol"),
    "lion": ("LiON", "lion"),
}
ID_COLUMNS = [
    "record_id", "study_id", "lnp_id", "il_id", "il_smiles",
    "canonical_il_smiles", "delivery", "delivery_class_3",
    "delivery_class_label", "oracle_split", "cv_fold",
]
LION_FORMULATION_MAP = {
    "IL_molratio": "il_mol_ratio",
    "HL_molratio": "helper_lipid_mol_ratio",
    "CHL_molratio": "cholesterol_mol_ratio",
    "PEG_molratio": "peg_lipid_mol_ratio",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode the exact LNPHub classification cohort with frozen molecular encoders."
    )
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-dir", default=Path("/content/notebooks/encoder_repos"), type=Path)
    parser.add_argument("--methods", nargs="+", required=True, choices=sorted(ENCODER_NAMES))
    parser.add_argument("--lion-mode", choices=["smiles_only", "smiles_formulations"], default="smiles_only")
    return parser.parse_args()


def clone_if_needed(repo_dir: Path, name: str, url: str) -> Path:
    destination = repo_dir / name
    if destination.exists():
        print(f"{name}: repo ready -> {destination}", flush=True)
    else:
        print(f"{name}: cloning {url}", flush=True)
        subprocess.run(["git", "clone", "--depth", "1", url, str(destination)], check=True)
    return destination


def canonicalize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from rdkit import Chem

    if "record_id" not in frame.columns or "il_smiles" not in frame.columns:
        raise ValueError("The exact cohort must contain record_id and il_smiles.")
    if frame["record_id"].duplicated().any():
        duplicate = frame.loc[frame["record_id"].duplicated(), "record_id"].head().tolist()
        raise ValueError(f"record_id must be unique; examples of duplicates: {duplicate}")

    out = frame.copy().reset_index(drop=True)
    canonical = []
    for smiles in out["il_smiles"].astype(str):
        mol = Chem.MolFromSmiles(smiles.strip())
        canonical.append(Chem.MolToSmiles(mol, canonical=True) if mol is not None else np.nan)
    out["canonical_il_smiles"] = canonical
    invalid = out["canonical_il_smiles"].isna()
    if invalid.any():
        bad = out.loc[invalid, ["record_id", "il_smiles"]].head(20)
        raise ValueError(
            f"Exact coverage is required, but {int(invalid.sum())} SMILES are invalid:\n{bad.to_string(index=False)}"
        )
    unique = out[["canonical_il_smiles"]].drop_duplicates().reset_index(drop=True)
    return out, unique


def save_unique_vectors(
    cohort: pd.DataFrame,
    unique_smiles: pd.DataFrame,
    vectors: np.ndarray,
    output_dir: Path,
    method: str,
    manifest_extra: dict | None = None,
) -> Path:
    encoder_name, prefix = ENCODER_NAMES[method]
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(unique_smiles):
        raise ValueError(
            f"{encoder_name}: expected {len(unique_smiles)} unique-SMILES vectors, got {vectors.shape}."
        )
    columns = [f"{prefix}_{i:04d}" for i in range(vectors.shape[1])]
    unique_vectors = pd.concat(
        [unique_smiles.reset_index(drop=True), pd.DataFrame(vectors, columns=columns)], axis=1
    )
    ids = cohort[[column for column in ID_COLUMNS if column in cohort.columns]].copy()
    output = ids.merge(unique_vectors, on="canonical_il_smiles", how="left", validate="many_to_one")
    return save_record_vectors(
        output, columns, output_dir, encoder_name, method, len(unique_smiles), manifest_extra
    )


def save_record_vectors(
    output: pd.DataFrame,
    vector_columns: list[str],
    output_dir: Path,
    encoder_name: str,
    method: str,
    n_unique_smiles: int,
    manifest_extra: dict | None = None,
) -> Path:
    matrix = output[vector_columns].to_numpy(dtype=np.float32)
    if len(output) == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{encoder_name}: empty encoded output.")
    if not np.isfinite(matrix).all():
        bad_rows = int((~np.isfinite(matrix)).any(axis=1).sum())
        raise ValueError(f"{encoder_name}: {bad_rows} records contain non-finite vectors.")
    if not np.any(matrix):
        raise ValueError(f"{encoder_name}: every saved vector value is zero.")
    if output["record_id"].duplicated().any():
        raise ValueError(f"{encoder_name}: duplicate record_id values in encoded output.")

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{encoder_name}_encoder_vectors.parquet"
    output.to_parquet(path, index=False)
    manifest = {
        "encoder": encoder_name,
        "method_key": method,
        "output_file": str(path),
        "n_records": int(len(output)),
        "n_unique_smiles": int(n_unique_smiles),
        "embedding_dim": int(len(vector_columns)),
        "finite_fraction": float(np.isfinite(matrix).mean()),
        "zero_fraction": float((matrix == 0).mean()),
    }
    manifest.update(manifest_extra or {})
    (output_dir / f"{encoder_name}_encoder_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"Saved {encoder_name}: {path}\n"
        f"  records={len(output):,}; unique_smiles={n_unique_smiles:,}; dim={len(vector_columns):,}",
        flush=True,
    )
    return path


def cleanup_accelerator() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def encode_hf(unique_smiles: pd.DataFrame, method: str) -> np.ndarray:
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    jobs = {
        "chemberta_mtr": ("DeepChem/ChemBERTa-77M-MTR", 32, 256, False),
        "molformer": ("ibm-research/MoLFormer-XL-both-10pct", 16, 256, True),
    }
    model_id, batch_size, max_length, trust_remote_code = jobs[method]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== {ENCODER_NAMES[method][0]} on {device} ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    try:
        model = AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    except Exception:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
    model.to(device).eval()
    smiles = unique_smiles["canonical_il_smiles"].astype(str).tolist()
    chunks = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(smiles), batch_size), desc=ENCODER_NAMES[method][0]):
            tokens = tokenizer(
                smiles[start:start + batch_size], padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            result = model(**tokens, output_hidden_states=True, return_dict=True)
            hidden = getattr(result, "last_hidden_state", None)
            if hidden is None:
                hidden = result.hidden_states[-1]
            mask = tokens.get(
                "attention_mask", torch.ones(hidden.shape[:2], device=device)
            ).unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            chunks.append(pooled.cpu().numpy().astype(np.float32))
    del model
    cleanup_accelerator()
    return np.vstack(chunks)


def encode_unimol(unique_smiles: pd.DataFrame) -> np.ndarray:
    from tqdm.auto import tqdm
    from unimol_tools import UniMolRepr

    print("\n=== Uni-Mol ===", flush=True)
    model = UniMolRepr(data_type="molecule", remove_hs=False, batch_size=64)
    smiles = unique_smiles["canonical_il_smiles"].astype(str).tolist()
    chunks = []
    for start in tqdm(range(0, len(smiles), 64), desc="Uni-Mol"):
        representation = model.get_repr(smiles[start:start + 64], return_atomic_reprs=False)
        if isinstance(representation, dict):
            representation = representation.get("cls_repr", representation.get("molecular_repr"))
        elif isinstance(representation, list) and representation and isinstance(representation[0], dict):
            representation = [
                item.get("cls_repr", item.get("molecular_repr")) for item in representation
            ]
        array = np.asarray(representation, dtype=np.float32)
        if array.ndim == 3 and array.shape[1] == 1:
            array = array[:, 0, :]
        if array.ndim == 1:
            array = array.reshape(1, -1)
        chunks.append(array)
    del model
    cleanup_accelerator()
    return np.vstack(chunks)


def load_class(path: Path, class_name: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {class_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def encode_pretrained_gnn(unique_smiles: pd.DataFrame, method: str, repo_dir: Path) -> np.ndarray:
    import torch
    from rdkit import Chem
    from torch_geometric.data import Batch, Data
    from tqdm.auto import tqdm

    if method == "agile":
        repo = clone_if_needed(repo_dir, "AGILE", "https://github.com/bowang-lab/AGILE.git")
        model_class = load_class(repo / "models/agile_pretrain.py", "AGILE", "lnphub_exact_agile")
        checkpoint = repo / "ckpt/pretrained_agile_60k/checkpoints/model.pth"
    else:
        repo = clone_if_needed(repo_dir, "MolCLR", "https://github.com/yuyangw/MolCLR.git")
        model_class = load_class(repo / "models/ginet_molclr.py", "GINet", "lnphub_exact_molclr")
        checkpoint = repo / "ckpt/pretrained_gin/checkpoints/model.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing official checkpoint: {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0, pool="mean").to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {key.replace("module.", ""): value for key, value in state.items()}
    message = model.load_state_dict(state, strict=False)
    print(
        f"\n=== {ENCODER_NAMES[method][0]} on {device} ===\n"
        f"Missing keys={len(message.missing_keys)}; unexpected keys={len(message.unexpected_keys)}",
        flush=True,
    )
    model.eval()

    atom_list = list(range(1, 119))
    chirality_list = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    ]
    bond_list = [
        Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC,
    ]
    bond_direction_list = [
        Chem.rdchem.BondDir.NONE, Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
    ]

    def graph(smiles: str) -> Data:
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        atoms = []
        for atom in mol.GetAtoms():
            atomic_number = atom.GetAtomicNum()
            if atomic_number not in atom_list:
                raise ValueError(f"Unsupported atomic number {atomic_number} in {smiles}")
            chirality = atom.GetChiralTag()
            chirality_index = chirality_list.index(chirality) if chirality in chirality_list else 0
            atoms.append([atom_list.index(atomic_number), chirality_index])
        rows, columns, edges = [], [], []
        for bond in mol.GetBonds():
            bond_type = bond_list.index(bond.GetBondType()) if bond.GetBondType() in bond_list else 0
            direction = (
                bond_direction_list.index(bond.GetBondDir())
                if bond.GetBondDir() in bond_direction_list else 0
            )
            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            rows.extend([begin, end]); columns.extend([end, begin])
            edges.extend([[bond_type, direction], [bond_type, direction]])
        return Data(
            x=torch.tensor(atoms, dtype=torch.long),
            edge_index=torch.tensor([rows, columns], dtype=torch.long).reshape(2, -1),
            edge_attr=torch.tensor(edges, dtype=torch.long).reshape(-1, 2),
            y=torch.zeros((1, 1), dtype=torch.float),
        )

    graphs = [graph(smiles) for smiles in tqdm(
        unique_smiles["canonical_il_smiles"].astype(str), desc=f"Build {ENCODER_NAMES[method][0]} graphs"
    )]
    chunks = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(graphs), 128), desc=ENCODER_NAMES[method][0]):
            batch = Batch.from_data_list(graphs[start:start + 128]).to(device)
            embedding, _ = model(batch)
            chunks.append(embedding.cpu().numpy().astype(np.float32))
    del model, graphs
    cleanup_accelerator()
    return np.vstack(chunks)


def ensure_lion_chemprop(repo_dir: Path) -> tuple[Path, Path, Path]:
    lnpdb = clone_if_needed(repo_dir, "LNPDB", "https://github.com/evancollins1/LNPDB.git")
    lion_dir = lnpdb / "data/LNPDB_for_LiON"
    checkpoint_dir = lion_dir / "single_split/trained_model_checkpoints"
    schema_path = lion_dir / "all_data_extra_x.csv"
    if not checkpoint_dir.exists() or not schema_path.exists():
        raise FileNotFoundError("The official LiON checkpoint or 92-column schema is missing.")

    venv = repo_dir.parent / "lion_py38"
    python = venv / "bin/python"
    executable = venv / "bin/chemprop_fingerprint"
    uv = shutil.which("uv")
    if uv is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
        uv_command = [sys.executable, "-m", "uv"]
    else:
        uv_command = [uv]
    if not python.exists():
        subprocess.run([*uv_command, "python", "install", "3.8"], check=False)
        subprocess.run([*uv_command, "venv", str(venv), "--python", "3.8"], check=True)
    check = subprocess.run([str(python), "-c", "import chemprop, pkg_resources"], capture_output=True)
    if check.returncode != 0 or not executable.exists():
        subprocess.run([
            *uv_command, "pip", "install", "--python", str(python), "setuptools<70",
            "chemprop==1.7.0", "rdkit-pypi", "pandas", "numpy",
        ], check=True)
    return executable, checkpoint_dir, schema_path


def run_live(command: list[str], label: str) -> None:
    print("$", " ".join(command), flush=True)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    if process.wait() != 0:
        raise RuntimeError(f"{label} failed.")


def encode_lion(
    cohort: pd.DataFrame,
    unique_smiles: pd.DataFrame,
    output_dir: Path,
    repo_dir: Path,
    mode: str,
) -> Path:
    executable, checkpoint_dir, schema_path = ensure_lion_chemprop(repo_dir)
    schema = pd.read_csv(schema_path, nrows=0).columns.tolist()
    if len(schema) != 92:
        raise ValueError(f"Expected LiON's 92-column schema, found {len(schema)} columns.")

    if mode == "smiles_only":
        work = unique_smiles.copy()
        smiles = work["canonical_il_smiles"].astype(str)
        features = pd.DataFrame(0.0, index=work.index, columns=schema)
    else:
        work = cohort.reset_index(drop=True).copy()
        smiles = work["canonical_il_smiles"].astype(str)
        features = pd.DataFrame(0.0, index=work.index, columns=schema)
        for lion_column, source_column in LION_FORMULATION_MAP.items():
            if lion_column not in features.columns:
                raise ValueError(f"LiON's official feature schema is missing {lion_column}.")
            if source_column not in work.columns:
                raise ValueError(f"LiON formulation mode requires {source_column}.")
            features[lion_column] = pd.to_numeric(work[source_column], errors="coerce").fillna(0.0)

    temporary = output_dir / f"_lion_{mode}_input"
    temporary.mkdir(parents=True, exist_ok=True)
    data_path = temporary / "cohort.csv"
    feature_path = temporary / "features.csv"
    raw_path = temporary / "fingerprints.csv"
    pd.DataFrame({"IL_SMILES": smiles, "Experiment_value": np.zeros(len(work))}).to_csv(
        data_path, index=False
    )
    features.to_csv(feature_path, index=False)
    run_live([
        str(executable), "--checkpoint_dir", str(checkpoint_dir),
        "--test_path", str(data_path), "--features_path", str(feature_path),
        "--preds_path", str(raw_path),
    ], f"LiON {mode} encoding")

    raw = pd.read_csv(raw_path)
    vector_columns = [
        column for column in raw.columns
        if column not in {"smiles", "target", "IL_SMILES", "Experiment_value", "preds"}
        and pd.to_numeric(raw[column], errors="coerce").notna().any()
    ]
    vectors = raw[vector_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    print(
        f"LiON mode={mode}; records encoded={len(vectors):,}; "
        f"92-vector nonzero entries={int(np.count_nonzero(features.to_numpy())):,}",
        flush=True,
    )
    if mode == "smiles_only":
        return save_unique_vectors(
            cohort,
            unique_smiles,
            vectors,
            output_dir,
            "lion",
            {
                "lion_mode": mode,
                "lion_auxiliary_schema_size": len(schema),
                "lion_populated_auxiliary_columns": [],
            },
        )

    columns = [f"lion_{index:04d}" for index in range(vectors.shape[1])]
    ids = work[[column for column in ID_COLUMNS if column in work.columns]].copy()
    output = pd.concat([ids.reset_index(drop=True), pd.DataFrame(vectors, columns=columns)], axis=1)
    return save_record_vectors(
        output,
        columns,
        output_dir,
        "LiON",
        "lion",
        len(unique_smiles),
        {
            "lion_mode": mode,
            "lion_auxiliary_schema_size": len(schema),
            "lion_populated_auxiliary_columns": list(LION_FORMULATION_MAP),
            "lion_lnp_source_columns": LION_FORMULATION_MAP,
        },
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.repo_dir.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(args.cohort, low_memory=False)
    cohort, unique_smiles = canonicalize(cohort)
    print(
        f"Exact cohort: {len(cohort):,} records; {cohort['record_id'].nunique():,} record IDs; "
        f"{len(unique_smiles):,} unique valid SMILES",
        flush=True,
    )

    status_path = args.output_dir / "encoding_status.csv"
    if status_path.exists():
        status = pd.read_csv(status_path).to_dict("records")
        requested_names = {ENCODER_NAMES[method][0] for method in args.methods}
        status = [row for row in status if row.get("encoder") not in requested_names]
    else:
        status = []
    for method in args.methods:
        try:
            if method in {"chemberta_mtr", "molformer"}:
                vectors = encode_hf(unique_smiles, method)
                path = save_unique_vectors(cohort, unique_smiles, vectors, args.output_dir, method)
            elif method == "uni_mol":
                vectors = encode_unimol(unique_smiles)
                path = save_unique_vectors(cohort, unique_smiles, vectors, args.output_dir, method)
            elif method in {"agile", "molclr_gin"}:
                vectors = encode_pretrained_gnn(unique_smiles, method, args.repo_dir)
                path = save_unique_vectors(cohort, unique_smiles, vectors, args.output_dir, method)
            elif method == "lion":
                path = encode_lion(
                    cohort, unique_smiles, args.output_dir, args.repo_dir, args.lion_mode
                )
            else:
                raise ValueError(f"Unsupported encoder: {method}")
            status.append({"encoder": ENCODER_NAMES[method][0], "status": "encoded", "file": str(path)})
        except Exception as error:
            status.append({"encoder": ENCODER_NAMES[method][0], "status": "failed", "error": repr(error)})
            pd.DataFrame(status).to_csv(status_path, index=False)
            raise
        pd.DataFrame(status).to_csv(status_path, index=False)

    print("\nEncoding status:", flush=True)
    print(pd.DataFrame(status).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
