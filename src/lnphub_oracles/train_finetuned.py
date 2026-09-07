from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from .features import smiles_series, valid_smiles_mask
from .io_utils import ensure_dir, method_dir, now_utc, read_table, write_json
from .metrics import format_regression_metrics, regression_metrics
from .splits import cv_fold_indices


HF_METHODS = {
    "chemberta_mtr": ("ChemBERTa-MTR-Finetuned", "DeepChem/ChemBERTa-77M-MTR", False),
    "molformer": ("MoLFormer-Finetuned", "ibm-research/MoLFormer-XL-both-10pct", True),
}
GNN_METHODS = {
    "agile": ("AGILE-Finetuned", "AGILE", "models/agile_pretrain.py", "AGILE", "ckpt/pretrained_agile_60k/checkpoints/model.pth"),
    "molclr_gin": ("MolCLR-GIN-Finetuned", "MolCLR", "models/ginet_molclr.py", "GINet", "ckpt/pretrained_gin/checkpoints/model.pth"),
}
UNSUPPORTED = {
    "kpgt": ("KPGT-Finetuned", "KPGT fine-tuning needs the official base.pth checkpoint and KPGT fine-tuning workflow."),
    "lion": ("LiON-Finetuned", "LiON fine-tuning needs the LNPDB/Chemprop training workflow with exact LiON formulation features."),
    "unimol": ("Uni-Mol-Finetuned", "Uni-Mol fine-tuning needs the official Uni-Mol training interface; this workflow trains Uni-Mol as a frozen encoder."),
}
DEFAULT_METHODS = list(GNN_METHODS) + list(HF_METHODS) + list(UNSUPPORTED)


class SmilesDataset(Dataset):
    def __init__(self, smiles, y, tokenizer, max_length=256):
        self.smiles = list(smiles)
        self.y = np.asarray(y, dtype=np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        tok = self.tokenizer(
            self.smiles[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in tok.items()}
        item["labels"] = torch.tensor(self.y[idx], dtype=torch.float32)
        return item


class HFRegressor(nn.Module):
    def __init__(self, model_id: str, trust_remote_code: bool):
        super().__init__()
        from transformers import AutoModel, AutoModelForSequenceClassification

        try:
            self.encoder = AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        except Exception:
            self.encoder = AutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        hidden = getattr(self.encoder.config, "hidden_size", None) or getattr(self.encoder.config, "d_model", None)
        if hidden is None:
            raise ValueError(f"Cannot infer hidden size for {model_id}")
        self.head = nn.Sequential(nn.Dropout(0.15), nn.Linear(hidden, 1))

    def forward(self, **batch):
        labels = batch.pop("labels", None)
        out = self.encoder(**batch, output_hidden_states=True, return_dict=True)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            hidden = out.hidden_states[-1]
        mask = batch.get("attention_mask", torch.ones(hidden.shape[:2], device=hidden.device)).unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pred = self.head(pooled).view(-1)
        if labels is None:
            return pred
        return pred, nn.functional.mse_loss(pred, labels)


class PretrainedGNNRegressor(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(0.15), nn.Linear(512, 1))

    def forward(self, batch):
        h, _ = self.encoder(batch)
        return self.head(h).view(-1)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune supported pretrained molecular encoders.")
    parser.add_argument("--subset", required=True, type=Path)
    parser.add_argument("--test-subset", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-dir", default=Path("/content/notebooks/encoder_repos"), type=Path)
    parser.add_argument("--target-col", default="delivery")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def load_class(path: Path, class_name: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)


def load_state_dict(path: Path, device: str):
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if isinstance(obj, dict) and "model_state_dict" in obj:
        obj = obj["model_state_dict"]
    return {k.replace("module.", ""): v for k, v in obj.items()}


def pretrained_graph(smiles: str, y: float):
    from rdkit import Chem
    from torch_geometric.data import Data

    atom_list = list(range(1, 119))
    chirality_list = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    ]
    bond_list = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    bonddir_list = [
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
    ]
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    atoms = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num not in atom_list:
            return None
        chiral = atom.GetChiralTag()
        atoms.append([atom_list.index(atomic_num), chirality_list.index(chiral) if chiral in chirality_list else 0])
    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        bt, bd = bond.GetBondType(), bond.GetBondDir()
        btype = bond_list.index(bt) if bt in bond_list else 0
        bdir = bonddir_list.index(bd) if bd in bonddir_list else 0
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [a, b]
        col += [b, a]
        edge_feat += [[btype, bdir], [btype, bdir]]
    return Data(
        x=torch.tensor(atoms, dtype=torch.long),
        edge_index=torch.tensor([row, col], dtype=torch.long),
        edge_attr=torch.tensor(edge_feat, dtype=torch.long),
        y=torch.tensor([float(y)], dtype=torch.float32),
    )


def build_pretrained_graphs(frame, y_z):
    graphs, keep = [], []
    for i, (smi, yi) in enumerate(zip(smiles_series(frame).tolist(), y_z)):
        graph = pretrained_graph(smi, yi)
        if graph is not None:
            graphs.append(graph)
            keep.append(i)
    return graphs, np.asarray(keep, dtype=int)


def train_loop(model, loaders, args, device):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    best_state, best_val = None, float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in loaders["train"]:
            if isinstance(batch, dict):
                batch = {k: v.to(device) for k, v in batch.items()}
                _, loss = model(**batch)
            else:
                batch = batch.to(device)
                loss = nn.functional.mse_loss(model(batch), batch.y.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        val_loss = evaluate_loss(model, loaders["val"], device)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:03d} val_mse={val_loss:.5g}")
    if best_state:
        with torch.inference_mode():
            model.load_state_dict(best_state)


def evaluate_loss(model, loader, device):
    losses = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            if isinstance(batch, dict):
                batch = {k: v.to(device) for k, v in batch.items()}
                _, loss = model(**batch)
            else:
                batch = batch.to(device)
                loss = nn.functional.mse_loss(model(batch), batch.y.view(-1))
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def predict(model, loader, device):
    yt, yp = [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            if isinstance(batch, dict):
                labels = batch["labels"].detach().cpu().numpy()
                pred = model(**{k: v.to(device) for k, v in batch.items() if k != "labels"}).detach().cpu().numpy()
            else:
                batch = batch.to(device)
                labels = batch.y.view(-1).detach().cpu().numpy()
                pred = model(batch).detach().cpu().numpy()
            yt.append(labels)
            yp.append(pred)
    return np.concatenate(yt), np.concatenate(yp)


def write_unavailable(output_dir: Path, method_name: str, reason: str):
    out = method_dir(output_dir, method_name)
    write_json(out / "metadata.json", {
        "method_name": method_name,
        "status": "not_trained",
        "reason": reason,
        "trained_at_utc": now_utc(),
    })
    print(f"Skipped {method_name}: {reason}")
    return {"method": method_name, "output_dir": str(out), "status": "not_trained", "reason": reason}


def save_outputs(out_dir, method_name, model, frame, split_metrics, cv_pred_rows, test_pred, args, y_mean, y_std, extra_meta):
    torch.save({"state_dict": model.state_dict(), **extra_meta}, out_dir / "model.pt")
    pd.concat(cv_pred_rows, ignore_index=True).to_csv(out_dir / "cv_predictions.csv", index=False)
    if test_pred is not None:
        test_pred.to_csv(out_dir / "untouched_test_predictions.csv", index=False)
    fold_df = pd.DataFrame([v for k, v in split_metrics.items() if k.startswith("fold_")])
    metric_rows = [{"split": k, **v} for k, v in split_metrics.items()]
    metric_rows.append({"split": "cv_mean", **fold_df.mean(numeric_only=True).to_dict()})
    metric_rows.append({"split": "cv_std", **fold_df.std(numeric_only=True).to_dict()})
    pd.DataFrame(metric_rows).to_csv(out_dir / "metrics.csv", index=False)
    write_json(out_dir / "metrics.json", split_metrics)
    write_json(out_dir / "metadata.json", {
        "method_name": method_name,
        "status": "trained",
        "target_col": args.target_col,
        "target_mean": y_mean,
        "target_std": y_std,
        "n_trainval_rows": len(frame),
        "cv": "5-fold cross-validation on trainval split",
        "untouched_test_used_for_training": False,
        "trained_at_utc": now_utc(),
        "model_file": str(out_dir / "model.pt"),
        "metadata": extra_meta,
    })
    cv_mean = next(row for row in metric_rows if row["split"] == "cv_mean")
    out = {"method": method_name, "output_dir": str(out_dir), "status": "trained"}
    out.update({f"cv_mean_{k}": v for k, v in cv_mean.items() if k != "split"})
    if "untouched_test" in split_metrics:
        out.update({f"untouched_test_{k}": v for k, v in split_metrics["untouched_test"].items()})
    print(f"{method_name} | cv_mean: {format_regression_metrics(cv_mean)}", flush=True)
    if "untouched_test" in split_metrics:
        print(f"{method_name} | untouched_test: {format_regression_metrics(split_metrics['untouched_test'])}", flush=True)
    print(f"{method_name}: saved outputs -> {out_dir}", flush=True)
    return out


def train_hf(method_key, frame, test_frame, y_z, test_y_z, args, device, y_mean, y_std):
    from transformers import AutoTokenizer

    method_name, model_id, trust_remote_code = HF_METHODS[method_key]
    out_dir = method_dir(args.output_dir, method_name)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    ds = SmilesDataset(smiles_series(frame).tolist(), y_z, tokenizer)
    test_ds = SmilesDataset(smiles_series(test_frame).tolist(), test_y_z, tokenizer) if test_frame is not None else None
    split_metrics, cv_pred_rows = {}, []
    for fold, train_idx, val_idx in cv_fold_indices(frame):
        print(f"\n=== {method_name} | fold {fold} ===")
        model = HFRegressor(model_id, trust_remote_code).to(device)
        loaders = {
            "train": DataLoader(Subset(ds, train_idx.tolist()), batch_size=args.batch_size, shuffle=True),
            "val": DataLoader(Subset(ds, val_idx.tolist()), batch_size=args.batch_size, shuffle=False),
        }
        train_loop(model, loaders, args, device)
        yt_z, yp_z = predict(model, loaders["val"], device)
        yt, yp = yt_z * y_std + y_mean, yp_z * y_std + y_mean
        split = f"fold_{fold}"
        split_metrics[split] = regression_metrics(yt, yp)
        print(f"{method_name} | {split}: {format_regression_metrics(split_metrics[split])}", flush=True)
        part = frame.iloc[val_idx][[c for c in ["record_id", "study_id", "lnp_id", "il_id", "il_smiles"] if c in frame.columns]].copy()
        part["split"] = split
        part["cv_fold"] = fold
        part["y_true"] = yt
        part["y_pred"] = yp
        cv_pred_rows.append(part)

    print(f"\n=== {method_name} | final trainval model ===")
    final_model = HFRegressor(model_id, trust_remote_code).to(device)
    all_idx = np.arange(len(ds))
    train_loop(final_model, {
        "train": DataLoader(Subset(ds, all_idx.tolist()), batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(Subset(ds, all_idx.tolist()), batch_size=args.batch_size, shuffle=False),
    }, args, device)
    tokenizer.save_pretrained(out_dir / "tokenizer")
    final_model.encoder.save_pretrained(out_dir / "encoder")
    test_pred = None
    if test_ds is not None:
        yt_z, yp_z = predict(final_model, DataLoader(test_ds, batch_size=args.batch_size, shuffle=False), device)
        yt, yp = yt_z * y_std + y_mean, yp_z * y_std + y_mean
        split_metrics["untouched_test"] = regression_metrics(yt, yp)
        test_pred = test_frame[[c for c in ["record_id", "study_id", "lnp_id", "il_id", "il_smiles"] if c in test_frame.columns]].copy()
        test_pred["split"] = "untouched_test"
        test_pred["y_true"] = yt
        test_pred["y_pred"] = yp
    return save_outputs(out_dir, method_name, final_model, frame, split_metrics, cv_pred_rows, test_pred, args, y_mean, y_std, {"model_id": model_id, "encoder_trainable": True})


def train_pretrained_gnn(method_key, frame, test_frame, y_z, test_y_z, args, device, y_mean, y_std):
    from torch_geometric.loader import DataLoader as GraphLoader

    method_name, repo_name, model_rel, cls_name, ckpt_rel = GNN_METHODS[method_key]
    repo = args.repo_dir / repo_name
    model_path = repo / model_rel
    ckpt_path = repo / ckpt_rel
    if not model_path.exists() or not ckpt_path.exists():
        return write_unavailable(args.output_dir, method_name, f"Missing model file or checkpoint under {repo}")
    cls = load_class(model_path, cls_name, f"{method_key}_finetune")
    graphs, keep = build_pretrained_graphs(frame, y_z)
    frame = frame.iloc[keep].reset_index(drop=True)
    y_z = y_z[keep]
    test_graphs, test_keep = ([], np.asarray([], dtype=int))
    if test_frame is not None:
        test_graphs, test_keep = build_pretrained_graphs(test_frame, test_y_z)
        test_frame = test_frame.iloc[test_keep].reset_index(drop=True)

    split_metrics, cv_pred_rows = {}, []
    out_dir = method_dir(args.output_dir, method_name)
    for fold, train_idx, val_idx in cv_fold_indices(frame):
        print(f"\n=== {method_name} | fold {fold} ===")
        encoder = cls(num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0, pool="mean").to(device)
        msg = encoder.load_state_dict(load_state_dict(ckpt_path, device), strict=False)
        print("Missing keys:", len(msg.missing_keys), "Unexpected keys:", len(msg.unexpected_keys))
        model = PretrainedGNNRegressor(encoder).to(device)
        loaders = {
            "train": GraphLoader(Subset(graphs, train_idx.tolist()), batch_size=args.batch_size, shuffle=True),
            "val": GraphLoader(Subset(graphs, val_idx.tolist()), batch_size=args.batch_size, shuffle=False),
        }
        train_loop(model, loaders, args, device)
        yt_z, yp_z = predict(model, loaders["val"], device)
        yt, yp = yt_z * y_std + y_mean, yp_z * y_std + y_mean
        split = f"fold_{fold}"
        split_metrics[split] = regression_metrics(yt, yp)
        print(f"{method_name} | {split}: {format_regression_metrics(split_metrics[split])}", flush=True)
        part = frame.iloc[val_idx][[c for c in ["record_id", "study_id", "lnp_id", "il_id", "il_smiles"] if c in frame.columns]].copy()
        part["split"] = split
        part["cv_fold"] = fold
        part["y_true"] = yt
        part["y_pred"] = yp
        cv_pred_rows.append(part)

    print(f"\n=== {method_name} | final trainval model ===")
    encoder = cls(num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0, pool="mean").to(device)
    encoder.load_state_dict(load_state_dict(ckpt_path, device), strict=False)
    final_model = PretrainedGNNRegressor(encoder).to(device)
    all_idx = np.arange(len(graphs))
    train_loop(final_model, {
        "train": GraphLoader(Subset(graphs, all_idx.tolist()), batch_size=args.batch_size, shuffle=True),
        "val": GraphLoader(Subset(graphs, all_idx.tolist()), batch_size=args.batch_size, shuffle=False),
    }, args, device)
    test_pred = None
    if test_graphs:
        yt_z, yp_z = predict(final_model, GraphLoader(test_graphs, batch_size=args.batch_size, shuffle=False), device)
        yt, yp = yt_z * y_std + y_mean, yp_z * y_std + y_mean
        split_metrics["untouched_test"] = regression_metrics(yt, yp)
        test_pred = test_frame[[c for c in ["record_id", "study_id", "lnp_id", "il_id", "il_smiles"] if c in test_frame.columns]].copy()
        test_pred["split"] = "untouched_test"
        test_pred["y_true"] = yt
        test_pred["y_pred"] = yp
    return save_outputs(out_dir, method_name, final_model, frame, split_metrics, cv_pred_rows, test_pred, args, y_mean, y_std, {"repo": str(repo), "checkpoint": str(ckpt_path), "encoder_trainable": True})


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    frame = read_table(args.subset)
    y = pd.to_numeric(frame[args.target_col], errors="coerce").to_numpy(dtype=np.float32)
    ok = np.isfinite(y) & valid_smiles_mask(smiles_series(frame))
    frame = frame.loc[ok].reset_index(drop=True)
    y = y[ok]
    y_mean = float(np.mean(y))
    y_std = float(np.std(y) if np.std(y) > 0 else 1.0)
    y_z = ((y - y_mean) / y_std).astype(np.float32)

    test_frame = None
    test_y_z = None
    if args.test_subset:
        raw_test = read_table(args.test_subset)
        test_y = pd.to_numeric(raw_test[args.target_col], errors="coerce").to_numpy(dtype=np.float32)
        test_ok = np.isfinite(test_y) & valid_smiles_mask(smiles_series(raw_test))
        test_frame = raw_test.loc[test_ok].reset_index(drop=True)
        test_y_z = ((test_y[test_ok] - y_mean) / y_std).astype(np.float32)

    rows = []
    for method in args.methods:
        key = method.lower().replace("-", "_")
        if key in UNSUPPORTED:
            name, reason = UNSUPPORTED[key]
            rows.append(write_unavailable(args.output_dir, name, reason))
            continue
        try:
            if key in HF_METHODS:
                rows.append(train_hf(key, frame, test_frame, y_z, test_y_z, args, device, y_mean, y_std))
            elif key in GNN_METHODS:
                rows.append(train_pretrained_gnn(key, frame, test_frame, y_z, test_y_z, args, device, y_mean, y_std))
            else:
                rows.append(write_unavailable(args.output_dir, method, "Unknown method key."))
        except Exception as exc:
            name = HF_METHODS.get(key, GNN_METHODS.get(key, (method,)))[0]
            rows.append(write_unavailable(args.output_dir, name, repr(exc)))

    manifest = pd.DataFrame(rows)
    manifest.to_csv(Path(args.output_dir) / "finetuned_training_manifest.csv", index=False)
    print("\nSaved fine-tuned manifest:")
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
