from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from .features import smiles_series, valid_smiles_mask
from .io_utils import ensure_dir, method_dir, now_utc, read_table, write_json
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
    "lion": ("LiON-Finetuned", "LiON fine-tuning needs the original LNPDB/Chemprop formulation-training workflow, not only ionizable-lipid SMILES."),
    "uni_mol": ("Uni-Mol-Finetuned", "Uni-Mol fine-tuning needs the official Uni-Mol training interface/checkpoint workflow; this pipeline supports Uni-Mol as a frozen encoder."),
    "unimol": ("Uni-Mol-Finetuned", "Uni-Mol fine-tuning needs the official Uni-Mol training interface/checkpoint workflow; this pipeline supports Uni-Mol as a frozen encoder."),
}
DEFAULT_METHODS = ["agile", "lion", "chemberta_mtr", "molformer", "molclr_gin", "uni_mol"]
CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = {0: "low", 1: "medium", 2: "high"}


class SmilesClassificationDataset(Dataset):
    def __init__(self, smiles, y, tokenizer, max_length: int = 256):
        self.smiles = list(smiles)
        self.y = np.asarray(y, dtype=np.int64)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokenized = self.tokenizer(
            self.smiles[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in tokenized.items()}
        item["labels"] = torch.tensor(self.y[idx], dtype=torch.long)
        return item


class HFClassifier(nn.Module):
    def __init__(self, model_id: str, trust_remote_code: bool, dropout: float = 0.15):
        super().__init__()
        from transformers import AutoModel, AutoModelForSequenceClassification

        try:
            self.encoder = AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        except Exception:
            self.encoder = AutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        hidden = getattr(self.encoder.config, "hidden_size", None) or getattr(self.encoder.config, "d_model", None)
        if hidden is None:
            raise ValueError(f"Cannot infer hidden size for {model_id}")
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, len(CLASS_LABELS)))

    def forward(self, **batch):
        labels = batch.pop("labels", None)
        out = self.encoder(**batch, output_hidden_states=True, return_dict=True)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            hidden = out.hidden_states[-1]
        mask = batch.get("attention_mask", torch.ones(hidden.shape[:2], device=hidden.device)).unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        logits = self.head(pooled)
        if labels is None:
            return logits
        return logits, F.cross_entropy(logits, labels)


class PretrainedGNNClassifier(nn.Module):
    def __init__(self, encoder, dropout: float = 0.15):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, len(CLASS_LABELS)))

    def forward(self, batch):
        h, _ = self.encoder(batch)
        return self.head(h)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune pretrained molecular encoders for LNPHub 3-class classification.")
    parser.add_argument("--trainval-subset", required=True, type=Path)
    parser.add_argument("--test-subset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-dir", default=Path("/content/notebooks/encoder_repos"), type=Path)
    parser.add_argument("--target-col", default="delivery_class_3")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def load_class(path: Path, class_name: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {class_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def load_state_dict(path: Path, device: torch.device):
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if isinstance(obj, dict) and "model_state_dict" in obj:
        obj = obj["model_state_dict"]
    return {str(k).replace("module.", ""): v for k, v in obj.items()}


def pretrained_graph(smiles: str, y: int):
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
    edge_index = torch.tensor([row, col], dtype=torch.long) if row else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_feat, dtype=torch.long) if edge_feat else torch.empty((0, 2), dtype=torch.long)
    return Data(
        x=torch.tensor(atoms, dtype=torch.long),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([int(y)], dtype=torch.long),
    )


def build_pretrained_graphs(frame: pd.DataFrame, target_col: str):
    graphs, keep = [], []
    smiles = smiles_series(frame).tolist()
    y = pd.to_numeric(frame[target_col], errors="coerce").to_numpy()
    for idx, (smi, yi) in enumerate(zip(smiles, y)):
        if not np.isfinite(yi):
            continue
        graph = pretrained_graph(smi, int(yi))
        if graph is not None:
            graph.row_index = int(idx)
            graphs.append(graph)
            keep.append(idx)
    return graphs, np.asarray(keep, dtype=int)


def predict_proba(model, loader, device: torch.device):
    y_true, probs = [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            if isinstance(batch, dict):
                labels = batch["labels"].detach().cpu().numpy()
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                logits = model(**inputs)
            else:
                batch = batch.to(device)
                labels = batch.y.view(-1).detach().cpu().numpy()
                logits = model(batch)
            y_true.append(labels)
            probs.append(F.softmax(logits, dim=1).detach().cpu().numpy())
    y_true = np.concatenate(y_true).astype(int)
    proba = np.vstack(probs)
    pred = np.argmax(proba, axis=1)
    return y_true, pred, proba


def evaluate_loss(model, loader, device: torch.device):
    losses = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            if isinstance(batch, dict):
                batch = {k: v.to(device) for k, v in batch.items()}
                _, loss = model(**batch)
            else:
                batch = batch.to(device)
                loss = F.cross_entropy(model(batch), batch.y.view(-1))
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def train_loop(model, train_loader, val_loader, args, device: torch.device, method_name: str, fold_label: str):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            if isinstance(batch, dict):
                batch = {k: v.to(device) for k, v in batch.items()}
                _, loss = model(**batch)
            else:
                batch = batch.to(device)
                loss = F.cross_entropy(model(batch), batch.y.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(float(loss.detach().cpu()))

        val_loss = evaluate_loss(model, val_loader, device)
        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 2 == 0 or bad_epochs >= args.patience:
            print(f"{method_name} | {fold_label} | epoch {epoch:03d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, best_val={best_val:.4f}", flush=True)
        if bad_epochs >= args.patience:
            print(f"{method_name} | {fold_label}: early stopping at epoch {epoch}", flush=True)
            break
    if best_state is not None:
        with torch.inference_mode():
            model.load_state_dict(best_state)
    return best_val


def classification_metrics(y_true, y_pred, proba) -> dict[str, float]:
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    try:
        out["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, proba, labels=CLASS_LABELS, multi_class="ovr", average="macro"))
    except Exception:
        out["roc_auc_ovr_macro"] = np.nan
    try:
        y_bin = label_binarize(y_true, classes=CLASS_LABELS)
        out["average_precision_macro"] = float(average_precision_score(y_bin, proba, average="macro"))
    except Exception:
        out["average_precision_macro"] = np.nan
    try:
        out["log_loss"] = float(log_loss(y_true, proba, labels=CLASS_LABELS))
    except Exception:
        out["log_loss"] = np.nan
    return out


def format_metrics(metrics: dict[str, float]) -> str:
    return (
        f"n={metrics.get('n', 0):,.0f} | "
        f"acc={metrics.get('accuracy', np.nan):.3f} | "
        f"bal_acc={metrics.get('balanced_accuracy', np.nan):.3f} | "
        f"macroF1={metrics.get('macro_f1', np.nan):.3f} | "
        f"AUROC={metrics.get('roc_auc_ovr_macro', np.nan):.3f} | "
        f"AUPRC={metrics.get('average_precision_macro', np.nan):.3f} | "
        f"MCC={metrics.get('mcc', np.nan):.3f}"
    )


def id_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in ["record_id", "study_id", "lnp_id", "il_id", "il_smiles", "canonical_il_smiles", "delivery", "delivery_class_label"] if c in frame.columns]


def save_prediction_frame(frame_part, split, y_true, y_pred, proba, cv_fold=None) -> pd.DataFrame:
    out = frame_part[id_columns(frame_part)].copy()
    out["split"] = split
    if cv_fold is not None:
        out["cv_fold"] = cv_fold
    out["y_true"] = y_true
    out["y_pred"] = y_pred
    for cls in CLASS_LABELS:
        out[f"prob_{CLASS_NAMES[cls]}"] = proba[:, cls]
    return out


def write_unavailable(output_dir: Path, method_name: str, reason: str) -> dict:
    out_dir = method_dir(output_dir, method_name)
    write_json(out_dir / "metadata.json", {
        "method": method_name,
        "status": "not_trained",
        "reason": reason,
        "trained_at_utc": now_utc(),
    })
    print(f"Skipped {method_name}: {reason}", flush=True)
    return {"method": method_name, "status": "not_trained", "output_dir": str(out_dir), "reason": reason}


def save_common_outputs(
    *,
    output_dir: Path,
    method_name: str,
    model,
    model_payload: dict,
    metric_rows: list[dict],
    cv_predictions: list[pd.DataFrame],
    test_predictions: pd.DataFrame,
    y_test,
    test_pred,
    args,
    extra_meta: dict,
) -> dict:
    out_dir = method_dir(output_dir, method_name)
    torch.save({"state_dict": model.state_dict(), **model_payload}, out_dir / "model.pt")
    pd.concat(cv_predictions, ignore_index=True).to_csv(out_dir / "cv_predictions.csv", index=False)
    test_predictions.to_csv(out_dir / "untouched_test_predictions.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(out_dir / "metrics.csv", index=False)

    cm = confusion_matrix(y_test, test_pred, labels=CLASS_LABELS)
    cm_df = pd.DataFrame(cm, index=[f"true_{CLASS_NAMES[c]}" for c in CLASS_LABELS], columns=[f"pred_{CLASS_NAMES[c]}" for c in CLASS_LABELS])
    cm_df.to_csv(out_dir / "untouched_test_confusion_matrix.csv")
    per_class = precision_recall_fscore_support(y_test, test_pred, labels=CLASS_LABELS, zero_division=0)
    pd.DataFrame({
        "class": [CLASS_NAMES[c] for c in CLASS_LABELS],
        "precision": per_class[0],
        "recall": per_class[1],
        "f1": per_class[2],
        "support": per_class[3],
    }).to_csv(out_dir / "untouched_test_per_class_metrics.csv", index=False)

    write_json(out_dir / "metadata.json", {
        "method": method_name,
        "status": "trained",
        "target": args.target_col,
        "class_names": CLASS_NAMES,
        "trainval_path": str(args.trainval_subset),
        "untouched_test_path": str(args.test_subset),
        "cv": "leave-one-study-out CV on balanced trainval data",
        "untouched_test_used_for_training": False,
        "epochs_max": args.epochs,
        "early_stopping_patience": args.patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "trained_at_utc": now_utc(),
        "output_dir": str(out_dir),
        "metadata": extra_meta,
    })

    cv_mean = next(row for row in metric_rows if row["split"] == "cv_mean")
    test_metrics = next(row for row in metric_rows if row["split"] == "untouched_test")
    out = {
        "method": method_name,
        "status": "trained",
        "output_dir": str(out_dir),
        "cv_accuracy": cv_mean.get("accuracy"),
        "cv_balanced_accuracy": cv_mean.get("balanced_accuracy"),
        "cv_macro_f1": cv_mean.get("macro_f1"),
        "cv_roc_auc_ovr_macro": cv_mean.get("roc_auc_ovr_macro"),
        "cv_average_precision_macro": cv_mean.get("average_precision_macro"),
        "test_accuracy": test_metrics.get("accuracy"),
        "test_balanced_accuracy": test_metrics.get("balanced_accuracy"),
        "test_macro_f1": test_metrics.get("macro_f1"),
        "test_roc_auc_ovr_macro": test_metrics.get("roc_auc_ovr_macro"),
        "test_average_precision_macro": test_metrics.get("average_precision_macro"),
    }
    print(f"{method_name}: saved outputs -> {out_dir}", flush=True)
    return out


def train_hf(method_key, frame, test_frame, args, device: torch.device) -> dict:
    from transformers import AutoTokenizer

    method_name, model_id, trust_remote_code = HF_METHODS[method_key]
    out_dir = method_dir(args.output_dir, method_name)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    dataset = SmilesClassificationDataset(smiles_series(frame).tolist(), frame[args.target_col].to_numpy(dtype=int), tokenizer)
    test_dataset = SmilesClassificationDataset(smiles_series(test_frame).tolist(), test_frame[args.target_col].to_numpy(dtype=int), tokenizer)
    cv_predictions, metric_rows = [], []

    for fold, train_idx, val_idx in cv_fold_indices(frame):
        validation_study = ", ".join(sorted(frame.iloc[val_idx]["study_id"].astype(str).unique()))
        print("", flush=True)
        print(f"{method_name} | fold {fold}: validation study = {validation_study}", flush=True)
        print(f"{method_name} | fold {fold}: train={len(train_idx):,}, val={len(val_idx):,}", flush=True)
        model = HFClassifier(model_id, trust_remote_code).to(device)
        train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx.tolist()), batch_size=args.batch_size, shuffle=False)
        best_loss = train_loop(model, train_loader, val_loader, args, device, method_name, f"fold_{fold}")
        y_true, y_pred, proba = predict_proba(model, val_loader, device)
        metrics = classification_metrics(y_true, y_pred, proba)
        metrics.update({"method": method_name, "split": f"fold_{fold}", "fold": fold, "validation_study": validation_study, "best_val_loss": best_loss})
        metric_rows.append(metrics)
        print(f"{method_name} | fold_{fold}: {format_metrics(metrics)}", flush=True)
        cv_predictions.append(save_prediction_frame(frame.iloc[val_idx], f"fold_{fold}", y_true, y_pred, proba, cv_fold=fold))

    fold_metrics = pd.DataFrame(metric_rows)
    cv_mean = fold_metrics.drop(columns=["method", "split", "fold", "validation_study"], errors="ignore").mean(numeric_only=True).to_dict()
    cv_mean.update({"method": method_name, "split": "cv_mean", "fold": -1, "validation_study": "leave_one_study_out"})
    cv_std = fold_metrics.drop(columns=["method", "split", "fold", "validation_study"], errors="ignore").std(numeric_only=True).to_dict()
    cv_std.update({"method": method_name, "split": "cv_std", "fold": -1, "validation_study": "leave_one_study_out"})
    metric_rows.extend([cv_mean, cv_std])
    print("", flush=True)
    print(f"{method_name} | cv_mean: {format_metrics(cv_mean)}", flush=True)

    print(f"{method_name}: fitting final model on all trainval rows...", flush=True)
    final_model = HFClassifier(model_id, trust_remote_code).to(device)
    all_idx = np.arange(len(dataset))
    train_loop(
        final_model,
        DataLoader(Subset(dataset, all_idx.tolist()), batch_size=args.batch_size, shuffle=True),
        DataLoader(Subset(dataset, all_idx.tolist()), batch_size=args.batch_size, shuffle=False),
        args,
        device,
        method_name,
        "final_trainval",
    )
    tokenizer.save_pretrained(out_dir / "tokenizer")
    final_model.encoder.save_pretrained(out_dir / "encoder")

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    y_true, y_pred, proba = predict_proba(final_model, test_loader, device)
    test_metrics = classification_metrics(y_true, y_pred, proba)
    test_metrics.update({"method": method_name, "split": "untouched_test", "fold": -1, "validation_study": ", ".join(sorted(test_frame["study_id"].astype(str).unique()))})
    metric_rows.append(test_metrics)
    print(f"{method_name} | untouched_test: {format_metrics(test_metrics)}", flush=True)
    test_predictions = save_prediction_frame(test_frame, "untouched_test", y_true, y_pred, proba)

    return save_common_outputs(
        output_dir=args.output_dir,
        method_name=method_name,
        model=final_model,
        model_payload={"model_id": model_id, "method_key": method_key, "model_type": "hf_sequence_encoder_classifier"},
        metric_rows=metric_rows,
        cv_predictions=cv_predictions,
        test_predictions=test_predictions,
        y_test=y_true,
        test_pred=y_pred,
        args=args,
        extra_meta={"model_id": model_id, "encoder_trainable": True},
    )


def train_pretrained_gnn(method_key, frame, test_frame, args, device: torch.device) -> dict:
    from torch_geometric.loader import DataLoader as GraphLoader

    method_name, repo_name, model_rel, class_name, ckpt_rel = GNN_METHODS[method_key]
    repo = args.repo_dir / repo_name
    model_path = repo / model_rel
    ckpt_path = repo / ckpt_rel
    if not model_path.exists() or not ckpt_path.exists():
        return write_unavailable(args.output_dir, method_name, f"Missing model file or checkpoint under {repo}")

    encoder_class = load_class(model_path, class_name, f"{method_key}_finetune_classification")
    graphs, keep = build_pretrained_graphs(frame, args.target_col)
    frame = frame.iloc[keep].reset_index(drop=True)
    test_graphs, test_keep = build_pretrained_graphs(test_frame, args.target_col)
    test_frame = test_frame.iloc[test_keep].reset_index(drop=True)
    cv_predictions, metric_rows = [], []

    print(f"{method_name}: trainval graphs={len(graphs):,}; untouched-test graphs={len(test_graphs):,}", flush=True)
    for fold, train_idx, val_idx in cv_fold_indices(frame):
        validation_study = ", ".join(sorted(frame.iloc[val_idx]["study_id"].astype(str).unique()))
        print("", flush=True)
        print(f"{method_name} | fold {fold}: validation study = {validation_study}", flush=True)
        print(f"{method_name} | fold {fold}: train={len(train_idx):,}, val={len(val_idx):,}", flush=True)
        encoder = encoder_class(num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0, pool="mean").to(device)
        msg = encoder.load_state_dict(load_state_dict(ckpt_path, device), strict=False)
        print("Missing keys:", len(msg.missing_keys), "Unexpected keys:", len(msg.unexpected_keys), flush=True)
        model = PretrainedGNNClassifier(encoder).to(device)
        train_loader = GraphLoader(Subset(graphs, train_idx.tolist()), batch_size=args.batch_size, shuffle=True)
        val_loader = GraphLoader(Subset(graphs, val_idx.tolist()), batch_size=args.batch_size, shuffle=False)
        best_loss = train_loop(model, train_loader, val_loader, args, device, method_name, f"fold_{fold}")
        y_true, y_pred, proba = predict_proba(model, val_loader, device)
        metrics = classification_metrics(y_true, y_pred, proba)
        metrics.update({"method": method_name, "split": f"fold_{fold}", "fold": fold, "validation_study": validation_study, "best_val_loss": best_loss})
        metric_rows.append(metrics)
        print(f"{method_name} | fold_{fold}: {format_metrics(metrics)}", flush=True)
        cv_predictions.append(save_prediction_frame(frame.iloc[val_idx], f"fold_{fold}", y_true, y_pred, proba, cv_fold=fold))

    fold_metrics = pd.DataFrame(metric_rows)
    cv_mean = fold_metrics.drop(columns=["method", "split", "fold", "validation_study"], errors="ignore").mean(numeric_only=True).to_dict()
    cv_mean.update({"method": method_name, "split": "cv_mean", "fold": -1, "validation_study": "leave_one_study_out"})
    cv_std = fold_metrics.drop(columns=["method", "split", "fold", "validation_study"], errors="ignore").std(numeric_only=True).to_dict()
    cv_std.update({"method": method_name, "split": "cv_std", "fold": -1, "validation_study": "leave_one_study_out"})
    metric_rows.extend([cv_mean, cv_std])
    print("", flush=True)
    print(f"{method_name} | cv_mean: {format_metrics(cv_mean)}", flush=True)

    print(f"{method_name}: fitting final model on all trainval graphs...", flush=True)
    final_encoder = encoder_class(num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0, pool="mean").to(device)
    final_encoder.load_state_dict(load_state_dict(ckpt_path, device), strict=False)
    final_model = PretrainedGNNClassifier(final_encoder).to(device)
    all_idx = np.arange(len(graphs))
    train_loop(
        final_model,
        GraphLoader(Subset(graphs, all_idx.tolist()), batch_size=args.batch_size, shuffle=True),
        GraphLoader(Subset(graphs, all_idx.tolist()), batch_size=args.batch_size, shuffle=False),
        args,
        device,
        method_name,
        "final_trainval",
    )
    test_loader = GraphLoader(test_graphs, batch_size=args.batch_size, shuffle=False)
    y_true, y_pred, proba = predict_proba(final_model, test_loader, device)
    test_metrics = classification_metrics(y_true, y_pred, proba)
    test_metrics.update({"method": method_name, "split": "untouched_test", "fold": -1, "validation_study": ", ".join(sorted(test_frame["study_id"].astype(str).unique()))})
    metric_rows.append(test_metrics)
    print(f"{method_name} | untouched_test: {format_metrics(test_metrics)}", flush=True)
    test_predictions = save_prediction_frame(test_frame, "untouched_test", y_true, y_pred, proba)

    return save_common_outputs(
        output_dir=args.output_dir,
        method_name=method_name,
        model=final_model,
        model_payload={"method_key": method_key, "model_type": "pretrained_gnn_classifier", "repo": str(repo), "checkpoint": str(ckpt_path)},
        metric_rows=metric_rows,
        cv_predictions=cv_predictions,
        test_predictions=test_predictions,
        y_test=y_true,
        test_pred=y_pred,
        args=args,
        extra_meta={"repo": str(repo), "checkpoint": str(ckpt_path), "encoder_trainable": True},
    )


def normalize_method_key(method: str) -> str:
    key = method.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "chemberta_mtr_finetuned": "chemberta_mtr",
        "molformer_finetuned": "molformer",
        "agile_finetuned": "agile",
        "molclr_gin_finetuned": "molclr_gin",
        "lion_finetuned": "lion",
        "uni_mol_finetuned": "uni_mol",
        "unimol_finetuned": "uni_mol",
    }
    return aliases.get(key, key)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    frame = read_table(args.trainval_subset)
    test_frame = read_table(args.test_subset)
    for name, table in [("trainval", frame), ("untouched_test", test_frame)]:
        table[args.target_col] = pd.to_numeric(table[args.target_col], errors="coerce")
        before = len(table)
        table.dropna(subset=[args.target_col], inplace=True)
        table[args.target_col] = table[args.target_col].astype(int)
        ok = valid_smiles_mask(smiles_series(table))
        table.drop(index=table.index[~ok], inplace=True)
        table.reset_index(drop=True, inplace=True)
        print(f"{name}: {before:,} -> {len(table):,} rows with valid SMILES and class labels", flush=True)
        print(f"{name} class counts:", dict(table[args.target_col].value_counts().sort_index()), flush=True)

    manifest_rows = []
    for method in args.methods:
        key = normalize_method_key(method)
        print("", flush=True)
        print("=" * 80, flush=True)
        print(method, flush=True)
        print("=" * 80, flush=True)
        try:
            if key in HF_METHODS:
                manifest_rows.append(train_hf(key, frame.copy(), test_frame.copy(), args, device))
            elif key in GNN_METHODS:
                manifest_rows.append(train_pretrained_gnn(key, frame.copy(), test_frame.copy(), args, device))
            elif key in UNSUPPORTED:
                name, reason = UNSUPPORTED[key]
                manifest_rows.append(write_unavailable(args.output_dir, name, reason))
            else:
                manifest_rows.append(write_unavailable(args.output_dir, method, "Unknown method key."))
        except Exception as exc:
            name = HF_METHODS.get(key, GNN_METHODS.get(key, (method,)))[0]
            manifest_rows.append(write_unavailable(args.output_dir, name, repr(exc)))

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = Path(args.output_dir) / "finetuned_classification_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print("", flush=True)
    print("Saved fine-tuned classification manifest:", manifest_path, flush=True)
    print(manifest.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
