from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from .features import smiles_series, valid_smiles_mask
from .io_utils import ensure_dir, method_dir, now_utc, read_table, write_json


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
    "uni_mol": ("Uni-Mol-Finetuned", "Uni-Mol fine-tuning needs the official Uni-Mol training interface/checkpoint workflow; this notebook supports Uni-Mol as a frozen encoder."),
    "unimol": ("Uni-Mol-Finetuned", "Uni-Mol fine-tuning needs the official Uni-Mol training interface/checkpoint workflow; this notebook supports Uni-Mol as a frozen encoder."),
}
DEFAULT_METHODS = ["agile", "lion", "chemberta_mtr", "molformer", "molclr_gin", "uni_mol"]
CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = {0: "low", 1: "medium", 2: "high"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune pretrained molecular encoders with optional extra data layers.")
    parser.add_argument("--trainval-subset", type=Path)
    parser.add_argument("--train-subset", type=Path)
    parser.add_argument("--validation-subset", type=Path)
    parser.add_argument("--test-subset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-dir", default=Path("/content/notebooks/encoder_repos"), type=Path)
    parser.add_argument("--target-col", default="delivery_class_3")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--tabular-cols", nargs="*", default=[])
    parser.add_argument("--cv-description", default="random stratified 5-fold CV with random held-out test set")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    random_mode = args.train_subset is not None or args.validation_subset is not None
    cv_mode = args.trainval_subset is not None
    if random_mode == cv_mode:
        raise ValueError("Use either --trainval-subset for CV mode, or --train-subset plus --validation-subset for random-split mode.")
    if random_mode and (args.train_subset is None or args.validation_subset is None):
        raise ValueError("Random-split mode requires both --train-subset and --validation-subset.")
    return args


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def id_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in ["record_id", "study_id", "lnp_id", "il_id", "il_smiles", "canonical_il_smiles", "delivery", "delivery_class_label"] if c in frame.columns]


def clean_frame(path: Path, target_col: str, split_name: str) -> pd.DataFrame:
    frame = read_table(path).copy()
    if target_col not in frame.columns:
        raise ValueError(f"{split_name} subset is missing target column: {target_col}")
    frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce")
    before = len(frame)
    ok = frame[target_col].isin(CLASS_LABELS) & valid_smiles_mask(smiles_series(frame))
    frame = frame.loc[ok].reset_index(drop=True)
    frame[target_col] = frame[target_col].astype(int)
    print(f"{split_name}: {before:,} -> {len(frame):,} rows with valid SMILES and class labels", flush=True)
    print(f"{split_name} class counts: {dict(frame[target_col].value_counts().sort_index())}", flush=True)
    return frame


def fit_tabular_preprocessor(frame: pd.DataFrame, cols: list[str]) -> Pipeline | None:
    if not cols:
        return None
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing tabular column(s): {missing}")
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    pipe.fit(frame[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32))
    return pipe


def transform_tabular(frame: pd.DataFrame, cols: list[str], preprocessor: Pipeline | None) -> np.ndarray | None:
    if not cols:
        return None
    if preprocessor is None:
        raise ValueError("Tabular columns supplied without a fitted preprocessor.")
    values = frame[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    return preprocessor.transform(values).astype(np.float32)


class SmilesTabularDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, target_col: str, tokenizer, tabular_values: np.ndarray | None = None, max_length: int = 256):
        self.frame = frame.reset_index(drop=True)
        self.smiles = smiles_series(self.frame).tolist()
        self.y = self.frame[target_col].to_numpy(dtype=np.int64)
        self.tokenizer = tokenizer
        self.tabular_values = tabular_values
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.frame)

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
        item["sample_id"] = torch.tensor(idx, dtype=torch.long)
        if self.tabular_values is not None:
            item["tabular"] = torch.tensor(self.tabular_values[idx], dtype=torch.float32)
        return item


class HFMultimodalClassifier(nn.Module):
    def __init__(self, model_id: str, trust_remote_code: bool, tabular_dim: int = 0, dropout: float = 0.15):
        super().__init__()
        from transformers import AutoModel, AutoModelForSequenceClassification

        try:
            self.encoder = AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        except Exception:
            self.encoder = AutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        hidden = getattr(self.encoder.config, "hidden_size", None) or getattr(self.encoder.config, "d_model", None)
        if hidden is None:
            raise ValueError(f"Cannot infer hidden size for {model_id}")
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden + tabular_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, len(CLASS_LABELS)),
        ) if tabular_dim else nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, len(CLASS_LABELS)))

    def forward(self, **batch):
        labels = batch.pop("labels", None)
        tabular = batch.pop("tabular", None)
        batch.pop("sample_id", None)
        out = self.encoder(**batch, output_hidden_states=True, return_dict=True)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            hidden = out.hidden_states[-1]
        mask = batch.get("attention_mask", torch.ones(hidden.shape[:2], device=hidden.device)).unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        if tabular is not None:
            pooled = torch.cat([pooled, tabular.float()], dim=1)
        logits = self.head(pooled)
        if labels is None:
            return logits
        return logits, F.cross_entropy(logits, labels)


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


def pretrained_graph(smiles: str, y: int, sample_id: int, tabular: np.ndarray | None = None):
    from rdkit import Chem
    from torch_geometric.data import Data

    atom_list = list(range(1, 119))
    chirality_list = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    ]
    bond_list = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bonddir_list = [Chem.rdchem.BondDir.NONE, Chem.rdchem.BondDir.ENDUPRIGHT, Chem.rdchem.BondDir.ENDDOWNRIGHT]
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
    graph = Data(
        x=torch.tensor(atoms, dtype=torch.long),
        edge_index=torch.tensor([row, col], dtype=torch.long) if row else torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.tensor(edge_feat, dtype=torch.long).view(-1, 2),
        y=torch.tensor([int(y)], dtype=torch.long),
        sample_id=torch.tensor([int(sample_id)], dtype=torch.long),
    )
    if tabular is not None:
        graph.tabular = torch.tensor(tabular.reshape(1, -1), dtype=torch.float32)
    return graph


def build_pretrained_graphs(frame: pd.DataFrame, target_col: str, tabular_values: np.ndarray | None = None):
    graphs, keep = [], []
    smiles = smiles_series(frame).tolist()
    y = frame[target_col].to_numpy(dtype=int)
    for idx, (smi, yi) in enumerate(zip(smiles, y)):
        tab = tabular_values[idx] if tabular_values is not None else None
        graph = pretrained_graph(smi, int(yi), idx, tab)
        if graph is not None:
            graphs.append(graph)
            keep.append(idx)
    return graphs, frame.iloc[keep].reset_index(drop=True)


class PretrainedGNNMultimodalClassifier(nn.Module):
    def __init__(self, encoder, tabular_dim: int = 0, dropout: float = 0.15):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512 + tabular_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, len(CLASS_LABELS)),
        ) if tabular_dim else nn.Sequential(nn.Dropout(dropout), nn.Linear(512, len(CLASS_LABELS)))

    def forward(self, batch):
        h, _ = self.encoder(batch)
        tabular = getattr(batch, "tabular", None)
        if tabular is not None:
            if tabular.dim() == 1:
                tabular = tabular.view(h.size(0), -1)
            h = torch.cat([h, tabular.float()], dim=1)
        return self.head(h)


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


def train_loop(model, train_loader, val_loader, args, device: torch.device, method_name: str, split_label: str):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
                loss = F.cross_entropy(model(batch), batch.y.view(-1).long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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
            print(f"{method_name} | {split_label} | epoch {epoch:03d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, best_val={best_val:.4f}", flush=True)
        if bad_epochs >= args.patience:
            print(f"{method_name} | {split_label}: early stopping at epoch {epoch}", flush=True)
            break
    if best_state is not None:
        with torch.inference_mode():
            model.load_state_dict(best_state)
    return best_val


def evaluate_loss(model, loader, device: torch.device) -> float:
    losses = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            if isinstance(batch, dict):
                batch = {k: v.to(device) for k, v in batch.items()}
                _, loss = model(**batch)
            else:
                batch = batch.to(device)
                loss = F.cross_entropy(model(batch), batch.y.view(-1).long())
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def predict_proba(model, loader, device: torch.device):
    y_true, probs, sample_ids = [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            if isinstance(batch, dict):
                labels = batch["labels"].detach().cpu().numpy()
                ids = batch["sample_id"].detach().cpu().numpy()
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                logits = model(**inputs)
            else:
                batch = batch.to(device)
                labels = batch.y.view(-1).detach().cpu().numpy()
                ids = batch.sample_id.view(-1).detach().cpu().numpy()
                logits = model(batch)
            y_true.append(labels)
            sample_ids.append(ids)
            probs.append(F.softmax(logits, dim=1).detach().cpu().numpy())
    y_true = np.concatenate(y_true).astype(int)
    proba = np.vstack(probs)
    row_sums = proba.sum(axis=1, keepdims=True)
    proba = np.divide(proba, row_sums, out=np.full_like(proba, 1.0 / proba.shape[1]), where=row_sums > 0)
    pred = np.argmax(proba, axis=1)
    return y_true, pred, proba, np.concatenate(sample_ids).astype(int)


def save_prediction_frame(frame: pd.DataFrame, split: str, y_true, y_pred, proba, sample_ids, cv_fold=None) -> pd.DataFrame:
    out = frame.iloc[sample_ids][id_columns(frame)].reset_index(drop=True).copy()
    out["split"] = split
    if cv_fold is not None:
        out["cv_fold"] = cv_fold
    out["y_true"] = y_true
    out["y_pred"] = y_pred
    for cls in CLASS_LABELS:
        out[f"prob_{CLASS_NAMES[cls]}"] = proba[:, cls]
    return out


def save_outputs(
    output_dir: Path,
    method_name: str,
    model,
    metric_rows: list[dict],
    prediction_tables: dict[str, pd.DataFrame],
    test_y,
    test_pred,
    args,
    meta: dict,
    tabular_preprocessor=None,
):
    out_dir = method_dir(output_dir, method_name)
    torch.save({"state_dict": model.state_dict(), **meta}, out_dir / "model.pt")
    if tabular_preprocessor is not None:
        joblib.dump(tabular_preprocessor, out_dir / "tabular_preprocessor.joblib")
    for name, table in prediction_tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(out_dir / "metrics.csv", index=False)
    cm = confusion_matrix(test_y, test_pred, labels=CLASS_LABELS)
    pd.DataFrame(cm, index=[f"true_{CLASS_NAMES[c]}" for c in CLASS_LABELS], columns=[f"pred_{CLASS_NAMES[c]}" for c in CLASS_LABELS]).to_csv(out_dir / "test_confusion_matrix.csv")
    per_class = precision_recall_fscore_support(test_y, test_pred, labels=CLASS_LABELS, zero_division=0)
    pd.DataFrame({
        "class": [CLASS_NAMES[c] for c in CLASS_LABELS],
        "precision": per_class[0],
        "recall": per_class[1],
        "f1": per_class[2],
        "support": per_class[3],
    }).to_csv(out_dir / "test_per_class_metrics.csv", index=False)
    write_json(out_dir / "feature_schema.json", {
        "molecular_input": "fine-tuned pretrained molecular encoder",
        "tabular_columns": args.tabular_cols,
        "n_tabular_features": len(args.tabular_cols),
        "class_names": CLASS_NAMES,
    })
    write_json(out_dir / "metadata.json", {
        "method": method_name,
        "status": "trained",
        "target": args.target_col,
        "class_names": CLASS_NAMES,
        "trainval_path": str(args.trainval_subset) if args.trainval_subset else None,
        "train_path": str(args.train_subset) if args.train_subset else None,
        "validation_path": str(args.validation_subset) if args.validation_subset else None,
        "test_path": str(args.test_subset),
        "tabular_columns": args.tabular_cols,
        "test_used_for_training": False,
        "epochs_max": args.epochs,
        "early_stopping_patience": args.patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "trained_at_utc": now_utc(),
        "output_dir": str(out_dir),
        "metadata": meta,
    })
    print(f"{method_name}: saved outputs -> {out_dir}", flush=True)
    return out_dir


def write_unavailable(output_dir: Path, method_name: str, reason: str) -> dict:
    out_dir = method_dir(output_dir, method_name)
    write_json(out_dir / "metadata.json", {"method": method_name, "status": "not_trained", "reason": reason, "trained_at_utc": now_utc()})
    print(f"Skipped {method_name}: {reason}", flush=True)
    return {"method": method_name, "status": "not_trained", "output_dir": str(out_dir), "reason": reason}


def metrics_row(method_name: str, status: str, out_dir: Path, primary: dict, test: dict, prefix: str) -> dict:
    row = {"method": method_name, "status": status, "output_dir": str(out_dir)}
    for metric in ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc_ovr_macro", "average_precision_macro", "mcc", "log_loss"]:
        row[f"{prefix}_{metric}"] = primary.get(metric)
        row[f"test_{metric}"] = test.get(metric)
    return row


def cv_mean_std(metric_rows: list[dict]):
    fold_df = pd.DataFrame([r for r in metric_rows if str(r.get("split", "")).startswith("fold_")])
    cv_mean = fold_df.drop(columns=["method", "split", "fold", "validation_study"], errors="ignore").mean(numeric_only=True).to_dict()
    cv_mean.update({"split": "cv_mean", "fold": -1})
    cv_std = fold_df.drop(columns=["method", "split", "fold", "validation_study"], errors="ignore").std(numeric_only=True).to_dict()
    cv_std.update({"split": "cv_std", "fold": -1})
    return cv_mean, cv_std


def hf_dataset(frame, target_col, tokenizer, tabular):
    return SmilesTabularDataset(frame, target_col, tokenizer, tabular)


def train_hf_random(key, train, val, test, args, device, tabular_dim):
    from transformers import AutoTokenizer

    method_name, model_id, trust_remote_code = HF_METHODS[key]
    method_name = f"{method_name} Random Split"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    prep = fit_tabular_preprocessor(train, args.tabular_cols)
    train_ds = hf_dataset(train, args.target_col, tokenizer, transform_tabular(train, args.tabular_cols, prep))
    val_ds = hf_dataset(val, args.target_col, tokenizer, transform_tabular(val, args.tabular_cols, prep))
    test_ds = hf_dataset(test, args.target_col, tokenizer, transform_tabular(test, args.tabular_cols, prep))
    model = HFMultimodalClassifier(model_id, trust_remote_code, tabular_dim=tabular_dim).to(device)
    train_loop(model, DataLoader(train_ds, batch_size=args.batch_size, shuffle=True), DataLoader(val_ds, batch_size=args.batch_size), args, device, method_name, "random_split")
    yv, pv, probv, idv = predict_proba(model, DataLoader(val_ds, batch_size=args.batch_size), device)
    yt, pt, probt, idt = predict_proba(model, DataLoader(test_ds, batch_size=args.batch_size), device)
    val_metrics = classification_metrics(yv, pv, probv); val_metrics.update({"method": method_name, "split": "validation"})
    test_metrics = classification_metrics(yt, pt, probt); test_metrics.update({"method": method_name, "split": "test"})
    print(f"{method_name} | validation: {format_metrics(val_metrics)}", flush=True)
    print(f"{method_name} | test: {format_metrics(test_metrics)}", flush=True)
    out_dir = method_dir(args.output_dir, method_name)
    tokenizer.save_pretrained(out_dir / "tokenizer")
    model.encoder.save_pretrained(out_dir / "encoder")
    save_outputs(args.output_dir, method_name, model, [val_metrics, test_metrics], {
        "validation_predictions": save_prediction_frame(val, "validation", yv, pv, probv, idv),
        "test_predictions": save_prediction_frame(test, "test", yt, pt, probt, idt),
    }, yt, pt, args, {"model_id": model_id, "method_key": key, "model_type": "hf_sequence_encoder_classifier", "encoder_trainable": True}, prep)
    return metrics_row(method_name, "trained", out_dir, val_metrics, test_metrics, "validation")


def train_gnn_random(key, train, val, test, args, device, tabular_dim):
    from torch_geometric.loader import DataLoader as GraphLoader

    method_name, repo_name, model_rel, class_name, ckpt_rel = GNN_METHODS[key]
    method_name = f"{method_name} Random Split"
    repo = args.repo_dir / repo_name
    model_path, ckpt_path = repo / model_rel, repo / ckpt_rel
    if not model_path.exists() or not ckpt_path.exists():
        return write_unavailable(args.output_dir, method_name, f"Missing model file or checkpoint under {repo}")
    encoder_class = load_class(model_path, class_name, f"{key}_finetune_multimodal_random")
    prep = fit_tabular_preprocessor(train, args.tabular_cols)
    train_graphs, train = build_pretrained_graphs(train, args.target_col, transform_tabular(train, args.tabular_cols, prep))
    val_graphs, val = build_pretrained_graphs(val, args.target_col, transform_tabular(val, args.tabular_cols, prep))
    test_graphs, test = build_pretrained_graphs(test, args.target_col, transform_tabular(test, args.tabular_cols, prep))
    encoder = encoder_class(num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0, pool="mean").to(device)
    msg = encoder.load_state_dict(load_state_dict(ckpt_path, device), strict=False)
    print("Missing keys:", len(msg.missing_keys), "Unexpected keys:", len(msg.unexpected_keys), flush=True)
    model = PretrainedGNNMultimodalClassifier(encoder, tabular_dim=tabular_dim).to(device)
    train_loop(model, GraphLoader(train_graphs, batch_size=args.batch_size, shuffle=True), GraphLoader(val_graphs, batch_size=args.batch_size), args, device, method_name, "random_split")
    yv, pv, probv, idv = predict_proba(model, GraphLoader(val_graphs, batch_size=args.batch_size), device)
    yt, pt, probt, idt = predict_proba(model, GraphLoader(test_graphs, batch_size=args.batch_size), device)
    val_metrics = classification_metrics(yv, pv, probv); val_metrics.update({"method": method_name, "split": "validation"})
    test_metrics = classification_metrics(yt, pt, probt); test_metrics.update({"method": method_name, "split": "test"})
    print(f"{method_name} | validation: {format_metrics(val_metrics)}", flush=True)
    print(f"{method_name} | test: {format_metrics(test_metrics)}", flush=True)
    out_dir = save_outputs(args.output_dir, method_name, model, [val_metrics, test_metrics], {
        "validation_predictions": save_prediction_frame(val, "validation", yv, pv, probv, idv),
        "test_predictions": save_prediction_frame(test, "test", yt, pt, probt, idt),
    }, yt, pt, args, {"method_key": key, "model_type": "pretrained_gnn_classifier", "repo": str(repo), "checkpoint": str(ckpt_path), "encoder_trainable": True}, prep)
    return metrics_row(method_name, "trained", out_dir, val_metrics, test_metrics, "validation")


def train_hf_cv(key, trainval, test, args, device, tabular_dim):
    from transformers import AutoTokenizer

    method_name, model_id, trust_remote_code = HF_METHODS[key]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    metric_rows, cv_predictions = [], []
    folds = sorted(int(f) for f in trainval["cv_fold"].dropna().unique() if int(f) >= 0)
    for fold in folds:
        val_mask = trainval["cv_fold"].astype(int).to_numpy() == fold
        tr = trainval.loc[~val_mask].reset_index(drop=True)
        va = trainval.loc[val_mask].reset_index(drop=True)
        prep = fit_tabular_preprocessor(tr, args.tabular_cols)
        tr_ds = hf_dataset(tr, args.target_col, tokenizer, transform_tabular(tr, args.tabular_cols, prep))
        va_ds = hf_dataset(va, args.target_col, tokenizer, transform_tabular(va, args.tabular_cols, prep))
        model = HFMultimodalClassifier(model_id, trust_remote_code, tabular_dim=tabular_dim).to(device)
        print(f"\n{method_name} | fold {fold}: train={len(tr):,}, val={len(va):,}", flush=True)
        best = train_loop(model, DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True), DataLoader(va_ds, batch_size=args.batch_size), args, device, method_name, f"fold_{fold}")
        y, p, proba, ids = predict_proba(model, DataLoader(va_ds, batch_size=args.batch_size), device)
        metrics = classification_metrics(y, p, proba); metrics.update({"method": method_name, "split": f"fold_{fold}", "fold": fold, "best_val_loss": best})
        metric_rows.append(metrics)
        cv_predictions.append(save_prediction_frame(va, f"fold_{fold}", y, p, proba, ids, cv_fold=fold))
        print(f"{method_name} | fold_{fold}: {format_metrics(metrics)}", flush=True)
    cv_mean, cv_std = cv_mean_std(metric_rows)
    cv_mean.update({"method": method_name}); cv_std.update({"method": method_name})
    metric_rows += [cv_mean, cv_std]
    prep = fit_tabular_preprocessor(trainval, args.tabular_cols)
    trainval_ds = hf_dataset(trainval, args.target_col, tokenizer, transform_tabular(trainval, args.tabular_cols, prep))
    test_ds = hf_dataset(test, args.target_col, tokenizer, transform_tabular(test, args.tabular_cols, prep))
    final = HFMultimodalClassifier(model_id, trust_remote_code, tabular_dim=tabular_dim).to(device)
    train_loop(final, DataLoader(trainval_ds, batch_size=args.batch_size, shuffle=True), DataLoader(trainval_ds, batch_size=args.batch_size), args, device, method_name, "final_trainval")
    yt, pt, probt, idt = predict_proba(final, DataLoader(test_ds, batch_size=args.batch_size), device)
    test_metrics = classification_metrics(yt, pt, probt); test_metrics.update({"method": method_name, "split": "test", "fold": -1})
    metric_rows.append(test_metrics)
    print(f"{method_name} | cv_mean: {format_metrics(cv_mean)}", flush=True)
    print(f"{method_name} | test: {format_metrics(test_metrics)}", flush=True)
    out_dir = method_dir(args.output_dir, method_name)
    tokenizer.save_pretrained(out_dir / "tokenizer")
    final.encoder.save_pretrained(out_dir / "encoder")
    save_outputs(args.output_dir, method_name, final, metric_rows, {
        "cv_predictions": pd.concat(cv_predictions, ignore_index=True),
        "test_predictions": save_prediction_frame(test, "test", yt, pt, probt, idt),
    }, yt, pt, args, {"model_id": model_id, "method_key": key, "model_type": "hf_sequence_encoder_classifier", "encoder_trainable": True}, prep)
    return metrics_row(method_name, "trained", out_dir, cv_mean, test_metrics, "cv")


def train_gnn_cv(key, trainval, test, args, device, tabular_dim):
    from torch_geometric.loader import DataLoader as GraphLoader

    method_name, repo_name, model_rel, class_name, ckpt_rel = GNN_METHODS[key]
    repo = args.repo_dir / repo_name
    model_path, ckpt_path = repo / model_rel, repo / ckpt_rel
    if not model_path.exists() or not ckpt_path.exists():
        return write_unavailable(args.output_dir, method_name, f"Missing model file or checkpoint under {repo}")
    encoder_class = load_class(model_path, class_name, f"{key}_finetune_multimodal_cv")
    metric_rows, cv_predictions = [], []
    folds = sorted(int(f) for f in trainval["cv_fold"].dropna().unique() if int(f) >= 0)
    for fold in folds:
        val_mask = trainval["cv_fold"].astype(int).to_numpy() == fold
        tr = trainval.loc[~val_mask].reset_index(drop=True)
        va = trainval.loc[val_mask].reset_index(drop=True)
        prep = fit_tabular_preprocessor(tr, args.tabular_cols)
        tr_graphs, tr = build_pretrained_graphs(tr, args.target_col, transform_tabular(tr, args.tabular_cols, prep))
        va_graphs, va = build_pretrained_graphs(va, args.target_col, transform_tabular(va, args.tabular_cols, prep))
        print(f"\n{method_name} | fold {fold}: train={len(tr_graphs):,}, val={len(va_graphs):,}", flush=True)
        encoder = encoder_class(num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0, pool="mean").to(device)
        msg = encoder.load_state_dict(load_state_dict(ckpt_path, device), strict=False)
        print("Missing keys:", len(msg.missing_keys), "Unexpected keys:", len(msg.unexpected_keys), flush=True)
        model = PretrainedGNNMultimodalClassifier(encoder, tabular_dim=tabular_dim).to(device)
        best = train_loop(model, GraphLoader(tr_graphs, batch_size=args.batch_size, shuffle=True), GraphLoader(va_graphs, batch_size=args.batch_size), args, device, method_name, f"fold_{fold}")
        y, p, proba, ids = predict_proba(model, GraphLoader(va_graphs, batch_size=args.batch_size), device)
        metrics = classification_metrics(y, p, proba); metrics.update({"method": method_name, "split": f"fold_{fold}", "fold": fold, "best_val_loss": best})
        metric_rows.append(metrics)
        cv_predictions.append(save_prediction_frame(va, f"fold_{fold}", y, p, proba, ids, cv_fold=fold))
        print(f"{method_name} | fold_{fold}: {format_metrics(metrics)}", flush=True)
    cv_mean, cv_std = cv_mean_std(metric_rows)
    cv_mean.update({"method": method_name}); cv_std.update({"method": method_name})
    metric_rows += [cv_mean, cv_std]
    prep = fit_tabular_preprocessor(trainval, args.tabular_cols)
    tv_graphs, trainval = build_pretrained_graphs(trainval, args.target_col, transform_tabular(trainval, args.tabular_cols, prep))
    test_graphs, test = build_pretrained_graphs(test, args.target_col, transform_tabular(test, args.tabular_cols, prep))
    encoder = encoder_class(num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0, pool="mean").to(device)
    encoder.load_state_dict(load_state_dict(ckpt_path, device), strict=False)
    final = PretrainedGNNMultimodalClassifier(encoder, tabular_dim=tabular_dim).to(device)
    train_loop(final, GraphLoader(tv_graphs, batch_size=args.batch_size, shuffle=True), GraphLoader(tv_graphs, batch_size=args.batch_size), args, device, method_name, "final_trainval")
    yt, pt, probt, idt = predict_proba(final, GraphLoader(test_graphs, batch_size=args.batch_size), device)
    test_metrics = classification_metrics(yt, pt, probt); test_metrics.update({"method": method_name, "split": "test", "fold": -1})
    metric_rows.append(test_metrics)
    print(f"{method_name} | cv_mean: {format_metrics(cv_mean)}", flush=True)
    print(f"{method_name} | test: {format_metrics(test_metrics)}", flush=True)
    out_dir = save_outputs(args.output_dir, method_name, final, metric_rows, {
        "cv_predictions": pd.concat(cv_predictions, ignore_index=True),
        "test_predictions": save_prediction_frame(test, "test", yt, pt, probt, idt),
    }, yt, pt, args, {"method_key": key, "model_type": "pretrained_gnn_classifier", "repo": str(repo), "checkpoint": str(ckpt_path), "encoder_trainable": True}, prep)
    return metrics_row(method_name, "trained", out_dir, cv_mean, test_metrics, "cv")


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tabular_dim = len(args.tabular_cols)
    print("Device:", device, flush=True)
    print("Methods:", ", ".join(args.methods), flush=True)
    print("Extra-layer columns:", tabular_dim, flush=True)
    if args.tabular_cols:
        print(", ".join(args.tabular_cols), flush=True)

    manifest_rows = []
    if args.trainval_subset is not None:
        trainval = clean_frame(args.trainval_subset, args.target_col, "trainval")
        test = clean_frame(args.test_subset, args.target_col, "test")
        for method in args.methods:
            key = normalize_method_key(method)
            print("\n" + "=" * 80, flush=True)
            print(method, flush=True)
            print("=" * 80, flush=True)
            try:
                if key in HF_METHODS:
                    manifest_rows.append(train_hf_cv(key, trainval.copy(), test.copy(), args, device, tabular_dim))
                elif key in GNN_METHODS:
                    manifest_rows.append(train_gnn_cv(key, trainval.copy(), test.copy(), args, device, tabular_dim))
                elif key in UNSUPPORTED:
                    name, reason = UNSUPPORTED[key]
                    manifest_rows.append(write_unavailable(args.output_dir, name, reason))
                else:
                    manifest_rows.append(write_unavailable(args.output_dir, method, "Unknown method key."))
            except Exception as exc:
                name = HF_METHODS.get(key, GNN_METHODS.get(key, (method,)))[0]
                manifest_rows.append(write_unavailable(args.output_dir, name, repr(exc)))
    else:
        train = clean_frame(args.train_subset, args.target_col, "train")
        val = clean_frame(args.validation_subset, args.target_col, "validation")
        test = clean_frame(args.test_subset, args.target_col, "test")
        for method in args.methods:
            key = normalize_method_key(method)
            print("\n" + "=" * 80, flush=True)
            print(method, flush=True)
            print("=" * 80, flush=True)
            try:
                if key in HF_METHODS:
                    manifest_rows.append(train_hf_random(key, train.copy(), val.copy(), test.copy(), args, device, tabular_dim))
                elif key in GNN_METHODS:
                    manifest_rows.append(train_gnn_random(key, train.copy(), val.copy(), test.copy(), args, device, tabular_dim))
                elif key in UNSUPPORTED:
                    name, reason = UNSUPPORTED[key]
                    manifest_rows.append(write_unavailable(args.output_dir, f"{name} Random Split", reason))
                else:
                    manifest_rows.append(write_unavailable(args.output_dir, method, "Unknown method key."))
            except Exception as exc:
                name = HF_METHODS.get(key, GNN_METHODS.get(key, (method,)))[0]
                manifest_rows.append(write_unavailable(args.output_dir, name, repr(exc)))

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = args.output_dir / "finetuned_multimodal_classification_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print("\nSaved fine-tuned multimodal classification manifest:", manifest_path, flush=True)
    print(manifest.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
