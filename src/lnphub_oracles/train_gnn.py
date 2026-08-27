from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from .features import smiles_series
from .graph_data import build_graphs
from .io_utils import method_dir, now_utc, read_table, write_json
from .metrics import regression_metrics
from .splits import cv_fold_indices
from .torch_models import DMPNNRegressor, GCNRegressor, GINRegressor


MODELS = {
    "gcn": ("GCN", GCNRegressor),
    "gin": ("GIN", GINRegressor),
    "dmpnn": ("D-MPNN", DMPNNRegressor),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train graph neural network LNPHub oracles.")
    parser.add_argument("--subset", required=True, type=Path)
    parser.add_argument("--test-subset", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-col", default="delivery")
    parser.add_argument("--methods", nargs="+", default=list(MODELS))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def train_one(model, graphs, split_idx, args, device):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()
    loaders = {
        k: DataLoader(Subset(graphs, idx.tolist()), batch_size=args.batch_size, shuffle=(k == "train"))
        for k, idx in split_idx.items()
    }
    best_state, best_val = None, float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in loaders["train"]:
            batch = batch.to(device)
            opt.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch.y.view(-1))
            loss.backward()
            opt.step()
        model.eval()
        vals = []
        with torch.inference_mode():
            for batch in loaders["val"]:
                batch = batch.to(device)
                vals.append(loss_fn(model(batch), batch.y.view(-1)).item())
        val_loss = float(np.mean(vals)) if vals else float("inf")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0 or epoch == 1:
            print(f"epoch {epoch:03d} val_mse={val_loss:.5g}")
    if best_state:
        model.load_state_dict(best_state)
    return loaders


def predict(model, loader, device):
    y_true, y_pred = [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            y_true.append(batch.y.view(-1).detach().cpu().numpy())
            y_pred.append(model(batch).detach().cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    frame = read_table(args.subset)
    y = pd.to_numeric(frame[args.target_col], errors="coerce").to_numpy(dtype=np.float32)
    smiles = smiles_series(frame)
    graphs, keep = build_graphs(smiles, y)
    frame = frame.iloc[keep].reset_index(drop=True)
    print(f"Graphs used: {len(graphs):,}")

    test_frame = None
    test_graphs = None
    if args.test_subset:
        raw_test = read_table(args.test_subset)
        test_y = pd.to_numeric(raw_test[args.target_col], errors="coerce").to_numpy(dtype=np.float32)
        test_smiles = smiles_series(raw_test)
        test_graphs, test_keep = build_graphs(test_smiles, test_y)
        test_frame = raw_test.iloc[test_keep].reset_index(drop=True)
        print(f"Untouched test graphs: {len(test_graphs):,}")

    rows = []
    for key in args.methods:
        if key not in MODELS:
            raise ValueError(f"Unknown GNN method: {key}. Valid methods: {sorted(MODELS)}")
        name, cls = MODELS[key]
        print(f"\n=== {name} ===")
        out_dir = method_dir(args.output_dir, name)
        metrics, cv_pred_rows = {}, []
        for fold, train_idx, val_idx in cv_fold_indices(frame):
            print(f"Fold {fold}")
            model = cls().to(device)
            loaders = train_one(model, graphs, {"train": train_idx, "val": val_idx}, args, device)
            yt, yp = predict(model, loaders["val"], device)
            split = f"fold_{fold}"
            metrics[split] = regression_metrics(yt, yp)
            part = frame.iloc[val_idx][[c for c in ["record_id", "study_id", "lnp_id", "il_id", "il_smiles"] if c in frame.columns]].copy()
            part["split"] = split
            part["cv_fold"] = fold
            part["y_true"] = yt
            part["y_pred"] = yp
            cv_pred_rows.append(part)

        final_model = cls().to(device)
        all_idx = np.arange(len(graphs))
        train_one(final_model, graphs, {"train": all_idx, "val": all_idx}, args, device)
        if test_graphs:
            test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)
            yt, yp = predict(final_model, test_loader, device)
            metrics["untouched_test"] = regression_metrics(yt, yp)
            part = test_frame[[c for c in ["record_id", "study_id", "lnp_id", "il_id", "il_smiles"] if c in test_frame.columns]].copy()
            part["split"] = "untouched_test"
            part["y_true"] = yt
            part["y_pred"] = yp
            part.to_csv(out_dir / "untouched_test_predictions.csv", index=False)

        fold_df = pd.DataFrame([v for k, v in metrics.items() if k.startswith("fold_")])
        metric_rows = [{"split": k, **v} for k, v in metrics.items()]
        metric_rows.append({"split": "cv_mean", **fold_df.mean(numeric_only=True).to_dict()})
        metric_rows.append({"split": "cv_std", **fold_df.std(numeric_only=True).to_dict()})
        torch.save({"state_dict": final_model.state_dict(), "model_class": cls.__name__}, out_dir / "model.pt")
        pd.concat(cv_pred_rows, ignore_index=True).to_csv(out_dir / "cv_predictions.csv", index=False)
        pd.DataFrame(metric_rows).to_csv(out_dir / "metrics.csv", index=False)
        write_json(out_dir / "metrics.json", metrics)
        write_json(out_dir / "metadata.json", {
            "method_name": name,
            "target_col": args.target_col,
            "n_trainval_rows": len(graphs),
            "n_test_rows": len(test_graphs) if test_graphs else 0,
            "cv": "5-fold cross-validation on trainval split",
            "untouched_test_used_for_training": False,
            "trained_at_utc": now_utc(),
            "model_file": str(out_dir / "model.pt"),
            "metadata": {"input_subset": str(args.subset), "graph_featurization": "rdkit_atom_bond_graph"},
        })
        row = {"method": name, "output_dir": str(out_dir)}
        cv_mean = next(x for x in metric_rows if x["split"] == "cv_mean")
        row.update({f"cv_mean_{k}": v for k, v in cv_mean.items() if k != "split"})
        if "untouched_test" in metrics:
            row.update({f"untouched_test_{k}": v for k, v in metrics["untouched_test"].items()})
        rows.append(row)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(Path(args.output_dir) / "gnn_training_manifest.csv", index=False)
    print("\nSaved GNN manifest:")
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
