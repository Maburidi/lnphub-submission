from __future__ import annotations

import argparse
import copy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from .features import smiles_series, valid_smiles_mask
from .graph_data import build_graphs
from .io_utils import ensure_dir, now_utc, read_table, write_json
from .train_frozen_classification import (
    CLASS_LABELS,
    CLASS_NAMES,
    class_names_for,
    classification_metrics,
    format_metrics,
    id_columns,
    infer_class_labels,
    safe_name,
)
from .torch_models import DMPNNClassifier, GCNClassifier, GINClassifier


MODELS = {
    "gcn": ("GCN", GCNClassifier),
    "gin": ("GIN", GINClassifier),
    "dmpnn": ("D-MPNN", DMPNNClassifier),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 3-class LNPHub graph neural network classifiers.")
    parser.add_argument("--trainval-subset", type=Path)
    parser.add_argument("--train-subset", type=Path)
    parser.add_argument("--validation-subset", type=Path)
    parser.add_argument("--test-subset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-col", default="delivery_class_3")
    parser.add_argument("--methods", nargs="+", default=list(MODELS))
    parser.add_argument("--tabular-cols", nargs="*", default=[])
    parser.add_argument("--cv-description", default="random stratified 5-fold CV on balanced trainval data")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
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


def usable_frame(path: Path, target_col: str, split_name: str) -> pd.DataFrame:
    frame = read_table(path).copy()
    if target_col not in frame.columns:
        raise ValueError(f"{split_name} is missing target column: {target_col}")
    y = pd.to_numeric(frame[target_col], errors="coerce")
    smiles = smiles_series(frame)
    ok = y.notna() & valid_smiles_mask(smiles)
    dropped = int((~ok).sum())
    if dropped:
        print(f"{split_name}: dropping {dropped:,} rows with missing class or invalid SMILES", flush=True)
    frame = frame.loc[ok].reset_index(drop=True)
    frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce").astype(int)
    print(f"{split_name}: {len(frame):,} usable rows", flush=True)
    print(f"{split_name} class counts: {dict(frame[target_col].value_counts().sort_index())}", flush=True)
    return frame


def fit_tabular_preprocessor(frame: pd.DataFrame, tabular_cols: list[str]) -> Pipeline | None:
    if not tabular_cols:
        return None
    missing = [c for c in tabular_cols if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing tabular column(s): {missing}")
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    values = frame[tabular_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    preprocessor.fit(values)
    return preprocessor


def transform_tabular(frame: pd.DataFrame, tabular_cols: list[str], preprocessor: Pipeline | None) -> np.ndarray | None:
    if not tabular_cols:
        return None
    if preprocessor is None:
        raise ValueError("Tabular columns were provided but no preprocessor was fitted.")
    values = frame[tabular_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    return preprocessor.transform(values).astype(np.float32)


def graph_dataset(frame: pd.DataFrame, target_col: str, tabular_values: np.ndarray | None = None):
    y = frame[target_col].to_numpy(dtype=np.float32)
    graphs, keep = build_graphs(smiles_series(frame), y)
    frame = frame.iloc[keep].reset_index(drop=True)
    y_int = frame[target_col].to_numpy(dtype=int)
    if tabular_values is not None:
        tabular_values = tabular_values[keep]
    for i, graph in enumerate(graphs):
        graph.y = torch.tensor([int(y_int[i])], dtype=torch.long)
        graph.sample_id = torch.tensor([i], dtype=torch.long)
        if tabular_values is not None:
            graph.tabular = torch.tensor(tabular_values[i].reshape(1, -1), dtype=torch.float32)
    return graphs, frame


def make_loader(graphs, indices=None, *, batch_size: int = 64, shuffle: bool = False) -> DataLoader:
    if indices is None:
        dataset = graphs
    else:
        dataset = Subset(graphs, np.asarray(indices, dtype=int).tolist())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_model(model, train_graphs, val_graphs, args, device, label: str):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_loader = make_loader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(val_graphs, batch_size=args.batch_size, shuffle=False)

    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, batch.y.view(-1).long())
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.inference_mode():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                val_losses.append(float(loss_fn(logits, batch.y.view(-1).long()).detach().cpu()))
        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"{label} | epoch {epoch:03d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, best_val={best_val:.4f}", flush=True)
        if bad_epochs >= args.patience:
            print(f"{label}: early stopping at epoch {epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model, graphs, args, device):
    loader = make_loader(graphs, batch_size=args.batch_size, shuffle=False)
    y_true, y_pred, y_proba, row_idx = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            proba = torch.softmax(logits, dim=1)
            y_true.append(batch.y.view(-1).detach().cpu().numpy())
            y_pred.append(torch.argmax(proba, dim=1).detach().cpu().numpy())
            y_proba.append(proba.detach().cpu().numpy())
            row_idx.append(batch.sample_id.view(-1).detach().cpu().numpy())
    return (
        np.concatenate(y_true).astype(int),
        np.concatenate(y_pred).astype(int),
        np.concatenate(y_proba).astype(float),
        np.concatenate(row_idx).astype(int),
    )


def save_eval_tables(out_dir: Path, frame: pd.DataFrame, y_true, y_pred, proba, row_idx, split_name: str) -> dict[str, float]:
    proba = np.asarray(proba, dtype=float)
    row_sums = proba.sum(axis=1, keepdims=True)
    proba = np.divide(proba, row_sums, out=np.full_like(proba, 1.0 / proba.shape[1]), where=row_sums > 0)
    class_labels = infer_class_labels(y_true, y_pred, proba)
    class_names = class_names_for(class_labels)
    metrics = classification_metrics(y_true, y_pred, proba, class_labels=class_labels)
    metrics.update({"split": split_name})

    pred_df = frame.iloc[row_idx][id_columns(frame)].reset_index(drop=True).copy()
    pred_df["split"] = split_name
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = y_pred
    for i, cls in enumerate(class_labels):
        pred_df[f"prob_{class_names[cls]}"] = proba[:, i]
    pred_df.to_csv(out_dir / f"{split_name}_predictions.csv", index=False)

    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    pd.DataFrame(
        cm,
        index=[f"true_{class_names[c]}" for c in class_labels],
        columns=[f"pred_{class_names[c]}" for c in class_labels],
    ).to_csv(out_dir / f"{split_name}_confusion_matrix.csv")
    per_class = precision_recall_fscore_support(y_true, y_pred, labels=class_labels, zero_division=0)
    pd.DataFrame({
        "class": [class_names[c] for c in class_labels],
        "precision": per_class[0],
        "recall": per_class[1],
        "f1": per_class[2],
        "support": per_class[3],
    }).to_csv(out_dir / f"{split_name}_per_class_metrics.csv", index=False)
    return metrics


def split_graphs(graphs, frame: pd.DataFrame, fold: int | None = None):
    if fold is None:
        return graphs, frame
    fold_values = frame["cv_fold"].astype(int).to_numpy()
    val_idx = np.where(fold_values == fold)[0]
    train_idx = np.where(fold_values != fold)[0]
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    return train_graphs, val_graphs, train_idx, val_idx


def summarize_cv(metric_rows: list[dict]) -> tuple[dict, dict]:
    fold_df = pd.DataFrame([r for r in metric_rows if str(r.get("split", "")).startswith("fold_")])
    cv_mean = fold_df.drop(columns=["method", "split", "fold", "validation_study"], errors="ignore").mean(numeric_only=True).to_dict()
    cv_mean.update({"split": "cv_mean", "fold": -1})
    cv_std = fold_df.drop(columns=["method", "split", "fold", "validation_study"], errors="ignore").std(numeric_only=True).to_dict()
    cv_std.update({"split": "cv_std", "fold": -1})
    return cv_mean, cv_std


def run_random_split(args, device, tabular_dim: int):
    train = usable_frame(args.train_subset, args.target_col, "train")
    validation = usable_frame(args.validation_subset, args.target_col, "validation")
    test = usable_frame(args.test_subset, args.target_col, "test")
    class_labels = infer_class_labels(train[args.target_col], validation[args.target_col], test[args.target_col])
    class_names = class_names_for(class_labels)
    num_classes = len(class_labels)
    print("Class labels:", class_labels, flush=True)

    preprocessor = fit_tabular_preprocessor(train, args.tabular_cols)
    train_graphs, train = graph_dataset(train, args.target_col, transform_tabular(train, args.tabular_cols, preprocessor))
    val_graphs, validation = graph_dataset(validation, args.target_col, transform_tabular(validation, args.tabular_cols, preprocessor))
    test_graphs, test = graph_dataset(test, args.target_col, transform_tabular(test, args.tabular_cols, preprocessor))
    print(f"Graphs: train={len(train_graphs):,}; validation={len(val_graphs):,}; test={len(test_graphs):,}", flush=True)

    rows = []
    for key in args.methods:
        if key not in MODELS:
            raise ValueError(f"Unknown method: {key}. Valid methods: {sorted(MODELS)}")
        method_name, model_cls = MODELS[key]
        random_method_name = f"{method_name} random 80/10/10"
        print("\n" + "=" * 80, flush=True)
        print(random_method_name, flush=True)
        print("=" * 80, flush=True)
        out_dir = ensure_dir(args.output_dir / safe_name(random_method_name))

        set_seed(args.seed)
        model = model_cls(tabular_dim=tabular_dim, num_classes=num_classes).to(device)
        model = train_one_model(model, train_graphs, val_graphs, args, device, f"{method_name} | random_split")

        metric_rows = []
        y_true, y_pred, proba, row_idx = predict(model, val_graphs, args, device)
        val_metrics = save_eval_tables(out_dir, validation, y_true, y_pred, proba, row_idx, "validation")
        val_metrics.update({"method": random_method_name})
        metric_rows.append(val_metrics)
        print(f"{random_method_name} | validation: {format_metrics(val_metrics)}", flush=True)

        y_true, y_pred, proba, row_idx = predict(model, test_graphs, args, device)
        test_metrics = save_eval_tables(out_dir, test, y_true, y_pred, proba, row_idx, "test")
        test_metrics.update({"method": random_method_name})
        metric_rows.append(test_metrics)
        print(f"{random_method_name} | test: {format_metrics(test_metrics)}", flush=True)

        torch.save({
            "state_dict": model.state_dict(),
            "model_class": model_cls.__name__,
            "tabular_dim": tabular_dim,
            "num_classes": num_classes,
            "class_labels": class_labels,
            "class_names": class_names,
        }, out_dir / "model.pt")
        if preprocessor is not None:
            joblib.dump(preprocessor, out_dir / "tabular_preprocessor.joblib")
        pd.DataFrame(metric_rows).to_csv(out_dir / "metrics.csv", index=False)
        write_json(out_dir / "metadata.json", metadata_payload(args, random_method_name, tabular_dim, "random 80/10/10", out_dir, class_labels))
        write_json(out_dir / "feature_schema.json", feature_schema(args, tabular_dim, class_labels))
        print(f"{random_method_name}: saved outputs -> {out_dir}", flush=True)
        rows.append(manifest_row(random_method_name, out_dir, val_metrics, test_metrics))

    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.output_dir / "gnn_random_split_classification_manifest.csv", index=False)
    print("\nSaved random-split GNN manifest:", args.output_dir / "gnn_random_split_classification_manifest.csv", flush=True)
    print(manifest.to_string(index=False), flush=True)


def run_cv(args, device, tabular_dim: int):
    trainval_raw = usable_frame(args.trainval_subset, args.target_col, "trainval")
    test_raw = usable_frame(args.test_subset, args.target_col, "test")
    class_labels = infer_class_labels(trainval_raw[args.target_col], test_raw[args.target_col])
    class_names = class_names_for(class_labels)
    num_classes = len(class_labels)
    print("Class labels:", class_labels, flush=True)
    if "cv_fold" not in trainval_raw.columns:
        raise ValueError("trainval subset is missing cv_fold.")

    folds = sorted(int(f) for f in trainval_raw["cv_fold"].dropna().unique() if int(f) >= 0)
    if not folds:
        raise ValueError("No usable CV folds found.")

    rows = []
    for key in args.methods:
        if key not in MODELS:
            raise ValueError(f"Unknown method: {key}. Valid methods: {sorted(MODELS)}")
        method_name, model_cls = MODELS[key]
        print("\n" + "=" * 80, flush=True)
        print(method_name, flush=True)
        print("=" * 80, flush=True)
        out_dir = ensure_dir(args.output_dir / safe_name(method_name))

        metric_rows = []
        cv_predictions = []
        for fold in folds:
            val_mask = trainval_raw["cv_fold"].astype(int).to_numpy() == fold
            train_frame = trainval_raw.loc[~val_mask].reset_index(drop=True)
            val_frame = trainval_raw.loc[val_mask].reset_index(drop=True)
            validation_studies = ", ".join(sorted(val_frame["study_id"].astype(str).unique()))
            print(f"\n{method_name} | fold {fold}: train={len(train_frame):,}, val={len(val_frame):,}; validation studies={validation_studies}", flush=True)

            preprocessor = fit_tabular_preprocessor(train_frame, args.tabular_cols)
            train_graphs, train_frame = graph_dataset(train_frame, args.target_col, transform_tabular(train_frame, args.tabular_cols, preprocessor))
            val_graphs, val_frame = graph_dataset(val_frame, args.target_col, transform_tabular(val_frame, args.tabular_cols, preprocessor))

            set_seed(args.seed + fold)
            model = model_cls(tabular_dim=tabular_dim, num_classes=num_classes).to(device)
            model = train_one_model(model, train_graphs, val_graphs, args, device, f"{method_name} | fold_{fold}")
            y_true, y_pred, proba, row_idx = predict(model, val_graphs, args, device)
            metrics = classification_metrics(y_true, y_pred, proba, class_labels=class_labels)
            metrics.update({"method": method_name, "split": f"fold_{fold}", "fold": fold, "validation_study": validation_studies})
            metric_rows.append(metrics)
            print(f"{method_name} | fold_{fold}: {format_metrics(metrics)}", flush=True)

            pred_df = val_frame.iloc[row_idx][id_columns(val_frame)].reset_index(drop=True).copy()
            pred_df["split"] = f"fold_{fold}"
            pred_df["cv_fold"] = fold
            pred_df["y_true"] = y_true
            pred_df["y_pred"] = y_pred
            for i, cls in enumerate(class_labels):
                pred_df[f"prob_{class_names[cls]}"] = proba[:, i]
            cv_predictions.append(pred_df)

        cv_mean, cv_std = summarize_cv(metric_rows)
        cv_mean.update({"method": method_name, "validation_study": args.cv_description})
        cv_std.update({"method": method_name, "validation_study": args.cv_description})
        metric_rows.extend([cv_mean, cv_std])
        print(f"\n{method_name} | cv_mean: {format_metrics(cv_mean)}", flush=True)

        final_preprocessor = fit_tabular_preprocessor(trainval_raw, args.tabular_cols)
        trainval_graphs, trainval = graph_dataset(trainval_raw, args.target_col, transform_tabular(trainval_raw, args.tabular_cols, final_preprocessor))
        test_graphs, test = graph_dataset(test_raw, args.target_col, transform_tabular(test_raw, args.tabular_cols, final_preprocessor))
        print(f"{method_name}: fitting final model on all trainval graphs...", flush=True)
        set_seed(args.seed)
        final_model = model_cls(tabular_dim=tabular_dim, num_classes=num_classes).to(device)
        final_model = train_one_model(final_model, trainval_graphs, trainval_graphs, args, device, f"{method_name} | final_trainval")

        y_true, y_pred, proba, row_idx = predict(final_model, test_graphs, args, device)
        test_metrics = save_eval_tables(out_dir, test, y_true, y_pred, proba, row_idx, "test")
        test_metrics.update({"method": method_name, "split": "test", "fold": -1, "validation_study": ", ".join(sorted(test["study_id"].astype(str).unique()))})
        metric_rows.append(test_metrics)
        print(f"{method_name} | test: {format_metrics(test_metrics)}", flush=True)

        torch.save({
            "state_dict": final_model.state_dict(),
            "model_class": model_cls.__name__,
            "tabular_dim": tabular_dim,
            "num_classes": num_classes,
            "class_labels": class_labels,
            "class_names": class_names,
        }, out_dir / "model.pt")
        if final_preprocessor is not None:
            joblib.dump(final_preprocessor, out_dir / "tabular_preprocessor.joblib")
        pd.concat(cv_predictions, ignore_index=True).to_csv(out_dir / "cv_predictions.csv", index=False)
        pd.DataFrame(metric_rows).to_csv(out_dir / "metrics.csv", index=False)
        write_json(out_dir / "metadata.json", metadata_payload(args, method_name, tabular_dim, args.cv_description, out_dir, class_labels))
        write_json(out_dir / "feature_schema.json", feature_schema(args, tabular_dim, class_labels))
        print(f"{method_name}: saved outputs -> {out_dir}", flush=True)
        rows.append(manifest_row(method_name, out_dir, cv_mean, test_metrics, cv_prefix="cv", test_prefix="test"))

    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.output_dir / "gnn_cv_classification_manifest.csv", index=False)
    print("\nSaved CV GNN manifest:", args.output_dir / "gnn_cv_classification_manifest.csv", flush=True)
    print(manifest.to_string(index=False), flush=True)


def feature_schema(args, tabular_dim: int, class_labels: list[int] | None = None) -> dict:
    class_labels = class_labels or CLASS_LABELS
    class_names = class_names_for(class_labels)
    return {
        "graph_featurization": "rdkit atom/bond graph from il_smiles/canonical_il_smiles",
        "tabular_columns": args.tabular_cols,
        "tabular_dim": tabular_dim,
        "n_classes": len(class_labels),
        "class_labels": class_labels,
        "class_names": class_names,
    }


def metadata_payload(args, method_name: str, tabular_dim: int, cv_description: str, out_dir: Path, class_labels: list[int] | None = None) -> dict:
    class_labels = class_labels or CLASS_LABELS
    class_names = class_names_for(class_labels)
    return {
        "method": method_name,
        "feature_set": "graph_only" if tabular_dim == 0 else "graph_plus_tabular_context",
        "target": args.target_col,
        "class_labels": class_labels,
        "class_names": class_names,
        "trainval_path": str(args.trainval_subset) if args.trainval_subset else None,
        "train_path": str(args.train_subset) if args.train_subset else None,
        "validation_path": str(args.validation_subset) if args.validation_subset else None,
        "test_path": str(args.test_subset),
        "cv": cv_description,
        "tabular_columns": args.tabular_cols,
        "test_used_for_training": False,
        "trained_at_utc": now_utc(),
        "output_dir": str(out_dir),
    }


def manifest_row(method_name: str, out_dir: Path, primary_metrics: dict, test_metrics: dict, cv_prefix: str = "validation", test_prefix: str = "test") -> dict:
    row = {
        "method": method_name,
        "output_dir": str(out_dir),
    }
    for metric in ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc_ovr_macro", "average_precision_macro", "mcc", "log_loss"]:
        row[f"{cv_prefix}_{metric}"] = primary_metrics.get(metric)
        row[f"{test_prefix}_{metric}"] = test_metrics.get(metric)
    return row


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tabular_dim = len(args.tabular_cols)
    print("Device:", device, flush=True)
    print("Methods:", ", ".join(args.methods), flush=True)
    print("Tabular/context columns:", tabular_dim, flush=True)
    if args.tabular_cols:
        print(", ".join(args.tabular_cols), flush=True)

    if args.trainval_subset is not None:
        run_cv(args, device, tabular_dim)
    else:
        run_random_split(args, device, tabular_dim)


if __name__ == "__main__":
    main()
