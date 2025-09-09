# -*- coding: utf-8 -*-
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from tueplots import bundles


plt.rcParams.update(bundles.icml2022())


def ensure_dir(path: Path) -> Path:
    """Create directory if it does not exist and return it as Path."""
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path


def list_subfolders(directory) -> list[Path]:
    """Return all subfolders (recursively) under 'directory'."""
    directory = Path(directory)
    return [p for p in directory.rglob("*") if p.is_dir()]


def try_parse_float(value, default=float("nan")) -> float:
    """Safely parse a value to float, returning default on failure."""
    try:
        return float(value)
    except Exception:
        return default


def read_last_row_for_split(csv_path: Path, split: str) -> pd.Series | None:
    """
    Read a CSV and return the last row for the given split.
    Returns None if file is missing, columns are missing, or the split is absent.
    """
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    if "split" not in df.columns:
        return None

    df_split = df[df["split"] == split]
    if df_split.empty:
        return None

    return df_split.iloc[-1]


def extract_metrics(row: pd.Series) -> dict:
    """Extract required metrics from a pandas Series into plain floats."""
    def get(col, default=float("nan")):
        if col in row.index:
            return try_parse_float(row[col], default=default)
        return default

    return {
        "loss": get("loss"),
        "accuracy": get("accuracy"),
        "precision": get("precision"),
        "recall": get("recall"),
        "f1": get("f1"),
        "tp": get("tp", 0.0),
        "fp": get("fp", 0.0),
        "fn": get("fn", 0.0),
        "tn": get("tn", 0.0),
    }


def aggregate_metrics(dir_path: Path, split: str) -> dict:
    """
    Walk subfolders, read training_log.csv, and aggregate metrics for the given split.
    Returns a dict of lists plus the layer names.
    """
    losses, accs, precs, recalls, f1s = [], [], [], [], []
    tps, fps, fns, tns = [], [], [], []
    layer_names = []

    for sub in sorted(list_subfolders(dir_path)):
        csv_path = sub / "training_log.csv"
        row = read_last_row_for_split(csv_path, split)
        if row is None:
            continue

        m = extract_metrics(row)

        losses.append(m["loss"])
        accs.append(m["accuracy"])
        precs.append(m["precision"])
        recalls.append(m["recall"])
        f1s.append(m["f1"])

        tps.append(m["tp"])
        fps.append(m["fp"])
        fns.append(m["fn"])
        tns.append(m["tn"])

        layer_names.append(sub.name)

    return {
        "losses": losses,
        "accs": accs,
        "precs": precs,
        "recalls": recalls,
        "f1s": f1s,
        "tps": tps,
        "fps": fps,
        "fns": fns,
        "tns": tns,
        "layer_names": layer_names,
    }


def plot_loss_per_layer(x, xticks_labels, losses, save_path: Path) -> None:
    """Plot loss per layer and save as PNG."""
    plt.figure(figsize=(10, 4))
    plt.plot(x, losses, marker="o")
    plt.xticks(x, xticks_labels, rotation=45, ha="right")
    plt.xlabel("Layer")
    plt.ylabel("Loss")
    plt.title("Loss per Layer")
    plt.tight_layout()
    plt.savefig(save_path / "loss_per_layer.png")
    plt.close()


def plot_classification_metrics(x, xticks_labels, accs, precs, recalls, f1s, save_path: Path) -> None:
    """Plot classification metrics per layer and save as PNG."""
    plt.figure(figsize=(10, 4))
    plt.plot(x, accs, marker="o", label="Accuracy")
    plt.plot(x, precs, marker="o", label="Precision")
    plt.plot(x, recalls, marker="o", label="Recall")
    plt.plot(x, f1s, marker="o", label="F1")
    plt.xticks(x, xticks_labels, rotation=45, ha="right")
    plt.xlabel("Layer")
    plt.ylabel("Score")
    plt.title("Classification Metrics per Layer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / "classification_metrics_per_layer.png")
    plt.close()


def plot_confusion_counts(x, xticks_labels, tps, fps, fns, tns, save_path: Path) -> None:
    """Plot grouped bar chart for TP/FP/FN/TN per layer and save as PNG."""
    width = 0.2
    plt.figure(figsize=(12, 5))

    x_tp = [i - 1.5 * width for i in x]
    x_fp = [i - 0.5 * width for i in x]
    x_fn = [i + 0.5 * width for i in x]
    x_tn = [i + 1.5 * width for i in x]

    plt.bar(x_tp, tps, width=width, label="TP")
    plt.bar(x_fp, fps, width=width, label="FP")
    plt.bar(x_fn, fns, width=width, label="FN")
    plt.bar(x_tn, tns, width=width, label="TN")

    plt.xticks(x, xticks_labels, rotation=45, ha="right")
    plt.xlabel("Layer")
    plt.ylabel("Count")
    plt.title("Confusion Counts per Layer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / "confusion_counts_per_layer.png")
    plt.close()


# ----------------------------
# NEW: Cross-model accuracy histogram (per layer)
# ----------------------------
def aggregate_model_accuracies(
    base_dir: Path,
    model_dirs: list[str],
    model_names: list[str],
    split: str
) -> tuple[list[str], dict[str, list[float]]]:
    """
    Aggregate per-layer accuracies for multiple models.
    Returns (layer_labels, {model_name: accuracies_per_layer}).
    Cuts to the minimum common number of layers to keep groups aligned.
    """
    model_to_accs: dict[str, list[float]] = {}
    layer_counts = []

    for m_dir, m_name in zip(model_dirs, model_names):
        data = aggregate_metrics(base_dir / m_dir, split)
        if not data["accs"]:
            # Skip models with no data
            continue
        model_to_accs[m_name] = data["accs"]
        layer_counts.append(len(data["accs"]))

    if not model_to_accs:
        return [], {}

    # Align by minimum number of layers found across models
    min_layers = min(layer_counts)
    aligned = {name: accs[:min_layers] for name, accs in model_to_accs.items()}
    layer_labels = [str(i) for i in range(min_layers)]
    return layer_labels, aligned


def plot_accuracy_histogram_per_layer(
    layer_labels: list[str],
    model_to_accs: dict[str, list[float]],
    save_path: Path,
    filename: str = "accuracy_histogram_per_layer.png",
    title: str = "Accuracy per Layer across Models"
) -> None:
    """Create a grouped bar chart showing accuracy per layer for each model."""
    if not layer_labels or not model_to_accs:
        print("No data for cross-model histogram.")
        return

    ensure_dir(save_path)

    num_layers = len(layer_labels)
    num_models = len(model_to_accs)

    x = list(range(num_layers))
    total_width = 0.8  # total width allocated to each layer's group
    bar_width = total_width / max(1, num_models)

    plt.figure(figsize=(max(10, num_layers * 0.7), 5))

    # Center the group at each x and offset each model within the group
    offsets = [
        (-total_width / 2) + (i + 0.5) * bar_width
        for i in range(num_models)
    ]

    for offset, (model_name, accs) in zip(offsets, model_to_accs.items()):
        positions = [xi + offset for xi in x]
        plt.bar(positions, accs[:num_layers], width=bar_width, label=model_name)

    plt.xticks(x, layer_labels, rotation=45, ha="right")
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / filename)
    plt.close()


# ----------------------------
# Orchestration
# ----------------------------
def create_plots(dir_path, save_path, split: str = "val_epoch") -> None:
    """
    Read per-subfolder training_log.csv files, pick the last row for the given split,
    and plot:
      1) Loss per layer
      2) Accuracy / Precision / Recall / F1 per layer
      3) Confusion counts (TP / FP / FN / TN) per layer
    """
    save_path = ensure_dir(Path(save_path))
    dir_path = Path(dir_path)

    data = aggregate_metrics(dir_path, split)
    if not data["losses"]:
        print("No data found. Ensure subfolders contain training_log.csv with the requested split.")
        return

    x = list(range(len(data["layer_names"])))
    xticks_labels = [str(i) for i in x]

    plot_loss_per_layer(x, xticks_labels, data["losses"], save_path)
    plot_classification_metrics(
        x, xticks_labels, data["accs"], data["precs"], data["recalls"], data["f1s"], save_path
    )
    plot_confusion_counts(
        x, xticks_labels, data["tps"], data["fps"], data["fns"], data["tns"], save_path
    )


if __name__ == "__main__":
    # Directory names (relative to artifacts/) that hold per-layer subfolders
    models = [
        "exp1/",
        "exp2/",
        "exp3/",
        # "test/",
    ]
    # Display names used in legends
    model_names = ["Gemma-3-4B", "Qwen2-VL-2B", "FastVLM-0.5B"]

    # Per-model plots (unchanged)
    for model_dir in models:
        create_plots(f"artifacts/{model_dir}", f"report/figures/{model_dir}")

    # NEW: Cross-model accuracy histogram per layer
    base_dir = Path("artifacts")
    combined_save = ensure_dir(Path("report/figures/_combined"))
    layer_labels, model_to_accs = aggregate_model_accuracies(
        base_dir=base_dir,
        model_dirs=models,
        model_names=model_names,
        split="val_epoch",
    )
    plot_accuracy_histogram_per_layer(
        layer_labels=layer_labels,
        model_to_accs=model_to_accs,
        save_path=combined_save,
        filename="accuracy_histogram_per_layer.png",
        title="Accuracy per Layer across Models",
    )
