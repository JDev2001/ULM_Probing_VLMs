# -*- coding: utf-8 -*-
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from tueplots import bundles
import numpy as np

plt.rcParams.update(bundles.icml2022())
plt.rcParams["savefig.dpi"] = 500


PLOT_SIZE = (5, 2)            # Default size for line plots
WIDE_PLOT_SIZE = (7, 4)       # Default size for grouped bar plots
COMBINED_MIN_WIDTH = 3.5      # Min width for cross-model plot
COMBINED_WIDTH_PER_LAYER = 0.17
COMBINED_HEIGHT = 2.3


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
    plt.figure(figsize=PLOT_SIZE)
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
    plt.figure(figsize=PLOT_SIZE)
    plt.plot(x, accs, marker="o", label="Accuracy")
    plt.plot(x, precs, marker="o", label="Precision")
    plt.plot(x, recalls, marker="o", label="Recall")
    plt.plot(x, f1s, marker="o", label="F1")
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Baseline (0.5)")
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
    plt.figure(figsize=WIDE_PLOT_SIZE)

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
            continue
        model_to_accs[m_name] = data["accs"]
        layer_counts.append(len(data["accs"]))

    if not model_to_accs:
        return [], {}

    min_layers = min(layer_counts)
    aligned = {name: accs[:min_layers] for name, accs in model_to_accs.items()}
    layer_labels = [str(i) for i in range(min_layers)]
    return layer_labels, aligned



def aggregate_model_metric(
    base_dir: Path,
    model_dirs: list[str],
    model_names: list[str],
    split: str,
    metric_key: str,
) -> tuple[list[str], dict[str, list[float]]]:
    """
    Aggregate per-layer values for a given metric across multiple models.
    metric_key in {"accuracy", "precision", "recall", "f1"}.
    Returns (layer_labels, {model_name: values_per_layer}), aligned to min common depth.
    """
    keymap = {
        "accuracy": "accs",
        "precision": "precs",
        "recall": "recalls",
        "f1": "f1s",
    }
    if metric_key not in keymap:
        raise ValueError(f"Unsupported metric_key: {metric_key}")

    model_to_vals: dict[str, list[float]] = {}
    layer_counts = []

    for m_dir, m_name in zip(model_dirs, model_names):
        data = aggregate_metrics(base_dir / m_dir, split)
        vals = data[keymap[metric_key]]
        if not vals:
            continue
        model_to_vals[m_name] = vals
        layer_counts.append(len(vals))

    if not model_to_vals:
        return [], {}

    min_layers = min(layer_counts)
    aligned = {name: vals[:min_layers] for name, vals in model_to_vals.items()}
    layer_labels = [str(i) for i in range(min_layers)]
    return layer_labels, aligned
def plot_combined_metric_heatmap(
    layer_labels: list[str],
    model_to_f1s: dict[str, list[float]],
    model_to_precs: dict[str, list[float]],
    model_to_recalls: dict[str, list[float]],
    save_path: Path,
    filename: str,
    title: str,
) -> None:
    """
    Plot a combined heatmap: F1-score for color, Precision/Recall for annotation.
    MODIFIED: Axes swapped and cell height reduced for a more compact plot.
    """
    if not layer_labels or not model_to_f1s:
        print(f"No data for combined heatmap: {title}")
        return

    ensure_dir(save_path)

    model_names = list(model_to_f1s.keys())
    num_models = len(model_names)
    num_layers = len(layer_labels)

    # Build matrix for F1-score (for colors) - TRANSPOSED
    f1_mat = np.array([model_to_f1s[m][:num_layers] for m in model_names], dtype=float).T

    # Figure size - Adjusted for new orientation and reduced height
    fig_width = max(COMBINED_MIN_WIDTH, num_models * 1.5)
    fig_height = max(2.0, num_layers * 0.15)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), layout="constrained")

    im = ax.imshow(f1_mat, aspect="auto", vmin=0.0, vmax=1.0, cmap="YlOrRd")

    ax.set_xticks(np.arange(num_models), labels=model_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(num_layers), labels=layer_labels)
    ax.set_xlabel("Model")
    ax.set_ylabel("Layer")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("F1-Score", rotation=270, labelpad=15)

    # Custom annotation with Precision and Recall in a single line
    for i in range(num_layers):
        for j in range(num_models):
            model_name = model_names[j]
            prec = model_to_precs.get(model_name, [])[i]
            recall = model_to_recalls.get(model_name, [])[i]

            if np.isfinite(prec) and np.isfinite(recall):
                # MODIFIED: Text is now in a single line, font size slightly adjusted
                text_label = f"P:{prec:.2f} R:{recall:.2f}"
                ax.text(j, i, text_label, ha="center", va="center", fontsize=7, color='white',  bbox=dict(facecolor="black", alpha=0.2, boxstyle="square,pad=0.15"))

    fig.savefig(save_path / filename)
    plt.close(fig)


def plot_accuracy_lines_per_layer(
    layer_labels: list[str],
    model_to_accs: dict[str, list[float]],
    save_path: Path,
    filename: str = "accuracy_lines_per_layer.png",
    title: str = "Accuracy per Layer across Models"
) -> None:
    """
    Create a line plot showing accuracy per layer for each model.
    Each model is represented as a separate line.
    Y-axis is fixed to [0, 1].
    """
    if not layer_labels or not model_to_accs:
        print("No data for cross-model line plot.")
        return

    ensure_dir(save_path)

    num_layers = len(layer_labels)
    x = list(range(num_layers))

    fig_width = max(COMBINED_MIN_WIDTH, num_layers * COMBINED_WIDTH_PER_LAYER)
    plt.figure(figsize=(fig_width, COMBINED_HEIGHT))

    for model_name, accs in model_to_accs.items():
        plt.plot(x, accs[:num_layers], marker="o", label=model_name)

    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Baseline (0.5)")

    plt.xticks(x, layer_labels, rotation=45, ha="right")
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / filename)
    plt.close()


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
    models = [
        "exp1_1/",
        "exp1_3/",
    ]
    model_names = ["Qwen2-VL-2B", "FastVLM-0.5B"]

    for model_dir in models:
        ensure_dir(Path(f"report/figures/global/{model_dir}"))
        create_plots(f"artifacts/{model_dir}", f"report/figures/global/{model_dir}")

    base_dir = Path("artifacts")
    combined_save = ensure_dir(Path("report/figures/global/_combined_exp1"))

    # Combined line plot for accuracy
    layer_labels, model_to_accs = aggregate_model_accuracies(
        base_dir=base_dir,
        model_dirs=models,
        model_names=model_names,
        split="val_epoch",
    )
    plot_accuracy_lines_per_layer(
        layer_labels=layer_labels,
        model_to_accs=model_to_accs,
        save_path=combined_save,
        filename="accuracy_lines_per_layer.png",
        title="Accuracy per Layer across Models",
    )

    # Combined Heatmap for F1, Precision, and Recall
    print("Generating combined metrics heatmap...")

    # 1. Aggregate all required metrics
    layer_labels_f1, model_to_f1s = aggregate_model_metric(
        base_dir=base_dir, model_dirs=models, model_names=model_names,
        split="val_epoch", metric_key="f1"
    )
    _, model_to_precs = aggregate_model_metric(
        base_dir=base_dir, model_dirs=models, model_names=model_names,
        split="val_epoch", metric_key="precision"
    )
    _, model_to_recalls = aggregate_model_metric(
        base_dir=base_dir, model_dirs=models, model_names=model_names,
        split="val_epoch", metric_key="recall"
    )

    # 2. Call the new plotting function
    if layer_labels_f1: # Check if any data was loaded
        plot_combined_metric_heatmap(
            layer_labels=layer_labels_f1,
            model_to_f1s=model_to_f1s,
            model_to_precs=model_to_precs,
            model_to_recalls=model_to_recalls,
            save_path=combined_save,
            filename="combined_metrics_heatmap.png",
            title=r"Combined Metrics per Layer (F1-Score, Precision \& Recall)",
        )
    else:
        print("Skipping combined heatmap due to no data.")
