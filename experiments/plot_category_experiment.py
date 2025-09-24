# -*- coding: utf-8 -*-
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from tueplots import bundles
import numpy as np
import re

plt.rcParams.update(bundles.icml2022())
plt.rcParams["savefig.dpi"] = 500


PLOT_SIZE = (5, 2)
PLOT_SIZE2 = (3, 2)
WIDE_PLOT_SIZE = (5, 2)
COMBINED_MIN_WIDTH = 3.5
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

# FINALE, ROBUSTE VERSION DER FUNKTION
def extract_number_from_path(path: Path) -> float:
    """
    Extracts the layer number from a path name.
    It specifically looks for 'layer' followed by digits to be robust.
    """
    # re.IGNORECASE sorgt dafür, dass "layer" und "Layer" gefunden werden.
    # Die Klammern um (\d+) erstellen eine "capturing group".
    match = re.search(r'layer(\d+)', path.name, re.IGNORECASE)
    if match:
        # match.group(1) gibt den Inhalt der ersten capturing group zurück (nur die Zahl).
        return int(match.group(1))
    return float('inf')  # Ordner ohne "layer<zahl>" werden ans Ende sortiert


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

    subfolders = list_subfolders(dir_path)
    for sub in sorted(subfolders, key=extract_number_from_path):
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
        "losses": losses, "accs": accs, "precs": precs, "recalls": recalls, "f1s": f1s,
        "tps": tps, "fps": fps, "fns": fns, "tns": tns, "layer_names": layer_names,
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
    # Skipping to plot the baseline as the positive and negative categories are not balanced
    plt.axhline(0.5, color="white", linestyle="--", linewidth=1, label="Baseline (0.5)")
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
    base_dir: Path, model_dirs: list[str], model_names: list[str], split: str
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
    base_dir: Path, model_dirs: list[str], model_names: list[str], split: str, metric_key: str,
) -> tuple[list[str], dict[str, list[float]]]:
    """
    Aggregate per-layer values for a given metric across multiple models.
    metric_key in {"accuracy", "precision", "recall", "f1"}.
    Returns (layer_labels, {model_name: values_per_layer}), aligned to min common depth.
    """
    keymap = {"accuracy": "accs", "precision": "precs", "recall": "recalls", "f1": "f1s"}
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
    layer_labels: list[str], model_to_f1s: dict[str, list[float]], model_to_precs: dict[str, list[float]],
    model_to_recalls: dict[str, list[float]], save_path: Path, filename: str, title: str,
) -> None:
    """
    Plot a combined heatmap: F1-score for color, Precision/Recall for annotation.
    """
    if not layer_labels or not model_to_f1s:
        print(f"No data for combined heatmap: {title}")
        return
    ensure_dir(save_path)
    model_names = list(model_to_f1s.keys())
    num_models = len(model_names)
    num_layers = len(layer_labels)
    f1_mat = np.array([model_to_f1s[m][:num_layers] for m in model_names], dtype=float).T
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
    for i in range(num_layers):
        for j in range(num_models):
            model_name = model_names[j]
            prec = model_to_precs.get(model_name, [])[i]
            recall = model_to_recalls.get(model_name, [])[i]
            if np.isfinite(prec) and np.isfinite(recall):
                text_label = f"P:{prec:.2f} R:{recall:.2f}"
                ax.text(j, i, text_label, ha="center", va="center", fontsize=7, color='white', bbox=dict(facecolor="black", alpha=0.2, boxstyle="square,pad=0.15"))
    fig.savefig(save_path / filename)
    plt.close(fig)


def plot_accuracy_lines_per_layer(
    layer_labels: list[str], model_to_accs: dict[str, list[float]], save_path: Path,
    filename: str = "accuracy_lines_per_layer.png", title: str = "Accuracy per Layer across Models"
) -> None:
    """
    Create a line plot showing accuracy per layer for each model.
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
    # Skipping to plot the baseline as the positive and negative categories are not balanced
    plt.axhline(0.5, color="white", linestyle="--", linewidth=1, label="Baseline (0.5)")
    plt.xticks(x, layer_labels, rotation=45, ha="right")
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / filename)
    plt.close()


def extract_last_run_losses(csv_path: Path) -> tuple[pd.Series | None, pd.Series | None]:
    """
    Reads a training log, finds the last experimental run, and extracts
    the mean training and validation loss per epoch for that run.
    """
    if not csv_path.exists():
        return None, None
    try:
        df = pd.read_csv(csv_path)
        required_cols = ["epoch", "step", "split", "loss"]
        if not all(col in df.columns for col in required_cols):
            return None, None
    except Exception:
        return None, None
    start_indices = df.index[(df["epoch"] == 1) & (df["step"] == 20)].tolist()
    if not start_indices:
        start_indices = df.index[df["epoch"] == 1].tolist()
        if not start_indices:
            return None, None
    last_run_start_index = start_indices[-1]
    df_last_run = df.iloc[last_run_start_index:].copy()
    df_train = df_last_run[df_last_run["split"] == "train"]
    train_losses = df_train.groupby("epoch")["loss"].mean()
    df_val = df_last_run[df_last_run["split"] == "val_epoch"]
    val_losses = df_val.groupby("epoch")["loss"].mean()
    return train_losses, val_losses


def aggregate_model_training_progress(
    base_dir: Path, model_dirs: list[str], model_names: list[str], layer: int = 10,
) -> dict[str, dict]:
    """
    Aggregates training and validation loss curves for a SPECIFIC layer across multiple models.
    """
    all_models_data = {}
    print(f"Aggregating training progress for layer: {layer}")
    search_pattern = f"_layer{layer}_"
    for m_dir, m_name in zip(model_dirs, model_names):
        model_path = base_dir / m_dir
        was_found = False
        subfolders = list_subfolders(model_path)
        for subfolder in sorted(subfolders, key=extract_number_from_path):
            if search_pattern in subfolder.name:
                csv_path = subfolder / "training_log.csv"
                if not csv_path.exists():
                    print(f"  - INFO: Found matching folder '{subfolder.name}', but log file is missing. Skipping.")
                    continue
                train_losses, val_losses = extract_last_run_losses(csv_path)
                if train_losses is None and val_losses is None:
                    print(f"  - WARNING: No data could be extracted from '{csv_path}'.")
                    continue
                df_train = train_losses.rename("train") if train_losses is not None else pd.Series(dtype=float)
                df_val = val_losses.rename("val") if val_losses is not None else pd.Series(dtype=float)
                model_df = pd.concat([df_train, df_val], axis=1)
                all_models_data[m_name] = {
                    "train_loss": model_df["train"].tolist(),
                    "val_loss": model_df["val"].tolist(),
                    "epochs": model_df.index.tolist(),
                }
                was_found = True
                print(f"  - Found and processed data for model '{m_name}' in '{subfolder.name}'")
                break
        if not was_found:
            print(f"  - WARNING: No subfolder containing '{search_pattern}' with a valid log file was found for model '{m_name}'.")
    return all_models_data


def plot_combined_training_progress(
    progress_data: dict, save_path: Path, layer: int, title: str | None = None,
) -> None:
    """
    Plots the training and validation loss for a specific layer for multiple models.
    """
    if not progress_data:
        print(f"No data available for training progress plot for layer {layer}.")
        return
    ensure_dir(save_path)
    plt.figure(figsize=PLOT_SIZE2)
    ax = plt.gca()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (model_name, data) in enumerate(progress_data.items()):
        color = colors[i % len(colors)]
        epochs = data["epochs"]
        if data["train_loss"]:
            ax.plot(epochs, data["train_loss"], label=f"{model_name} Train", color=color, linestyle="-")
        if data["val_loss"]:
            ax.plot(epochs, data["val_loss"], label=f"{model_name} Val", color=color, linestyle="--")
    final_title = title if title else f"Training vs. Validation Loss for Layer {layer}"
    ax.set_title(final_title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    filename = f"combined_training_progress_layer_{layer}.png"
    plt.savefig(save_path / filename)
    plt.close()


def create_plots(dir_path, save_path, split: str = "val_epoch") -> None:
    """
    Read per-subfolder training_log.csv files, pick the last row for the given split, and plot.
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

def find_best_and_worst_layers(
    base_dir: Path, model_dirs: list[str], model_names: list[str], split: str = "val_epoch"
):
    layer_labels, model_to_f1s = aggregate_model_metric(
        base_dir=base_dir,
        model_dirs=model_dirs,
        model_names=model_names,
        split=split,
        metric_key="f1",
    )

    if not layer_labels or not model_to_f1s:
        print("No F1 data found across models.")
        return None, None

    f1_matrix = np.array([model_to_f1s[m] for m in model_names])
    mean_f1s = f1_matrix.mean(axis=0)

    best_idx = int(np.nanargmax(mean_f1s))
    worst_idx = int(np.nanargmin(mean_f1s))

    best_layer = (layer_labels[best_idx], mean_f1s[best_idx])
    worst_layer = (layer_labels[worst_idx], mean_f1s[worst_idx])

    return best_layer, worst_layer

if __name__ == "__main__":
    models = [
        "exp2_1/",
        "exp2_3/",
    ]
    model_names = ["Qwen2-VL-2B", "FastVLM-0.5B"]

    for model_dir in models:
        ensure_dir(Path(f"report/figures/local/{model_dir}"))
        create_plots(f"artifacts/{model_dir}", f"report/figures/local/{model_dir}")

    base_dir = Path("artifacts")
    combined_save = ensure_dir(Path("report/figures/local/_combined_exp2"))

    # Combined line plot for accuracy
    layer_labels, model_to_accs = aggregate_model_accuracies(
        base_dir=base_dir, model_dirs=models, model_names=model_names, split="val_epoch",
    )
    plot_accuracy_lines_per_layer(
        layer_labels=layer_labels, model_to_accs=model_to_accs, save_path=combined_save,
        filename="accuracy_lines_per_layer.png", title="Accuracy per Layer across Models",
    )

    # Combined Heatmap for F1, Precision, and Recall
    print("Generating combined metrics heatmap...")
    layer_labels_f1, model_to_f1s = aggregate_model_metric(
        base_dir=base_dir, model_dirs=models, model_names=model_names, split="val_epoch", metric_key="f1"
    )
    _, model_to_precs = aggregate_model_metric(
        base_dir=base_dir, model_dirs=models, model_names=model_names, split="val_epoch", metric_key="precision"
    )
    _, model_to_recalls = aggregate_model_metric(
        base_dir=base_dir, model_dirs=models, model_names=model_names, split="val_epoch", metric_key="recall"
    )

    if layer_labels_f1:
        plot_combined_metric_heatmap(
            layer_labels=layer_labels_f1, model_to_f1s=model_to_f1s, model_to_precs=model_to_precs,
            model_to_recalls=model_to_recalls, save_path=combined_save, filename="combined_metrics_heatmap.png",
            title=r"Combined Metrics per Layer (F1-Score, Precision \& Recall)",
        )

        # Combined line plot for F1 Score
        plot_accuracy_lines_per_layer(
            layer_labels=layer_labels_f1,
            model_to_accs=model_to_f1s,
            save_path=combined_save,
            filename="f1_lines_per_layer.png",
            title="F1 per Layer across Models",
        )
    else:
        print("Skipping combined heatmap due to no data.")

    print("\nGenerating combined training progress plot...")

    best, worst = find_best_and_worst_layers(
    base_dir=Path("artifacts"),
    model_dirs=models,
    model_names=model_names,
    split="val_epoch"
    )
    print("Best Layer (avg F1 across models):", best)
    print("Worst Layer (avg F1 across models):", worst)

    print("\nGenerating combined training progress plot...")
    for layer in [int(best[0]), int(worst[0])]:
        progress_data = aggregate_model_training_progress(
            base_dir=base_dir, model_dirs=models, model_names=model_names, layer=layer,
        )

        if progress_data:
            plot_combined_training_progress(
                progress_data=progress_data, save_path=combined_save, layer=layer,
            )
            print(f"Combined training plot saved for layer {layer}.")
        else:
            print("Skipping combined training progress plot due to no data.")
