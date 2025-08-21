import importlib
import math
import numpy as np
import os, sys

from pyparsing import col
os.environ["PYTHONPATH"] = "../"
sys.path.insert(0, "../")

from src.vllm.qwen import QwenVLProbe
from src.vllm.automodel import AutoModelVLM
import torch
from src.data.dataset_loader import DSLoader
from src.probes.classifier import build_classifier
from src.probes.trainer import Trainer, RunConfig
import torch
from torch.utils.data import DataLoader, TensorDataset
import src.vllm.qwen
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def load_category_ds():
    ds_loader = DSLoader(split="train")
    ds = ds_loader.get_category_ds()
    return ds

def load_caption_ds():
    ds_loader = DSLoader(split="train")
    ds = ds_loader.get_caption_ds()
    return ds

def get_repr(model,text,img):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img,
                },
                {"type": "text", "text": text},
            ],
        }
    ]

    hidden_out, label_out = model.get_hidden_states_batched(
        examples=[{"label":0, "messages":messages}],
        # output_layer="last_non_padding",
        output_layer="mean",
        return_layer=None,
        batch_size=8,
    )
    return hidden_out, label_out

def get_repr_for_layer(hidden_out, layer_num):
    return hidden_out[:, layer_num, :]


def train_probe(layer_repr_train, labels_train,layer_repr_eval,labels_eval,name):

    X_train = torch.stack([r.squeeze(0) for r in layer_repr_train]).to(torch.float32)
    y_train = torch.tensor(labels_train, dtype=torch.long)

    X_eval = torch.stack([r.squeeze(0) for r in layer_repr_eval]).to(torch.float32)
    y_eval = torch.tensor(labels_eval, dtype=torch.long)

    dataset_train = TensorDataset(X_train, y_train)
    dataset_eval = TensorDataset(X_eval, y_eval)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_labels = 2

    emb_dim = layer_repr_train[0].shape[1]
    model_head, criterion, optimizer = build_classifier(emb_dim, num_labels, device, lr=1e-3, dropout=0.1)

    config = RunConfig(
        model_name=name,
        device=device,
        lr=1e-3,
        dropout=0.1,
        epochs=20,
        log_interval=20,
        mixed_precision=False
    )


    train_loader = DataLoader(dataset_train,  batch_size=16, shuffle=True)
    eval_loader = DataLoader(dataset_eval, batch_size=16, shuffle=False)

    trainer = Trainer(model_head, criterion, optimizer, config)
    trainer.fit(train_loader, eval_loader)

def list_subfolders(directory):
    directory = Path(directory)
    return [p for p in directory.rglob("*") if p.is_dir()]

def create_plots(dir_path, save_path, split="val_epoch"):
    """
    Read per-subfolder training_log.csv files, pick the last row for the given split,
    and plot:
      1) Loss per layer
      2) Accuracy / Precision / Recall / F1 per layer
      3) Confusion counts (TP / FP / FN / TN) per layer
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_path = Path(save_path)

    dir_path = Path(dir_path)


    losses, accs, precs, recalls, f1s = [], [], [], [], []
    tps, fps, fns, tns = [], [], [], []
    layer_names = []

    subfolders = list_subfolders(dir_path)

    for sub in sorted(subfolders):
        csv_path = sub / "training_log.csv"
        if not csv_path.exists():
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        if "split" not in df.columns:
            continue

        df_split = df[df["split"] == split]
        if df_split.empty:
            continue

        row = df_split.iloc[-1]

        def get(col, default=float("nan")):
            if col in df_split.columns:
                try:
                    return float(row[col])
                except ValueError:
                    return default
            return default


        losses.append(get("loss"))
        accs.append(get("accuracy"))
        precs.append(get("precision"))
        recalls.append(get("recall"))
        f1s.append(get("f1"))

        tps.append(get("tp", 0))
        fps.append(get("fp", 0))
        fns.append(get("fn", 0))
        tns.append(get("tn", 0))

        layer_names.append(sub.name)

    if not losses:
        print("No data found. Ensure subfolders contain training_log.csv with the requested split.")
        return

    x = list(range(len(layer_names)))
    xticks_labels = [str(i) for i in x]

    # Loss per layer
    plt.figure(figsize=(10, 4))
    plt.plot(x, losses, marker='o')
    plt.xticks(x, xticks_labels, rotation=45, ha="right")
    plt.xlabel("Layer")
    plt.ylabel("Loss")
    plt.title(f"Loss per Layer")
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path / "loss_per_layer.png")

    # Metrics (accuracy, precision, recall, f1) per layer
    plt.figure(figsize=(10, 4))


    plt.plot(x, accs, marker='o', label="Accuracy")
    plt.plot(x, precs, marker='o', label="Precision")
    plt.plot(x, recalls, marker='o', label="Recall")
    plt.plot(x, f1s, marker='o', label="F1")

    plt.xticks(x, xticks_labels, rotation=45, ha="right")
    plt.xlabel("Layer")
    plt.ylabel("Score")
    plt.title(f"Classification Metrics per Layer")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path / "classification_metrics_per_layer.png")

    # Confusion counts (grouped bars) per layer
    width = 0.2
    plt.figure(figsize=(12, 5))

    # Offsets for grouped bars
    x_tp = [i - 1.5*width for i in x]
    x_fp = [i - 0.5*width for i in x]
    x_fn = [i + 0.5*width for i in x]
    x_tn = [i + 1.5*width for i in x]

    plt.bar(x_tp, tps, width=width, label="TP")
    plt.bar(x_fp, fps, width=width, label="FP")
    plt.bar(x_fn, fns, width=width, label="FN")
    plt.bar(x_tn, tns, width=width, label="TN")

    plt.xticks(x, xticks_labels, rotation=45, ha="right")
    plt.xlabel("Layer")
    plt.ylabel("Count")
    plt.title(f"Confusion Counts per Layer")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path / "confusion_counts_per_layer.png")

def load_captions_prompt():
    with open("src/prompts/global_features.txt", "r") as f:
        return f.read().strip()
