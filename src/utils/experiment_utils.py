import importlib
import math
from typing import Optional
import numpy as np
import os, sys

from pyparsing import col
os.environ["PYTHONPATH"] = "../"
sys.path.insert(0, "../")

import torch
from src.data.dataset_loader import DSLoader
from src.probes.classifier_captions import build_classifier
from src.probes.classifier_category import build_classifier_category
from src.probes.trainer_captions import TrainerCaptions, RunConfig
from src.probes.trainer_category import Trainer_category, RunConfig_category
import torch
from torch.utils.data import DataLoader, TensorDataset
import src.vllm.qwen
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path



def pool_tokens(
        layer_tensor: torch.Tensor,         # [B, S, H]
        attn_mask: Optional[torch.Tensor],  # [B, S] or None
        mode: str,
    ) -> torch.Tensor:
        """Vectorized pooling across the batch. Returns: [B, H]."""
        B, S, H = layer_tensor.shape
        device = layer_tensor.device
        if attn_mask is None:
            attn_mask = torch.ones((B, S), dtype=torch.long, device=device)

        lengths = attn_mask.sum(dim=1).clamp_min(1)  # [B]

        if mode == "CLS":
            return layer_tensor[:, 0, :]

        if mode == "mean":
            mask = attn_mask.unsqueeze(-1)
            summed = (layer_tensor * mask).sum(dim=1)
            return summed / lengths.unsqueeze(-1)

        if mode == "max":
            very_neg = torch.finfo(layer_tensor.dtype).min
            masked = layer_tensor.clone()
            masked[attn_mask == 0] = very_neg
            return masked.max(dim=1).values

        if mode.startswith("token_index_"):
            try:
                idx = int(mode.split("_")[-1])
            except Exception:
                idx = 0
            clamped = torch.clamp(torch.full_like(lengths, idx), min=0, max=(lengths - 1))
            gather_index = clamped.view(B, 1, 1).expand(B, 1, H)
            return layer_tensor.gather(dim=1, index=gather_index).squeeze(1)

        # Default: last non-padding token
        last_idx = (lengths - 1).view(B, 1, 1).expand(B, 1, H)
        return layer_tensor.gather(dim=1, index=last_idx).squeeze(1)

def load_category_ds():
    ds_loader = DSLoader(split="train")
    ds = ds_loader.get_category_ds()
    categories = ds_loader.get_categores()
    return ds, categories

def load_caption_ds():
    ds_loader = DSLoader(split="train")
    ds = ds_loader.get_caption_ds()
    return ds

def get_repr_batch(model,texts,imgs):

    examples = []
    for text, img in zip(texts, imgs):
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
        examples.append({"label": 0, "messages": messages})

    hidden_out, label_out = model.get_hidden_states_batched(
        examples=examples,
        # output_layer="last_non_padding",
        output_layer="mean",
        return_layer=None,
        batch_size=2,
    )
    return hidden_out, label_out


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
        batch_size=1,
    )
    return hidden_out, label_out

def get_repr_for_layer(hidden_out, layer_num):
    return hidden_out[layer_num, :]


def train_probe(layer_repr_train, labels_train,layer_repr_eval,labels_eval,name):

    X_train = torch.stack([r.squeeze(0) for r in layer_repr_train]).to(torch.float32)
    y_train = torch.tensor(labels_train, dtype=torch.long)

    X_eval = torch.stack([r.squeeze(0) for r in layer_repr_eval]).to(torch.float32)
    y_eval = torch.tensor(labels_eval, dtype=torch.long)

    dataset_train = TensorDataset(X_train, y_train)
    dataset_eval = TensorDataset(X_eval, y_eval)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_labels = 2

    emb_dim = layer_repr_train[0].shape[0]
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

    trainer = TrainerCaptions(model_head, criterion, optimizer, config)
    trainer.fit(train_loader, eval_loader)

def train_probe_local(layer_repr_train, labels_train, layer_repr_eval, labels_eval, name):

    X_train = torch.stack([r.squeeze(0) for r in layer_repr_train]).to(torch.float32)
    y_train = torch.tensor([l for (l, m) in labels_train], dtype=torch.float32)
    m_train = torch.tensor([m for (l, m) in labels_train], dtype=torch.float32)

    X_eval = torch.stack([r.squeeze(0) for r in layer_repr_eval]).to(torch.float32)
    y_eval = torch.tensor([l for (l, m) in labels_eval], dtype=torch.float32)
    m_eval = torch.tensor([m for (l, m) in labels_eval], dtype=torch.float32)

    dataset_train = TensorDataset(X_train, y_train, m_train)
    dataset_eval = TensorDataset(X_eval, y_eval, m_eval)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_labels = len(load_category_ds()[1])

    emb_dim = layer_repr_train[0].shape[0]
    model_head, criterion, optimizer = build_classifier_category(emb_dim, num_labels, device, lr=1e-3, dropout=0.1)

    config = RunConfig_category(
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

    trainer = Trainer_category(model_head, criterion, optimizer, config)
    trainer.fit(train_loader, eval_loader)


def load_captions_prompt():
    with open("src/prompts/global_features.txt", "r") as f:
        return f.read().strip()

def load_categories_prompt():
    with open("src/prompts/local_features.txt", "r") as f:
        return f.read().strip()
