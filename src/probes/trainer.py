import os
import csv
import json
from dataclasses import asdict
from typing import Optional, Dict, Any, Tuple

from tqdm import tqdm

from src.configs.ProbeRunConfig import RunConfig
import torch
from torch import nn
from torch.utils.data import DataLoader


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute accuracy given raw logits and integer targets."""
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0


def _safe_div(n: float, d: float) -> float:
    """Numerically safe division."""
    return float(n) / float(d) if d != 0 else 0.0


def _batch_micro_counts(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[int, int, int, int, int]:
    """
    Compute micro-averaged TP, FP, FN counts for a single-label classification batch.
    For multiclass, treat it as one-vs-rest and aggregate (micro).
    TN is derived at epoch end via: TN = num_classes * N - TP - FP - FN.

    Returns:
        tp, fp, fn, total_samples, num_classes
    """
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        correct = (preds == targets).sum().item()
        n = targets.numel()
        # In single-label multiclass, across one-vs-rest indicators:
        # total predicted positives = n, total true positives (positives in ground truth) = n.
        tp = int(correct)
        fp = int(n - tp)
        fn = int(n - tp)
        c = int(logits.shape[-1])
    return tp, fp, fn, n, c


class Trainer:
    """Handles training, evaluation, logging, and saving."""
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: RunConfig,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config

        # Prepare run directory
        self.run_dir = f"../artifacts/{config.model_name}_run"
        os.makedirs(self.run_dir, exist_ok=True)

        # Prepare logger
        self.log_path = os.path.join(self.run_dir, "training_log.csv")
        self._init_log()

        # Save config for reproducibility
        self._save_config()

        # AMP scaler (optional)
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)

        # Internal step counter
        self.global_step = 0

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None,
    ):
        """Train for N epochs; evaluate on val set if provided; save artifacts."""
        epochs = epochs or self.config.epochs
        print(f"Starting training for {epochs} epochs on device {self.config.device}")

        device = self.config.device
        self.model.train()

        for epoch in tqdm(range(1, epochs + 1), desc="Training Epochs"):
            running_loss = 0.0
            running_acc = 0.0
            count = 0

            # Epoch-level micro counts
            ep_tp = ep_fp = ep_fn = 0
            ep_n = 0
            ep_num_classes = None

            for i, batch in enumerate(train_loader, start=1):
                inputs, targets = self._unpack_batch(batch, device)

                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                    logits = self.model(inputs)
                    loss = self.criterion(logits, targets) / self.config.grad_accum_steps

                self.scaler.scale(loss).backward()

                if i % self.config.grad_accum_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                # Update running metrics
                step_loss = loss.detach().item() * self.config.grad_accum_steps
                step_acc = accuracy_from_logits(logits.detach(), targets)
                running_loss += step_loss
                running_acc += step_acc
                count += 1

                # Update epoch micro counts
                tp, fp, fn, n, c = _batch_micro_counts(logits.detach(), targets)
                ep_tp += tp
                ep_fp += fp
                ep_fn += fn
                ep_n += n
                ep_num_classes = ep_num_classes or c  # set once

                # Logging
                self.global_step += 1
                if self.global_step % self.config.log_interval == 0:
                    self._log({
                        "epoch": epoch,
                        "step": self.global_step,
                        "split": "train",
                        "loss": step_loss,
                        "accuracy": step_acc,
                        "lr": self._get_lr(),
                    })

            # Epoch-end logging (averages + micro metrics over the training batches seen this epoch)
            if count > 0:
                precision, recall, f1, tn = self._finalize_micro_metrics(
                    tp=ep_tp,
                    fp=ep_fp,
                    fn=ep_fn,
                    total_samples=ep_n,
                    num_classes=ep_num_classes or 1,
                )
                self._log({
                    "epoch": epoch,
                    "step": self.global_step,
                    "split": "train_epoch",
                    "loss": running_loss / count,
                    "accuracy": running_acc / count,
                    "lr": self._get_lr(),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "tp": ep_tp,
                    "fp": ep_fp,
                    "fn": ep_fn,
                    "tn": tn,
                })

            # Explicit evaluation on the training set (probing) at epoch end
            train_eval = self.evaluate(train_loader)
            self._log({
                "epoch": epoch,
                "step": self.global_step,
                "split": "train_eval_epoch",
                **train_eval,
                "lr": self._get_lr(),
            })

            # Validation
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                self._log({
                    "epoch": epoch,
                    "step": self.global_step,
                    "split": "val_epoch",
                    **val_metrics,
                    "lr": self._get_lr(),
                })

        self.save_artifacts()

    @torch.inference_mode()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate on a dataloader and return metrics (loss, accuracy, precision, recall, f1, tp, fp, fn, tn)."""
        device = self.config.device
        self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        total_batches = 0

        # Micro counts
        ep_tp = ep_fp = ep_fn = 0
        ep_n = 0
        ep_num_classes = None

        for batch in loader:
            inputs, targets = self._unpack_batch(batch, device)
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)

            total_loss += loss.item()
            total_acc += accuracy_from_logits(logits, targets)
            total_batches += 1

            tp, fp, fn, n, c = _batch_micro_counts(logits, targets)
            ep_tp += tp
            ep_fp += fp
            ep_fn += fn
            ep_n += n
            ep_num_classes = ep_num_classes or c

        self.model.train()

        if total_batches == 0:
            return {
                "loss": 0.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 0,
            }

        precision, recall, f1, tn = self._finalize_micro_metrics(
            tp=ep_tp,
            fp=ep_fp,
            fn=ep_fn,
            total_samples=ep_n,
            num_classes=ep_num_classes or 1,
        )

        return {
            "loss": total_loss / total_batches,
            "accuracy": total_acc / total_batches,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": ep_tp,
            "fp": ep_fp,
            "fn": ep_fn,
            "tn": tn,
        }

    def save_artifacts(self, filename_model: str = "model.pt", filename_optim: str = "optimizer.pt"):
        """Save model and optimizer state_dicts."""
        model_path = os.path.join(self.run_dir, filename_model)
        optim_path = os.path.join(self.run_dir, filename_optim)
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optim_path)

    def _unpack_batch(self, batch: Any, device: str):
        """
        Expects a batch like (inputs, targets) where:
        - inputs: torch.Tensor of shape [B, ...]
        - targets: torch.LongTensor of shape [B]
        Adjust here if your dataset uses dicts.
        """
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            inputs, targets = batch[0], batch[1]
        elif isinstance(batch, dict) and "inputs" in batch and "labels" in batch:
            inputs, targets = batch["inputs"], batch["labels"]
        else:
            raise ValueError("Unsupported batch format.")

        return inputs.to(device), targets.to(device)

    def _init_log(self):
        """Create CSV log with header if it does not exist."""
        if not os.path.exists(self.log_path):
            with open(self.log_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "epoch", "step", "split", "loss", "accuracy", "lr",
                        "precision", "recall", "f1", "tp", "fp", "fn", "tn"
                    ]
                )
                writer.writeheader()

    def _log(self, row: Dict[str, Any]):
        """Append a row to the CSV log."""
        keys = [
            "epoch", "step", "split", "loss", "accuracy", "lr",
            "precision", "recall", "f1", "tp", "fp", "fn", "tn"
        ]
        # Ensure all keys exist (keep CSV schema stable).
        filled = {}
        for k in keys:
            if k in row:
                filled[k] = row[k]
            else:
                filled[k] = 0 if k in {"tp", "fp", "fn", "tn"} else 0.0
        with open(self.log_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writerow(filled)

    def _save_config(self):
        """Persist basic configuration for later reproducibility/plotting."""
        cfg_path = os.path.join(self.run_dir, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, ensure_ascii=False, indent=2)

    def _get_lr(self) -> float:
        """Return current learning rate (assumes single param group)."""
        return self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 0.0

    def _finalize_micro_metrics(
        self,
        tp: int,
        fp: int,
        fn: int,
        total_samples: int,
        num_classes: int,
    ) -> Tuple[float, float, float, int]:
        """
        Compute precision/recall/F1 and TN from micro counts.
        For multiclass (single-label), TN is derived as:
        TN = num_classes * total_samples - TP - FP - FN
        """
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
        tn_total = int(num_classes * total_samples - tp - fp - fn)
        return precision, recall, f1, tn_total
