import os
import csv
import json
from dataclasses import asdict
from typing import Optional, Dict, Any, Tuple

from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataclasses import dataclass

@dataclass
class RunConfig:
    model_name: str = "default_model"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 10
    mixed_precision: bool = False
    grad_accum_steps: int = 1
    log_interval: int = 10
    lr: float = 1e-3
    dropout: float = 0.1

def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute Top-1 accuracy given raw logits (expects 2-class logits) and integer targets {0,1}."""
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0


def _safe_div(n: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Numerically safe elementwise division (tensor aware)."""
    d = torch.where(d == 0, torch.ones_like(d), d)
    return n / d


@torch.no_grad()
def batch_confusion_counts(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute TP/FP/FN for strictly BINARY single-label classification.
    Assumes logits shape [B, 2] and targets in {0,1}. Positive class is 1.
    Returns dict with:
      - tp, fp, fn: scalar tensors (torch.long)
      - N: scalar total samples in batch (torch.long)
    TN can be derived as: TN = N - TP - FP - FN.
    """
    preds = logits.argmax(dim=-1)      # [B]
    positive = torch.ones_like(preds)

    true_pos = (targets == positive)
    pred_pos = (preds == positive)

    tp = (pred_pos & true_pos).sum().to(torch.long)
    fp = (pred_pos & ~true_pos).sum().to(torch.long)
    fn = (~pred_pos & true_pos).sum().to(torch.long)
    n = torch.tensor(int(targets.numel()), dtype=torch.long)

    return {"tp": tp, "fp": fp, "fn": fn, "N": n}


def finalize_metrics(
    agg_tp: torch.Tensor,
    agg_fp: torch.Tensor,
    agg_fn: torch.Tensor,
    total_N: int,
) -> Tuple[float, float, float, int, int, int, int]:
    """
    Finalize BINARY metrics only. Positive class is 1.
    Returns (precision, recall, f1, TP, FP, FN, TN).
    """
    tp = agg_tp.to(torch.float64)
    fp = agg_fp.to(torch.float64)
    fn = agg_fn.to(torch.float64)

    precision = float(_safe_div(tp, tp + fp))
    recall = float(_safe_div(tp, tp + fn))
    denom = precision + recall
    f1 = float(0.0 if denom == 0.0 else (2 * precision * recall) / denom)

    TP = int(tp.item())
    FP = int(fp.item())
    FN = int(fn.item())
    TN = int(total_N - TP - FP - FN)

    return precision, recall, f1, TP, FP, FN, TN


class TrainerCaptions:
    """Handles training, evaluation, logging, and saving (binary classification only)."""
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
        self.run_dir = f"artifacts/{config.model_name}_run"
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

            for i, batch in enumerate(train_loader, start=1):
                inputs, targets = self._unpack_batch(batch, device)
                targets = targets.long()  # expected to be {0,1}

                logits = self.model(inputs)

                loss = self.criterion(logits, targets) / self.config.grad_accum_steps

                self.scaler.scale(loss).backward()

                if i % self.config.grad_accum_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                # Lightweight running metrics for quick feedback
                step_loss = loss.detach().item() * self.config.grad_accum_steps
                step_acc = accuracy_from_logits(logits.detach(), targets)
                running_loss += step_loss
                running_acc += step_acc
                count += 1

                # Step logging
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

            if count > 0:
                self._log({
                    "epoch": epoch,
                    "step": self.global_step,
                    "split": "train_epoch",
                    "loss": running_loss / count,
                    "accuracy": running_acc / count,
                    "lr": self._get_lr(),
                })

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
        """Evaluate on a dataloader and return binary metrics."""
        device = self.config.device
        self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        total_batches = 0

        # Accumulators for binary counts
        ep_tp = torch.tensor(0, dtype=torch.long)
        ep_fp = torch.tensor(0, dtype=torch.long)
        ep_fn = torch.tensor(0, dtype=torch.long)
        ep_N = 0

        for batch in loader:
            inputs, targets = self._unpack_batch(batch, device)
            targets = targets.long()  # {0,1}

            logits = self.model(inputs)
            loss = self.criterion(logits, targets)

            total_loss += loss.item()
            total_acc += accuracy_from_logits(logits, targets)
            total_batches += 1

            cnt = batch_confusion_counts(logits, targets)
            ep_tp = torch.tensor(0, dtype=torch.long, device=self.config.device)
            ep_fp = torch.tensor(0, dtype=torch.long, device=self.config.device)
            ep_fn = torch.tensor(0, dtype=torch.long, device=self.config.device)

            ep_N += int(cnt["N"])

        self.model.train()  # back to train mode

        if total_batches == 0:
            return {
                "loss": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            }

        precision, recall, f1, tp_val, fp_val, fn_val, tn_val = finalize_metrics(
            agg_tp=ep_tp,
            agg_fp=ep_fp,
            agg_fn=ep_fn,
            total_N=ep_N,
        )

        return {
            "loss": total_loss / total_batches,
            "accuracy": total_acc / total_batches,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp_val,
            "fp": fp_val,
            "fn": fn_val,
            "tn": tn_val,
        }

    def save_artifacts(self, filename_model: str = "model.pt", filename_optim: str = "optimizer.pt"):
        """Save model and optimizer state_dicts."""
        model_path = os.path.join(self.run_dir, filename_model)
        optim_path = os.path.join(self.run_dir, filename_optim)
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optim_path)

    def _unpack_batch(self, batch: Any, device: str):
        """
        Expects a batch like (inputs, targets) or {"inputs":..., "labels":...}.
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
        filled_row = {k: row.get(k, 0 if k in {"tp", "fp", "fn", "tn"} else 0.0) for k in keys}
        with open(self.log_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writerow(filled_row)

    def _save_config(self):
        """Persist basic configuration for later reproducibility/plotting."""
        cfg_path = os.path.join(self.run_dir, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, ensure_ascii=False, indent=2)

    def _get_lr(self) -> float:
        """Return current learning rate (assumes single param group)."""
        return self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 0.0
