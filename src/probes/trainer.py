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
    """Compute Top-1 accuracy given raw logits and integer targets."""
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0


def _safe_div(n: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Numerically safe elementwise division (tensor aware)."""
    # Use torch.where to avoid division by zero, replacing d with 1.0 where d is 0.
    d = torch.where(d == 0, torch.ones_like(d), d)
    return n / d


@torch.no_grad()
def batch_confusion_counts(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute per-class TP/FP/FN for single-label classification (binary or multiclass) on a batch.
    Returns dict with:
      - tp, fp, fn: tensors of shape [C]
      - N: scalar total samples in batch
      - C: scalar number of classes
    TN is derived later per class via: TN_k = N - TP_k - FP_k - FN_k.
    """
    preds = logits.argmax(dim=-1)       # [B]
    c = int(logits.shape[-1])
    n = int(targets.numel())

    tp = torch.zeros(c, dtype=torch.long)
    fp = torch.zeros(c, dtype=torch.long)
    fn = torch.zeros(c, dtype=torch.long)

    for k in range(c):
        pred_k = (preds == k)
        true_k = (targets == k)
        tp[k] = (pred_k & true_k).sum()
        fp[k] = (pred_k & ~true_k).sum()
        fn[k] = (~pred_k & true_k).sum()

    return {"tp": tp, "fp": fp, "fn": fn, "N": torch.tensor(n), "C": torch.tensor(c)}


def finalize_metrics(
    agg_tp: torch.Tensor,
    agg_fp: torch.Tensor,
    agg_fn: torch.Tensor,
    total_N: int,
    num_classes: int,
    average: str = "auto",  # "auto" | "micro" | "macro" | "weighted" | "binary"
) -> Tuple[float, float, float, int, int, int, int]:
    """
    Compute (precision, recall, f1, TP, FP, FN, TN) from accumulated per-class counts.
    - micro: sums counts across classes before ratios (TP, FP, FN returned are sums)
    - macro: mean of per-class metrics (TP, FP, FN returned are sums)
    - weighted: support-weighted mean of per-class metrics (TP, FP, FN returned are sums)
    - binary: classic binary view (TP, FP, FN returned are for the positive class 1)
    - auto: "binary" if C==2 else "macro"
    """
    tp = agg_tp.to(torch.float64)
    fp = agg_fp.to(torch.float64)
    fn = agg_fn.to(torch.float64)

    support = tp + fn

    # Resolve averaging strategy
    if average == "auto":
        average = "binary" if num_classes == 2 else "macro"

    print(f"Finalizing metrics with average='{average}', num_classes={num_classes}")

    # =================================================================================
    # FIX: START of corrected logic for binary classification
    # =================================================================================
    if average == "binary":
        if num_classes != 2:
            # Fallback for safety, although 'auto' logic prevents this.
            average = "macro"
        else:
            # Standard binary metrics are calculated w.r.t. the "positive" class (class 1).
            TP = tp[1]
            FP = fp[1]
            FN = fn[1]
            # TN for the positive class is everything that was not class 1 and not predicted as class 1.
            # This is equivalent to N - TP - FP - FN for that class.
            TN = total_N - TP - FP - FN

            precision = float(_safe_div(TP, TP + FP))
            recall = float(_safe_div(TP, TP + FN))
            f1_denom = precision + recall
            f1 = float(0.0 if f1_denom == 0.0 else (2 * precision * recall) / f1_denom)
            
            return precision, recall, f1, int(TP.item()), int(FP.item()), int(FN.item()), int(TN.item())
    # =================================================================================
    # FIX: END of corrected logic
    # =================================================================================

    if average == "micro":
        TP = tp.sum()
        FP = fp.sum()
        FN = fn.sum()
        TN = (total_N - tp - fp - fn).sum() # Sum of TNs for all classes
        
        precision = float(_safe_div(TP, TP + FP))
        recall = float(_safe_div(TP, TP + FN))
        f1_denom = precision + recall
        f1 = float(0.0 if f1_denom == 0.0 else (2 * precision * recall) / f1_denom)
        
        return precision, recall, f1, int(TP.item()), int(FP.item()), int(FN.item()), int(TN.item())

    # Per-class metrics for macro and weighted
    prec_c = _safe_div(tp, tp + fp)    # [C]
    rec_c  = _safe_div(tp, tp + fn)    # [C]
    f1_denom_c  = prec_c + rec_c
    f1_c   = torch.where(f1_denom_c == 0, torch.zeros_like(f1_denom_c), 2 * prec_c * rec_c / f1_denom_c)
    tn_total = int((total_N - tp - fp - fn).sum().item())

    if average == "macro":
        precision = float(prec_c.mean())
        recall = float(rec_c.mean())
        f1 = float(f1_c.mean())
        return precision, recall, f1, int(tp.sum().item()), int(fp.sum().item()), int(fn.sum().item()), tn_total

    if average == "weighted":
        total_support = support.sum()
        weights = torch.zeros_like(support) if total_support.item() == 0 else support / total_support
        
        precision = float((prec_c * weights).sum())
        recall = float((rec_c * weights).sum())
        f1 = float((f1_c * weights).sum())
        return precision, recall, f1, int(tp.sum().item()), int(fp.sum().item()), int(fn.sum().item()), tn_total

    # Fallback, should not be reached with the current logic but good practice
    TP = tp.sum(); FP = fp.sum(); FN = fn.sum(); TN = (total_N - tp - fp - fn).sum()
    precision = float(_safe_div(TP, TP + FP))
    recall = float(_safe_div(TP, TP + FN))
    f1_denom = precision + recall
    f1 = float(0.0 if f1_denom == 0.0 else (2 * precision * recall) / f1_denom)
    return precision, recall, f1, int(TP.item()), int(FP.item()), int(FN.item()), int(TN.item())


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

            # Epoch-level accumulators (per-class)
            ep_tp = ep_fp = ep_fn = None
            ep_N = 0
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

                # Running metrics
                step_loss = loss.detach().item() * self.config.grad_accum_steps
                step_acc = accuracy_from_logits(logits.detach(), targets)
                running_loss += step_loss
                running_acc += step_acc
                count += 1

                # Update epoch per-class counts
                cnt = batch_confusion_counts(logits.detach(), targets)
                if ep_tp is None:
                    ep_tp = cnt["tp"].clone()
                    ep_fp = cnt["fp"].clone()
                    ep_fn = cnt["fn"].clone()
                else:
                    ep_tp += cnt["tp"]
                    ep_fp += cnt["fp"]
                    ep_fn += cnt["fn"]
                ep_N += int(cnt["N"])
                ep_num_classes = ep_num_classes or int(cnt["C"])

                # Step logging (optional granularity)
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

            # Epoch-end logging on training stream
            if count > 0 and ep_tp is not None:
                precision, recall, f1, tp_val, fp_val, fn_val, tn_val = finalize_metrics(
                    agg_tp=ep_tp,
                    agg_fp=ep_fp,
                    agg_fn=ep_fn,
                    total_N=ep_N,
                    num_classes=ep_num_classes or 1,
                    average="auto",
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
                    "tp": tp_val,
                    "fp": fp_val,
                    "fn": fn_val,
                    "tn": tn_val,
                })

            # Explicit evaluation on the training set (clean eval mode)
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
        """Evaluate on a dataloader and return metrics."""
        device = self.config.device
        self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        total_batches = 0

        # Epoch-level accumulators for evaluation
        ep_tp = ep_fp = ep_fn = None
        ep_N = 0
        ep_num_classes = None

        for batch in loader:
            inputs, targets = self._unpack_batch(batch, device)
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)

            total_loss += loss.item()
            total_acc += accuracy_from_logits(logits, targets)
            total_batches += 1

            # Accumulate confusion matrix counts
            cnt = batch_confusion_counts(logits, targets)
            if ep_tp is None:
                ep_tp = cnt["tp"].clone()
                ep_fp = cnt["fp"].clone()
                ep_fn = cnt["fn"].clone()
            else:
                ep_tp += cnt["tp"]
                ep_fp += cnt["fp"]
                ep_fn += cnt["fn"]
            ep_N += int(cnt["N"])
            ep_num_classes = ep_num_classes or int(cnt["C"])

        self.model.train() # Set model back to training mode

        if total_batches == 0 or ep_tp is None:
            return {
                "loss": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            }

        # Finalize metrics using the corrected function
        precision, recall, f1, tp_val, fp_val, fn_val, tn_val = finalize_metrics(
            agg_tp=ep_tp,
            agg_fp=ep_fp,
            agg_fn=ep_fn,
            total_N=ep_N,
            num_classes=ep_num_classes or 1,
            average="auto",
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
        # Fill missing keys with default values for consistent CSV rows
        filled_row = {k: row.get(k, 0 if k in {"tp", "fp", "fn", "tn"} else 0.0) for k in keys}
        with open(self.log_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writerow(filled_row)

    def _save_config(self):
        """Persist basic configuration for later reproducibility/plotting."""
        cfg_path = os.path.join(self.run_dir, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            # Assuming config is a dataclass
            json.dump(asdict(self.config), f, ensure_ascii=False, indent=2)

    def _get_lr(self) -> float:
        """Return current learning rate (assumes single param group)."""
        return self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 0.0
