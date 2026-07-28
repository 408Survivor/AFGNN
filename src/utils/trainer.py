"""
Training and evaluation loops for AFGNN.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import compute_metrics, find_best_threshold


class EarlyStopping:
    """Early stopping helper."""

    def __init__(self, patience: int = 15, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == "max":
            self.is_better = lambda score, best: score > best
        else:
            self.is_better = lambda score, best: score < best

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return self.early_stop


def _compute_loss(logits, targets, criterion):
    """Compute loss given logits, targets, and a criterion."""
    if isinstance(criterion, nn.BCEWithLogitsLoss):
        return criterion(logits.squeeze(-1), targets.float())
    else:
        # FocalLoss or custom losses expect (N,) logits and (N,) targets
        return criterion(logits.squeeze(-1), targets.float())


def train_epoch(model, loader, optimizer, device, criterion, grad_clip=None):
    """Run one training epoch.

    Args:
        grad_clip: if not None, clip gradient norm to this value.

    Returns:
        avg_loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        logit = model(data)  # (B, 1)
        loss = _compute_loss(logit, data.y, criterion)

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, criterion=None, threshold: float = 0.5):
    """Evaluate model on a dataset.

    Args:
        threshold: classification threshold for binary predictions.

    Returns:
        dict with metrics and optionally loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_labels = []
    all_probs = []

    for data in loader:
        data = data.to(device)
        logit = model(data)

        if criterion is not None:
            loss = _compute_loss(logit, data.y, criterion)
            total_loss += loss.item()
            num_batches += 1

        probs = torch.sigmoid(logit).squeeze(-1).cpu().numpy()
        labels = data.y.cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels)

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    metrics = compute_metrics(all_labels, all_probs, threshold=threshold)
    if criterion is not None:
        metrics["loss"] = total_loss / max(num_batches, 1)

    return metrics


@torch.no_grad()
def get_predictions(model, loader, device):
    """Gather true labels and predicted probabilities from a loader.

    Returns:
        y_true: np.ndarray of shape (N,).
        y_score: np.ndarray of shape (N,).
    """
    model.eval()
    all_labels = []
    all_probs = []

    for data in loader:
        data = data.to(device)
        logit = model(data)
        probs = torch.sigmoid(logit).squeeze(-1).cpu().numpy()
        labels = data.y.cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels)

    y_true = np.concatenate(all_labels)
    y_score = np.concatenate(all_probs)
    return y_true, y_score


def train_model(
    model,
    train_loader,
    valid_loader,
    optimizer,
    device,
    criterion,
    num_epochs: int = 100,
    patience: int = 15,
    save_path: str = "experiments/checkpoints/best_model.pt",
    scheduler=None,
    grad_clip: float = None,
    early_stopping_metric: str = "f1",
    config: dict = None,
):
    """Full training loop with validation and early stopping.

    Args:
        scheduler: optional learning rate scheduler.
        grad_clip: optional gradient clipping max norm.
        early_stopping_metric: metric used for early stopping and model selection.
                               Supported: "f1", "auc", "accuracy", "loss".
        config: optional parsed config dict. When given, it is embedded in the
                saved checkpoint so the model can later be rebuilt without the
                original YAML (see utils.builders.build_model).

    Returns:
        history: list of dicts with epoch metrics.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    valid_metrics_for_stop = {"f1": "max", "auc": "max", "accuracy": "max", "loss": "min"}
    if early_stopping_metric not in valid_metrics_for_stop:
        raise ValueError(
            f"Unsupported early_stopping_metric: {early_stopping_metric}. "
            f"Choose from {list(valid_metrics_for_stop.keys())}"
        )

    mode = valid_metrics_for_stop[early_stopping_metric]
    early_stopper = EarlyStopping(patience=patience, mode=mode)
    history = []
    best_score = -float("inf") if mode == "max" else float("inf")

    print(
        f"[Early Stopping] Metric={early_stopping_metric}, mode={mode}, patience={patience}"
    )

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, criterion, grad_clip=grad_clip
        )
        valid_metrics = evaluate(model, valid_loader, device, criterion=criterion)

        valid_metrics["epoch"] = epoch
        valid_metrics["train_loss"] = train_loss
        history.append(valid_metrics)

        current_lr = optimizer.param_groups[0]["lr"]
        score = valid_metrics[early_stopping_metric]

        print(
            f"Epoch {epoch:03d} | "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={valid_metrics.get('loss', -1):.4f} | "
            f"val_acc={valid_metrics['accuracy']:.4f} | "
            f"val_prec={valid_metrics['precision']:.4f} | "
            f"val_rec={valid_metrics['recall']:.4f} | "
            f"val_f1={valid_metrics['f1']:.4f} | "
            f"val_auc={valid_metrics['auc']:.4f}"
        )

        is_better = (mode == "max" and score > best_score) or (
            mode == "min" and score < best_score
        )
        if is_better:
            best_score = score

            # Find the validation threshold that maximizes F1 for the chosen
            # best checkpoint. AUC is used for model selection; threshold tuning
            # gives the operating point for the final F1 report.
            val_labels, val_probs = get_predictions(model, valid_loader, device)
            best_threshold, best_threshold_f1 = find_best_threshold(
                val_labels, val_probs, metric="f1"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_score": best_score,
                    "early_stopping_metric": early_stopping_metric,
                    "best_threshold": best_threshold,
                    "best_threshold_f1": best_threshold_f1,
                    "config": config,
                },
                save_path,
            )
            print(
                f"  -> Saved best model (val_{early_stopping_metric}={best_score:.4f}, "
                f"val_threshold={best_threshold:.2f}, val_threshold_f1={best_threshold_f1:.4f})"
            )

        if scheduler is not None:
            # Support ReduceLROnPlateau and step-based schedulers
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(score)
            else:
                scheduler.step()

        if early_stopper(score):
            print(f"Early stopping at epoch {epoch}")
            break

    return history
