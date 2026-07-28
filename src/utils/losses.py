"""
Loss functions for AFGNN.

Includes:
- Weighted binary cross-entropy (handles class imbalance).
- Focal loss (down-weights easy examples, focuses on hard negatives).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for binary classification.

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Args:
        alpha: weighting factor for the rare class (positive class).
               If None, no alpha weighting is applied.
        gamma: focusing parameter (gamma=0 is standard BCE).
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: raw model outputs, shape (N,).
            targets: binary targets {0, 1}, shape (N,).
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)

        # Focal weight
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_weight = focal_weight * alpha_t

        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """Compute inverse-frequency weights for binary classes.

    Args:
        labels: tensor of shape (N,) with values 0 or 1.

    Returns:
        weights: tensor of shape (2,) for [class_0, class_1].
    """
    num_pos = (labels == 1).sum().float()
    num_neg = (labels == 0).sum().float()

    # Avoid division by zero
    num_pos = max(num_pos, 1.0)
    num_neg = max(num_neg, 1.0)

    total = num_pos + num_neg
    weight_pos = total / (2.0 * num_pos)
    weight_neg = total / (2.0 * num_neg)

    return torch.tensor([weight_neg, weight_pos], dtype=torch.float)
