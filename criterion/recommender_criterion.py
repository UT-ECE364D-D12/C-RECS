from typing import Dict

import torch.nn.functional as F
from torch import Tensor

from criterion.base_criterion import Criterion


class RecommenderCriterion(Criterion):
    """
    Loss function for rating prediction models.

    Args:
        loss_weights: Weights for each loss term, optional
    """

    def __init__(self, loss_weights: Dict[str, float] = {}) -> None:
        super().__init__()

        self.loss_weights = loss_weights

    def forward(self, predictions: Tensor, targets: Tensor) -> Dict[str, Tensor]:
        """
        Calculate the Mean Squared Error (MSE) loss between predictions and targets.

        Args:
            predictions: Predicted ratings.
            targets: Ground truth ratings.

        Returns:
            losses: Dictionary containing the MSE loss and overall loss.
        """

        # Calculate the Mean Squared Error (MSE) loss
        mse_loss = F.mse_loss(predictions, targets)

        losses = {"mse": mse_loss}

        # Calculate the weighted overall loss
        losses["overall"] = sum(losses[loss_name] * self.loss_weights.get(loss_name, 1) for loss_name in losses)

        return losses
