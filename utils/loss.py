from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from utils.misc import cosine_distance, pairwise_cosine_distance


class Criterion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Dict[str, Tensor]:
        raise NotImplementedError
    
    def __call__(self, predictions: Tensor, targets: Tensor) -> Dict[str, Tensor]:
        return super().__call__(predictions, targets)

    def reset_metrics(self) -> None:
        return
    
    def update_metrics(self, predictions: Tensor, targets: Tensor) -> None:
        return

    def get_metrics(self) -> Dict[str, Tensor]:
        return
    
class RecommenderCriterion(Criterion):
    def __init__(self, loss_weights: Dict[str, float] = {}) -> None:
        super().__init__()

        self.loss_weights = loss_weights

    def forward(self, predictions: Tensor, targets: Tensor) -> Dict[str, Tensor]:
        mse_loss = F.mse_loss(predictions, targets)

        losses = {"mse": mse_loss}

        losses["overall"] = sum(losses[loss_name] * self.loss_weights.get(loss_name, 1) for loss_name in losses)

        return losses
    
class DecoderCriterion(Criterion):
    def __init__(self, loss_weights: Dict[str, float] = {}) -> None:
        super().__init__()

        self.loss_weights = loss_weights

    def forward(self, predictions: Tensor, targets: Tensor) -> Dict[str, Tensor]:
        ce_loss = F.cross_entropy(predictions, targets)

        losses = {"ce": ce_loss}

        losses["overall"] = sum(losses[loss_name] * self.loss_weights.get(loss_name, 1) for loss_name in losses)

        return losses
    
class EncoderCriterion(Criterion):
    def __init__(self, expander: nn.Module, margin: float = 1.0, gamma: float = 1.0, loss_weights: Dict[str, float] = {}) -> None:
        super().__init__()

        self.expander = expander
        self.margin = margin
        self.gamma = gamma
        self.loss_weights = loss_weights
    
    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Dict[str, Tensor]:
        triplet_loss = self._get_triplet_loss(anchor, positive, negative)

        # Expand the embeddings to ensure VICReg doesn't remove complex interactions
        anchor, positive, negative = self.expander(anchor), self.expander(positive), self.expander(negative)

        variance_loss = (self._get_variance_loss(anchor) + self._get_variance_loss(positive) + self._get_variance_loss(negative)) / 3

        invariance_loss = self._get_invariance_loss(anchor, positive)

        covariance_loss = (self._get_covariance_loss(anchor) + self._get_covariance_loss(positive) + self._get_covariance_loss(negative)) / 3

        losses = {
            "triplet": triplet_loss,
            "variance": variance_loss,
            "invariance": invariance_loss,
            "covariance": covariance_loss
        }

        losses["overall"] = sum(losses[loss_name] * self.loss_weights.get(loss_name, 1) for loss_name in losses)

        return losses

    def _get_triplet_loss(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        distance_ap = cosine_distance(anchor, positive)
        distance_an = cosine_distance(anchor, negative)

        return F.relu(distance_ap - distance_an + self.margin)

    def _get_variance_loss(self, x: Tensor) -> Tensor:     
        """
        Returns the variance loss. Pushes the embeddings to have high variance across the batch dimension.
        """
        x = x - x.mean(dim=0)

        std = x.std(dim=0)

        var_loss = F.relu(self.gamma - std).mean()

        return var_loss
    
    def _get_invariance_loss(self, anchor: Tensor, positive: Tensor) -> Tensor:
        """
        Returns the invariance loss. Forces the representations of the same object to be similar.
        """
        return F.mse_loss(anchor, positive)
    
    def _get_covariance_loss(self, x: Tensor) -> Tensor:
        """
        Returns the covariance loss. Decorrelate the embeddings' dimensions, pushing the model to capture more information per dimension.
        """
        x = x - x.mean(dim=0)

        cov = (x.T @ x) / (x.shape[0] - 1)

        cov_loss = cov.fill_diagonal_(0.0).pow(2).sum() / x.shape[1]

        return cov_loss