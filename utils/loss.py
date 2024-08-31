from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn


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
    
class RequestCriterion(Criterion):
    def __init__(self, loss_weights: Dict[str, float] = {}) -> None:
        super().__init__()

        self.loss_weights = loss_weights

    def forward(self, predictions: Tensor, targets: Tensor) -> Dict[str, Tensor]:
        ce_loss = F.cross_entropy(predictions, targets)

        losses = {"ce": ce_loss}

        losses["overall"] = sum(losses[loss_name] * self.loss_weights.get(loss_name, 1) for loss_name in losses)

        return losses