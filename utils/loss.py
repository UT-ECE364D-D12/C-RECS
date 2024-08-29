from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Criterion(nn.Module):
    def __init__(self, loss_weights: Dict[str, float] = {}) -> None:
        super().__init__()

        self.loss_weights = loss_weights

    def forward(self, predictions: Tensor, targets: Tensor) -> Dict[str, Tensor]:
        self.update_metrics(predictions, targets)

        mse_loss = F.mse_loss(predictions, targets)

        losses = {"mse": mse_loss}

        losses["overall"] = sum(losses[loss_name] * self.loss_weights.get(loss_name, 1) for loss_name in losses)

        return losses
    
    def __call__(self, predictions: Tensor, targets: Tensor) -> Dict[str, Tensor]:
        return super().__call__(predictions, targets)
     
    # TODO: Implement metrics
    def reset_metrics(self) -> None:
        self.num_samples = torch.tensor(0)

    def update_metrics(self, predictions: Tensor, targets: Tensor) -> None:
        self.num_samples += len(targets)

    def get_metrics(self) -> Dict[str, Tensor]:
        return {}