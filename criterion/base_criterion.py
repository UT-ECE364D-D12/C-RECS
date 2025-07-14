from typing import Dict

import torch
from torch import Tensor, nn

from utils.misc import take_annotation_from


class Criterion(nn.Module):
    """
    Parent criterion class that defines the interface for all custom loss functions.

    Subclasses are expected to implement the `forward` method.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args) -> Dict[str, Tensor]:
        raise NotImplementedError

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs) -> Dict[str, Tensor]:
        return nn.Module.__call__(self, *args, **kwargs)

    def _check_for_nans(self, losses: Dict[str, Tensor]) -> None:
        nan_losses = [name for name, loss in losses.items() if torch.isnan(loss).any()]

        if nan_losses:
            raise ValueError(f"NaNs detected in losses: {nan_losses}")
