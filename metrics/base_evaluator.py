from typing import Dict

import torch
from torch import nn

from utils.misc import take_annotation_from


class Evaluator(nn.Module):
    """
    Defines the interface for all custom metrics.

    Samples can be added with the `update` method and the metrics can be calculated with the `calculate` method.
    Alternatively, the `forward` method can be used as a shortcut to calculate metrics for a single batch of data.

    The `reset` method should be called before starting to collect samples for a new evaluation run.
    """

    def __init__(self) -> None:
        super().__init__()

        self.reset()

    def reset(self) -> None:
        """
        Resets the internal state of the evaluator.
        """

        raise NotImplementedError

    @torch.no_grad()
    def update(self, *args) -> None:
        """
        Updates the internal state of the evaluator with new samples.
        """

        raise NotImplementedError

    @torch.no_grad()
    def calculate(self) -> Dict[str, float]:
        """
        Calculates the metrics and returns them as a dictionary.
        """

        raise NotImplementedError

    def forward(self, *args) -> Dict[str, float]:
        """
        Shortcut for calculating metrics for a batch. Clears the internal state.
        """

        self.reset()
        self.update(*args)
        metrics = self.calculate(*args)
        self.reset()

        return metrics

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)
