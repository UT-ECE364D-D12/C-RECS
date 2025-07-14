from typing import Dict, Tuple

import torch
from torch import Tensor

from metrics.base_evaluator import Evaluator
from model.types import Anchor, Negative, Positive


class ContentEvaluator(Evaluator):
    """
    Evaluator for content-based metrics.
    """

    def __init__(self) -> None:
        super().__init__()

        pass

    @torch.no_grad()
    def update(
        self,
        predictions: Tuple[Tensor, Anchor, Positive, Negative],
        rec_targets: Tensor,
    ) -> Dict[str, Tensor]:
        pass

    def calculate(self) -> Dict[str, float]:
        """
        Calculates the content-based metrics.

        Returns:
            Dict[str, float]: Dictionary containing the content-based metrics.
        """
        pass

    def reset(self) -> None:
        """
        Resets the internal state of the evaluator.
        """

        pass
