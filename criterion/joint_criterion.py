from typing import Dict, Tuple

from torch import Tensor

from criterion.base_criterion import Criterion
from criterion.encoder_criterion import EncoderCriterion
from criterion.recommender_criterion import RecommenderCriterion
from model.types import Anchor, Negative, Positive


class JointCriterion(Criterion):
    """
    A joint criterion that combines the recommender and encoder criteria.

    Args:
        loss_weights: Weights for each loss term
    """

    def __init__(self, loss_weights: Dict[str, float] = {}, **kwargs) -> None:
        super().__init__()

        self.loss_weights = loss_weights

        self.recommender_criterion = RecommenderCriterion()

        self.encoder_criterion = EncoderCriterion(**kwargs)

    def forward(
        self,
        predictions: Tuple[Tensor, Anchor, Positive, Negative],
        rec_targets: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Compute combined recommender and encoder losses.

        Args:
            predictions: Tuple containing recommendation predictions and triplet data
            rec_targets: Ground truth ratings for recommendations

        Returns:
            losses: Dictionary containing all loss components and overall weighted loss
        """
        rec_predictions, anchor, positive, negative = predictions

        recommender_losses = self.recommender_criterion(rec_predictions, rec_targets)
        del recommender_losses["overall"]

        encoder_losses = self.encoder_criterion(anchor, positive, negative)
        del encoder_losses["overall"]

        losses = {**recommender_losses, **encoder_losses}

        self._check_for_nans(losses)

        losses["overall"] = sum(losses[loss_name] * self.loss_weights.get(loss_name, 1) for loss_name in losses)

        return losses
