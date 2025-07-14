from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics.functional import pairwise_cosine_similarity

from criterion.base_criterion import Criterion
from model.layers import MultiLayerPerceptron
from model.types import Anchor, Negative, Positive


class EncoderCriterion(Criterion):
    """
    Encoder criterion that computes the triplet, focal, variance, invariance, and covariance losses.

    Args:
        expander (MultiLayerPerceptron): Expander to increase the dimensionality of the embeddings.
        loss_weights (Dict[str, float]): Weights for each loss term, optional
        triplet_margin (float): Margin for the triplet loss, optional
        triplet_scale (float): Scaling factor for the triplet loss, optional
        focal_gamma (float): Gamma parameter for the focal loss, optional
        vicreg_gamma (float): Gamma parameter for the variance loss, optional
    """

    def __init__(
        self,
        expander: MultiLayerPerceptron,
        loss_weights: Dict[str, float] = {},
        triplet_margin: float = 1.0,
        triplet_scale: float = 20.0,
        focal_gamma: float = 0.0,
        vicreg_gamma: float = 1.0,
    ) -> None:
        super().__init__()

        self.expander = expander
        self.triplet_margin = triplet_margin
        self.triplet_scale = triplet_scale
        self.focal_gamma = focal_gamma
        self.vicreg_gamma = vicreg_gamma
        self.loss_weights = loss_weights

    def forward(self, anchor: Anchor, positive: Positive, negative: Negative) -> Dict[str, Tensor]:
        # Unpack the triplets
        anchor_embeddings, anchor_logits, anchor_ids = anchor
        positive_embeddings, positive_logits, positive_ids = positive
        negative_embeddings, negative_logits, negative_ids = negative

        # Get the triplet loss
        triplet_loss = self._get_triplet_loss(anchor, positive, negative)

        # Get the id loss
        id_loss = (
            self._get_focal_loss(anchor_logits, anchor_ids)
            + self._get_focal_loss(positive_logits, positive_ids)
            + self._get_focal_loss(negative_logits, negative_ids)
        ) / 3

        # Expand the embeddings to a higher dimensionality for VICReg
        anchor_embeddings, positive_embeddings, negative_embeddings = (
            self.expander(anchor_embeddings),
            self.expander(positive_embeddings),
            self.expander(negative_embeddings),
        )

        # Compute the variance, invariance, and covariance losses
        variance_loss = (
            self._get_variance_loss(anchor_embeddings)
            + self._get_variance_loss(positive_embeddings)
            + self._get_variance_loss(negative_embeddings)
        ) / 3

        invariance_loss = self._get_invariance_loss(anchor_embeddings, positive_embeddings)

        covariance_loss = (
            self._get_covariance_loss(anchor_embeddings)
            + self._get_covariance_loss(positive_embeddings)
            + self._get_covariance_loss(negative_embeddings)
        ) / 3

        # Combine all losses into a dictionary
        losses = {
            "triplet": triplet_loss,
            "id": id_loss,
            "variance": variance_loss,
            "invariance": invariance_loss,
            "covariance": covariance_loss,
        }

        # The overall loss is a weighted sum of all losses
        losses["overall"] = sum(losses[loss_name] * self.loss_weights.get(loss_name, 1) for loss_name in losses)

        return losses

    def _get_triplet_loss(self, anchor: Anchor, positive: Positive, negative: Negative) -> Tensor:
        """
        Returns the triplet loss. Pulls the anchor and positive closer while pushing the negative away.
        """

        # Unpack
        anchor_embeddings, _, anchor_ids = anchor
        positive_embeddings, _, positive_ids = positive
        negative_embeddings, _, negative_ids = negative

        # Create the gallery (positives and negatives)
        gallery = torch.cat([positive_embeddings, negative_embeddings], dim=0)
        gallery_ids = torch.cat([positive_ids, negative_ids], dim=0)

        # For every anchor, we compute the similarity to all of the gallery items
        similarities = pairwise_cosine_similarity(anchor_embeddings, gallery) * self.triplet_scale

        # The positive for the ith anchor is the ith candidate, so the label for anchors[i] is i
        target_labels = torch.arange((batch_size := len(anchor_ids)), device=(device := similarities.device))

        # Create a mask to ignore duplicate pairs
        expanded_anchor_ids = anchor_ids.unsqueeze(1).expand(-1, 2 * batch_size)
        expanded_gallery_ids = gallery_ids.unsqueeze(0).expand(batch_size, -1)
        diagonal_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        valid_mask = (expanded_anchor_ids != expanded_gallery_ids) | diagonal_mask[:batch_size]

        return masked_cross_entropy_loss(similarities, target_labels, valid_mask)

    def _get_focal_loss(self, prediction_logits: Tensor, target_labels: Tensor) -> Tensor:
        """
        Returns the focal loss. Similar to cross-entropy loss but with a focus on hard examples.
        """
        prediction_probabilities = prediction_logits.softmax(dim=-1)

        log_probabilities = torch.log(prediction_probabilities.clamp_min(1e-7))

        loss_ce = F.nll_loss(log_probabilities, target_labels, reduction="none")

        probability_target = prediction_probabilities[torch.arange(len(prediction_probabilities)), target_labels]

        focal_term = (1 - probability_target) ** self.focal_gamma

        focal_loss = (focal_term * loss_ce).mean()

        return focal_loss

    def _get_variance_loss(self, embedding: Tensor) -> Tensor:
        """
        Returns the variance loss. Pushes the embeddings to have high variance across the batch dimension.
        """
        embedding = embedding - embedding.mean(dim=0)

        std = embedding.std(dim=0)

        var_loss = F.relu(self.vicreg_gamma - std).mean()

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


def masked_cross_entropy_loss(logits: Tensor, targets: Tensor, valid_mask: Tensor) -> Tensor:
    """
    Returns the cross-entropy loss of the input while ignoring masked-out logits.

    Args:
        logits (Tensor): Logits of the model.
        targets (Tensor): Target labels.
        valid_mask (Tensor): Mask to ignore invalid logits.
    """
    # Mask out invalid logits so they don't affect the maximum value
    logits = logits.masked_fill(~valid_mask, -1e9)

    # Subtract the maximum value for numerical stability
    logits = logits - logits.max(dim=-1, keepdim=True).values

    # Softmax
    exp_logits = torch.exp(logits) * valid_mask

    probs = exp_logits / (exp_logits.sum(dim=-1, keepdim=True))

    # Negative log-likelihood of the targets
    log_probs = torch.log(probs.clamp_min(1e-7))

    return F.nll_loss(log_probs, targets)
