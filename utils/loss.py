from typing import Dict, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchmetrics.functional import pairwise_cosine_similarity

from model.layers import MultiLayerPerceptron


class Criterion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, *args) -> Dict[str, Tensor]:
        raise NotImplementedError
    
    def __call__(self, *args) -> Dict[str, Tensor]:
        return super().__call__(*args)

    def _check_for_nans(self, losses: Dict[str, Tensor]) -> None:
        nan_losses = [name for name, loss in losses.items() if torch.isnan(loss).any()]

        if nan_losses:
            raise ValueError(f"NaNs detected in losses: {nan_losses}")

class RecommenderCriterion(Criterion):
    def __init__(self, loss_weights: Dict[str, float] = {}) -> None:
        super().__init__()

        self.loss_weights = loss_weights

    def forward(self, predictions: Tensor, targets: Tensor) -> Dict[str, Tensor]:
        mse_loss = F.mse_loss(predictions, targets)

        losses = {"mse": mse_loss}

        losses["overall"] = sum(losses[loss_name] * self.loss_weights.get(loss_name, 1) for loss_name in losses)

        return losses
    
class EncoderCriterion(Criterion):
    def __init__(self, expander: MultiLayerPerceptron, loss_weights: Dict[str, float] = {}, triplet_margin: float = 1.0, triplet_scale: float = 20.0, focal_gamma: float = 0.0, vicreg_gamma: float = 1.0) -> None:
        super().__init__()

        self.expander = expander
        self.triplet_margin = triplet_margin
        self.triplet_scale = triplet_scale
        self.focal_gamma = focal_gamma
        self.vicreg_gamma = vicreg_gamma
        self.loss_weights = loss_weights
    
    def forward(self, anchor: Tuple[Tensor, Tensor, Tensor], positive: Tuple[Tensor, Tensor, Tensor], negative: Tuple[Tensor, Tensor, Tensor]) -> Dict[str, Tensor]:
        anchor_embeddings, anchor_logits, anchor_ids = anchor
        positive_embeddings, positive_logits, positive_ids = positive
        negative_embeddings, negative_logits, negative_ids = negative

        anchor_ids, positive_ids, negative_ids = anchor_ids.to(device := anchor_embeddings.device), positive_ids.to(device), negative_ids.to(device)

        if torch.isnan(torch.stack([anchor_embeddings, positive_embeddings, negative_embeddings])).any():
            raise ValueError("NaNs detected in embeddings")

        if torch.isnan(torch.stack([anchor_logits, positive_logits, negative_logits])).any():
            raise ValueError("NaNs detected in logits")
        
        triplet_loss = self._get_triplet_loss((anchor_embeddings, anchor_ids), (positive_embeddings, positive_ids), (negative_embeddings, negative_ids))

        id_loss = (self._get_focal_loss(anchor_logits, anchor_ids) + self._get_focal_loss(positive_logits, positive_ids) + self._get_focal_loss(negative_logits, negative_ids)) / 3

        anchor_embeddings, positive_embeddings, negative_embeddings = self.expander(anchor_embeddings), self.expander(positive_embeddings), self.expander(negative_embeddings)

        variance_loss = (self._get_variance_loss(anchor_embeddings) + self._get_variance_loss(positive_embeddings) + self._get_variance_loss(negative_embeddings)) / 3

        invariance_loss = self._get_invariance_loss(anchor_embeddings, positive_embeddings)

        covariance_loss = (self._get_covariance_loss(anchor_embeddings) + self._get_covariance_loss(positive_embeddings) + self._get_covariance_loss(negative_embeddings)) / 3

        losses = {
            "triplet": triplet_loss,
            "id": id_loss,
            "variance": variance_loss,
            "invariance": invariance_loss,
            "covariance": covariance_loss
        }

        losses["overall"] = sum(losses[loss_name] * self.loss_weights.get(loss_name, 1) for loss_name in losses)

        return losses
    
    def _get_triplet_loss(self, anchor: Tuple[Tensor, Tensor], positive: Tuple[Tensor, Tensor], negative: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Returns the triplet loss. Pulls the anchor and positive closer while pushing the negative away.
        """

        anchor_embeddings, anchor_ids = anchor
        positive_embeddings, positive_ids = positive
        negative_embeddings, negative_ids = negative

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
    
    def _get_variance_loss(self, x: Tensor) -> Tensor:     
        """
        Returns the variance loss. Pushes the embeddings to have high variance across the batch dimension.
        """
        x = x - x.mean(dim=0)

        std = x.std(dim=0)

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

class CollaborativeCriterion(Criterion):
    """
    A joint criterion that combines the recommender and encoder criteria, used during collaborative training.
    """
    def __init__(self, loss_weights: Dict[str, float] = {}, **kwargs) -> None:
        super().__init__()

        self.loss_weights = loss_weights

        self.recommender_criterion = RecommenderCriterion()

        self.encoder_criterion = EncoderCriterion(**kwargs)

    def forward(self, rec_predictions: Tensor, rec_targets: Tensor, anchor: Tuple[Tensor, Tensor, Tensor], positive: Tuple[Tensor, Tensor, Tensor], negative: Tuple[Tensor, Tensor, Tensor]) -> Dict[str, Tensor]:
        recommender_losses = self.recommender_criterion(rec_predictions, rec_targets)
        del recommender_losses["overall"]

        encoder_losses = self.encoder_criterion(anchor, positive, negative)
        del encoder_losses["overall"]

        losses = {**recommender_losses, **encoder_losses}

        self._check_for_nans(losses)

        losses["overall"] = sum(losses[loss_name] * self.loss_weights.get(loss_name, 1) for loss_name in losses)

        return losses


def masked_cross_entropy_loss(logits: Tensor, targets: Tensor, valid_mask: Tensor) -> Tensor:
    """
    Returns the cross-entropy loss of the input while ignoring masked-out logits.
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