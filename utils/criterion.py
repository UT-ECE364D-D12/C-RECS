from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchmetrics.functional import pairwise_cosine_similarity

from model.layers import MultiLayerPerceptron
from utils.misc import take_annotation_from


class Criterion(nn.Module):
    """
    Parent criterion class that defines the interface for all custom loss functions.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args) -> Dict[str, Tensor]:
        raise NotImplementedError

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)

    def _check_for_nans(self, losses: Dict[str, Tensor]) -> None:
        nan_losses = [name for name, loss in losses.items() if torch.isnan(loss).any()]

        if nan_losses:
            raise ValueError(f"NaNs detected in losses: {nan_losses}")


class AgentCriterion(Criterion):
    """
    Agent criterion that computes the loss between the predicted and target actions.

    Args:
        loss_weights (Dict[str, float]): Weights for each loss term, optional
        focal_gamma (float): Gamma parameter for the focal loss, optional
        action_frequencies (Tensor): Frequencies of each action, optional
        num_actions (int): Number of actions, optional
    """

    def __init__(
        self,
        loss_weights: Dict[str, float] = {},
        focal_gamma: float = 0.0,
        action_frequencies: Tensor = None,
        num_actions: int = 2,
    ) -> None:
        super().__init__()

        self.loss_weights = loss_weights
        self.focal_gamma = focal_gamma
        self.num_actions = num_actions

        self.action_weights = self._calculate_action_weights(action_frequencies, num_actions)

        self._focal_loss = lambda *args: focal_loss(*args, focal_gamma=focal_gamma, weight=self.action_weights)

    def forward(self, predictions: Tensor, targets: Tensor) -> Dict[str, Tensor]:
        """
        Compute the focal loss between the predictions and the targets.

        Args:
            predictions (Tensor): Predicted actions.
            targets (Tensor): Target actions.

        Returns:
            Dict[str, Tensor]: Dictionary of losses ('action', 'overall').
        """

        focal_loss = self._focal_loss(predictions, targets)

        losses = {"action": focal_loss}

        losses["overall"] = sum(losses[loss_name] * self.loss_weights.get(loss_name, 1) for loss_name in losses)

        return losses

    def _calculate_action_weights(self, class_frequencies: Tensor, num_actions: int) -> Tensor:
        """
        Calculate the class weights for the focal loss based on the frequency of each action.

        Args:
            class_frequencies (Tensor): Frequencies of each action.
            num_actions (int): Number of actions.

        Returns:
            Tensor: Class weights for the focal loss.
        """

        class_weights = torch.ones(num_actions, dtype=torch.float32)

        class_weights[class_frequencies > 0] = class_frequencies.sum() / (num_actions * class_frequencies[class_frequencies > 0])

        return class_weights


class RecommenderCriterion(Criterion):
    """
    Recommender criterion that computes the Mean Squared Error (MSE) loss.

    Args:
        loss_weights (Dict[str, float]): Weights for each loss term, optional
    """

    def __init__(self, loss_weights: Dict[str, float] = {}) -> None:
        super().__init__()

        self.loss_weights = loss_weights

    def forward(self, predictions: Tensor, targets: Tensor) -> Dict[str, Tensor]:
        """
        Compute the MSE loss between the predictions and the targets.

        Args:
            predictions (Tensor): Predicted ratings.
            targets (Tensor): Target ratings.

        Returns:
            Dict[str, Tensor]: Dictionary of losses ('mse', 'overall').
        """

        mse_loss = F.mse_loss(predictions, targets)

        losses = {"mse": mse_loss}

        losses["overall"] = sum(losses[loss_name] * self.loss_weights.get(loss_name, 1) for loss_name in losses)

        return losses


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

        self._focal_loss = lambda *args: focal_loss(*args, focal_gamma=focal_gamma)

    def forward(
        self,
        anchor: Tuple[Tensor, Tensor, Tensor],
        positive: Tuple[Tensor, Tensor, Tensor],
        negative: Tuple[Tensor, Tensor, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Computes the losses for the triplets generated by the encoder.

        Args:
            anchor (Tuple[Tensor, Tensor, Tensor]): Anchor embeddings, logits, and IDs.
            positive (Tuple[Tensor, Tensor, Tensor]): Positive embeddings, logits, and IDs.
            negative (Tuple[Tensor, Tensor, Tensor]): Negative embeddings, logits, and IDs.

        Returns:
            Dict[str, Tensor]: Dictionary of losses ('triplet', 'id', 'variance', 'invariance', 'covariance', 'overall').
        """

        # Unpack the anchor, positive, and negative tuples
        anchor_embeddings, anchor_logits, anchor_ids = anchor
        positive_embeddings, positive_logits, positive_ids = positive
        negative_embeddings, negative_logits, negative_ids = negative

        # Move the tensors to the same device
        anchor_ids = anchor_ids.to(device := anchor_embeddings.device)
        positive_ids = positive_ids.to(device)
        negative_ids = negative_ids.to(device)

        # Check for NaNs in the embeddings and logits, indicating numerical instability
        if torch.isnan(torch.stack([anchor_embeddings, positive_embeddings, negative_embeddings])).any():
            raise ValueError("NaNs detected in embeddings")

        if torch.isnan(torch.stack([anchor_logits, positive_logits, negative_logits])).any():
            raise ValueError("NaNs detected in logits")

        # Compute the triplet loss
        triplet_loss = self._get_triplet_loss(
            (anchor_embeddings, anchor_ids),
            (positive_embeddings, positive_ids),
            (negative_embeddings, negative_ids),
        )

        # Compute the focal loss
        id_loss = (
            self._focal_loss(anchor_logits, anchor_ids)
            + self._focal_loss(positive_logits, positive_ids)
            + self._focal_loss(negative_logits, negative_ids)
        ) / 3

        # Increase the dimensionality of the embeddings for VICReg
        anchor_embeddings = self.expander(anchor_embeddings)
        positive_embeddings = self.expander(positive_embeddings)
        negative_embeddings = self.expander(negative_embeddings)

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

        # Calculate the overall loss
        losses = {
            "triplet": triplet_loss,
            "id": id_loss,
            "variance": variance_loss,
            "invariance": invariance_loss,
            "covariance": covariance_loss,
        }

        losses["overall"] = sum(losses[loss_name] * self.loss_weights.get(loss_name, 1) for loss_name in losses)

        return losses

    def _get_triplet_loss(self, anchor: Tuple[Tensor, Tensor], positive: Tuple[Tensor, Tensor], negative: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Pulls the anchor and positive closer while pushing the negative(s) away. An implementation of the Multiple Negative Ranking Loss,
        which is a variant of triplet loss that treats all other samples in the batch as negatives. Modified to ignore in-batch duplicates.

        Args:
            anchor (Tuple[Tensor, Tensor]): Anchor embeddings and IDs.
            positive (Tuple[Tensor, Tensor]): Positive embeddings and IDs.
            negative (Tuple[Tensor, Tensor]): Negative embeddings and IDs.

        Returns:
            triplet_loss (Tensor): Triplet loss.
        """

        # Unpack the triplets
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
        diagonal_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)[:batch_size]
        valid_mask = (expanded_anchor_ids != expanded_gallery_ids) | diagonal_mask

        return masked_cross_entropy_loss(similarities, target_labels, valid_mask)

    def _get_variance_loss(self, x: Tensor) -> Tensor:
        """
        Forces the model to learn diverse representations by pushing the embeddings to have high variance across the batch dimension.

        Args:
            x (Tensor): The embeddings.

        Returns:
            var_loss (Tensor): Variance loss.
        """

        x = x - x.mean(dim=0)

        std = x.std(dim=0)

        var_loss = F.relu(self.vicreg_gamma - std).mean()

        return var_loss

    def _get_invariance_loss(self, anchor: Tensor, positive: Tensor) -> Tensor:
        """
        Forces the representations of the same object to be similar.

        Args:
            anchor (Tensor): Anchor embeddings.
            positive (Tensor): Positive embeddings.

        Returns:
            invariance_loss (Tensor): Invariance loss.
        """

        return F.mse_loss(anchor, positive)

    def _get_covariance_loss(self, x: Tensor) -> Tensor:
        """
        Decorrelate the embeddings' dimensions, pushing the model to capture more information per dimension.

        Args:
            x (Tensor): The embeddings.

        Returns:
            cov_loss (Tensor): Covariance loss.
        """

        x = x - x.mean(dim=0)

        cov = (x.T @ x) / (x.shape[0] - 1)

        cov_loss = cov.fill_diagonal_(0.0).pow(2).sum() / x.shape[1]

        return cov_loss


class CollaborativeCriterion(Criterion):
    """
    A joint criterion that combines the agent, recommender, and encoder criteria, used during multi-turn collaborative training.

    Args:
        loss_weights (Dict[str, float]): Weights for each loss term, optional
        **kwargs: Additional arguments for the Agent, Recommender, and Encoder criteria.
    """

    def __init__(self, loss_weights: Dict[str, float] = {}, **kwargs) -> None:
        super().__init__()

        self.loss_weights = loss_weights

        self.agent_criterion = AgentCriterion(**kwargs["agent"])

        self.recommender_criterion = RecommenderCriterion()

        self.encoder_criterion = EncoderCriterion(**kwargs["encoder"])

    def forward(
        self,
        action_predictions: Tensor,
        action_targets: Tensor,
        rec_predictions: Tensor,
        rec_targets: Tensor,
        anchor: Tuple[Tensor, Tensor, Tensor],
        positive: Tuple[Tensor, Tensor, Tensor],
        negative: Tuple[Tensor, Tensor, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Compute the losses for the agent, recommender, and encoder.

        Args:
            action_predictions (Tensor): Predicted actions.
            action_targets (Tensor): Target actions.
            rec_predictions (Tensor): Predicted ratings.
            rec_targets (Tensor): Target ratings.
            anchor (Tuple[Tensor, Tensor, Tensor]): Anchor embeddings, logits, and IDs.
            positive (Tuple[Tensor, Tensor, Tensor]): Positive embeddings, logits, and IDs.
            negative (Tuple[Tensor, Tensor, Tensor]): Negative embeddings, logits, and IDs.

        Returns:
            Dict[str, Tensor]: Dictionary of losses ('action', 'mse', 'triplet', 'id', 'variance', 'invariance', 'covariance', 'overall').
        """

        agent_losses = self.agent_criterion(action_predictions, action_targets)
        del agent_losses["overall"]

        recommender_losses = self.recommender_criterion(rec_predictions, rec_targets)
        del recommender_losses["overall"]

        encoder_losses = self.encoder_criterion(anchor, positive, negative)
        del encoder_losses["overall"]

        losses = {**agent_losses, **recommender_losses, **encoder_losses}

        self._check_for_nans(losses)

        losses["overall"] = sum(losses[loss_name] * self.loss_weights.get(loss_name, 1) for loss_name in losses)

        return losses


def focal_loss(prediction_logits: Tensor, target_labels: Tensor, focal_gamma: float = 0.0, weight: Tensor = None) -> Tensor:
    """
    Returns the focal loss. Similar to cross-entropy loss but with a focus on hard examples.

    Args:
        prediction_logits (Tensor): Logits of the model.
        target_labels (Tensor): Target labels.
        focal_gamma (float): Gamma parameter for the focal loss, optional
        weight (Tensor): Weight for each class, optional

    Returns:
        focal_loss (Tensor): Focal loss.
    """

    prediction_probabilities = prediction_logits.softmax(dim=-1)

    log_probabilities = torch.log(prediction_probabilities.clamp_min(1e-7))

    loss_ce = F.nll_loss(log_probabilities, target_labels, weight=weight, reduction="none")

    probability_target = prediction_probabilities[torch.arange(len(prediction_probabilities)), target_labels]

    focal_term = (1 - probability_target) ** focal_gamma

    focal_loss = (focal_term * loss_ce).mean()

    return focal_loss


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
