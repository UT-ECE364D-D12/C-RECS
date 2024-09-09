from typing import Dict, Tuple, Union

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, recall_score
from torch import Tensor, nn

from utils.misc import cosine_distance, pairwise_cosine_distance


class Criterion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, *args) -> Dict[str, Tensor]:
        raise NotImplementedError
    
    def __call__(self, *args) -> Dict[str, Tensor]:
        return super().__call__(*args)

    def reset_metrics(self) -> None:
        return
    
    def _update_metrics(self, *args) -> None:
        return

    def get_metrics(self) -> Dict[str, Union[int, float]]:
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
    def __init__(self, expander: nn.Module, classifier: nn.Module, margin: float = 1.0, gamma: float = 1.0, loss_weights: Dict[str, float] = {}, max_rank: int = 50) -> None:
        super().__init__()

        self.expander = expander
        self.classifier = classifier
        self.margin = margin
        self.gamma = gamma
        self.loss_weights = loss_weights
        self.max_rank = max_rank
    
    def forward(self, anchor: Tuple[Tensor, Tensor], positive: Tuple[Tensor, Tensor], negative: Tuple[Tensor, Tensor]) -> Dict[str, Tensor]:
        anchor_embeddings, anchor_ids = anchor
        positive_embeddings, positive_ids = positive
        negative_embeddings, negative_ids = negative

        anchor_ids, positive_ids, negative_ids = anchor_ids.to(device := anchor_embeddings.device), positive_ids.to(device), negative_ids.to(device)

        prediction_anchor_logits, prediction_positive_logits, prediction_negative_logits = self.classifier(anchor_embeddings), self.classifier(positive_embeddings), self.classifier(negative_embeddings)
        
        self._update_metrics((anchor_embeddings, prediction_anchor_logits, anchor_ids), (positive_embeddings, prediction_positive_logits, positive_ids), (negative_embeddings, prediction_negative_logits, negative_ids))

        triplet_loss = self._get_triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)

        id_loss = (self._get_id_loss(prediction_anchor_logits, anchor_ids) + self._get_id_loss(prediction_positive_logits, positive_ids) + self._get_id_loss(prediction_negative_logits, negative_ids)) / 3

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

    def reset_metrics(self) -> None:
        self.id_ap = 0.0
        self.id_recall = 0.0
        self.reid_map = 0.0
        self.cmc_curve = torch.zeros(self.max_rank)

        self.num_samples = 0.0
        self.num_valid_queries = 0

    def get_metrics(self) -> Dict[str, Union[int, float]]:
        id_ap = self.id_ap / self.num_samples
        id_recall = self.id_recall / self.num_samples
        reid_map = self.reid_map / self.num_valid_queries
        cmc_curve = self.cmc_curve / self.num_valid_queries

        metrics = {
            "id_ap": id_ap,
            "id_recall": id_recall,
            "reid_map": reid_map,
            "rank_1": cmc_curve[0].item(),
            "rank_5": cmc_curve[4].item(),
            "rank_10": cmc_curve[9].item(),
        }

        return metrics
    
    def _update_metrics(self, anchor: Tuple[Tensor, Tensor, Tensor], positive: Tuple[Tensor, Tensor, Tensor], negative: Tuple[Tensor, Tensor, Tensor]) -> None:
        anchor_embeddings, prediction_anchor_logits, anchor_ids = anchor
        positive_embeddings, prediction_positive_logits, positive_ids = positive
        negative_embeddings, prediction_negative_logits, negative_ids = negative

        prediction_embeddings = torch.cat([anchor_embeddings, positive_embeddings, negative_embeddings])
        prediction_id_logits = torch.cat([prediction_anchor_logits, prediction_positive_logits, prediction_negative_logits])
        target_ids = torch.cat([anchor_ids, positive_ids, negative_ids])

        num_samples = len(prediction_embeddings)
        device = prediction_embeddings.device

        pairwise_distances = pairwise_cosine_distance(prediction_embeddings)
        pairwise_matches = target_ids.unsqueeze(0) == target_ids.unsqueeze(1)

        # Remove diagonal elements so we don't match the same object to itself
        diagonal_mask = ~torch.eye(num_samples, num_samples, dtype=bool)
        pairwise_distances = pairwise_distances[diagonal_mask].reshape(num_samples, num_samples - 1)
        pairwise_matches = pairwise_matches[diagonal_mask].reshape(num_samples, num_samples - 1)

        # Ignore any queries with no matches
        valid_mask = pairwise_matches.any(dim=-1)
        num_valid_queries = valid_mask.sum().item()
        pairwise_distances = pairwise_distances[valid_mask]
        pairwise_matches = pairwise_matches[valid_mask]

        # Sort vectors from closest to farthest
        pairwise_distances, indices = torch.sort(pairwise_distances, dim=-1)
        pairwise_matches = pairwise_matches.gather(-1, indices)

        # Calculate the Average Precision (https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision)
        cumulative_sum = torch.cumsum(pairwise_matches, dim=-1)
        precision = cumulative_sum / torch.arange(1, num_samples, device=device).unsqueeze(0)
        average_precision = (precision * pairwise_matches).sum(dim=-1) / pairwise_matches.sum(dim=-1)
        self.reid_map += average_precision.sum().item()

        # Calculate the Cumulative Matching Characteristics (CMC) curve
        cumulative_sum[cumulative_sum > 1] = 1
        cmc_curve = torch.full((self.max_rank,), num_valid_queries, dtype=torch.float32)
        cmc_curve[:num_samples - 1] = cumulative_sum[:, :self.max_rank].sum(dim=0).cpu()
        self.cmc_curve += cmc_curve

        self.num_valid_queries += num_valid_queries

        # Calculate the Average Precision and Recall for the ID classification task
        prediction_id_probabilities = prediction_id_logits.softmax(dim=-1)

        prediction_id_scores, prediction_id_labels = prediction_id_probabilities.max(dim=-1)

        self.id_ap += average_precision_score(target_ids.detach().cpu().view(-1, 1), prediction_id_scores.detach().cpu().view(-1, 1))

        self.id_recall += recall_score(target_ids.cpu(), prediction_id_labels.cpu(), average="micro")

        self.num_samples += num_samples
    
    def _get_triplet_loss(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        distance_ap = cosine_distance(anchor, positive)
        distance_an = cosine_distance(anchor, negative)

        return F.relu(distance_ap - distance_an + self.margin).mean()
    
    def _get_id_loss(self, prediction_logits: Tensor, target_ids: Tensor) -> Tensor:
        return F.cross_entropy(prediction_logits, target_ids)

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