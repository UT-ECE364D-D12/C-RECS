from typing import Dict, Tuple

import torch
from sklearn.metrics import accuracy_score, average_precision_score, recall_score
from torch import Tensor

from utils.misc import pairwise_cosine_distance


def get_reid_metrics(queries: Tuple[Tensor, Tensor], gallery: Tuple[Tensor, Tensor], device: str = "cpu") -> Dict[str, float]:
    query_embeddings, query_ids = queries
    gallery_embeddings, gallery_ids = gallery 

    num_gallery = len(gallery_embeddings)

    pairwise_distances = pairwise_cosine_distance(query_embeddings, gallery_embeddings)
    pairwise_matches = query_ids.unsqueeze(1) == gallery_ids.unsqueeze(0)

    # Ignore any queries with no matches
    valid_mask = pairwise_matches.any(dim=-1)
    assert valid_mask.all(), "All queries should have at least one match"
    num_valid_queries = valid_mask.sum().item()
    pairwise_distances = pairwise_distances[valid_mask]
    pairwise_matches = pairwise_matches[valid_mask]

    # Sort vectors from closest to farthest
    pairwise_distances, indices = torch.sort(pairwise_distances, dim=-1)
    pairwise_matches = pairwise_matches.gather(-1, indices)

    # Calculate the Average Precision (https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision)
    cumulative_sum = torch.cumsum(pairwise_matches, dim=-1)
    precision = cumulative_sum / torch.arange(1, num_gallery + 1, device=device).unsqueeze(0)
    average_precision = (precision * pairwise_matches).sum(dim=-1) / pairwise_matches.sum(dim=-1)
    reid_map = average_precision.mean()

    # Calculate the Cumulative Matching Characteristics (CMC) curve
    cumulative_sum[cumulative_sum > 1] = 1
    cmc_curve = cumulative_sum.sum(dim=0) / num_valid_queries

    return {
        "reid_map": reid_map.item(),
        "rank-1": cmc_curve[0].item(),
        "rank-5": cmc_curve[4].item(),
        "rank-10": cmc_curve[9].item(),
    }

def get_id_metrics(predictions: Tensor, target_ids: Tensor) -> Dict[str, float]:
    prediction_scores, prediction_ids = predictions[:, 0], predictions[:, 1].int()

    id_accuracy = accuracy_score(target_ids.cpu(), prediction_ids.cpu())

    id_ap = average_precision_score(target_ids.cpu().view(-1, 1), prediction_scores.cpu().view(-1, 1))

    id_recall = recall_score(target_ids.cpu(), prediction_ids.cpu(), average="micro")

    return {
        "id_accuracy": id_accuracy,
        "id_ap": id_ap,
        "id_recall": id_recall,
    }
