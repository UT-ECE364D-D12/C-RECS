from typing import Dict, List, Tuple, Union

import torch
from sklearn.metrics import ndcg_score
from torch import Tensor

from metrics.base_evaluator import Evaluator
from utils.misc import send_to_device


class RecommenderEvaluator(Evaluator):
    """
    Evaluator for recommendation metrics.

    Args:
        k: List of k values to compute metrics at (i.e. recall@k)
        threshold: The rating threshold (out of 1) to consider an item relevant for recall calculations.
    """

    def __init__(self, k: Union[int, List[int]], threshold: float = 0.5) -> None:

        assert threshold <= 1.0, f"Rating threshold is out of 1.0, received {threshold}"

        self.k = k if isinstance(k, list) else [k]
        self.threshold = threshold

        super().__init__()

    @torch.no_grad()
    def update(
        self,
        predictions: Tensor,
        targets: Tuple[Tensor, Tensor],
    ) -> None:
        predictions, targets = send_to_device((predictions, targets), "cpu")

        item_ids, item_ratings = targets

        true_ratings = torch.zeros_like(predictions)
        true_ratings[item_ids] = item_ratings

        for i, k in enumerate(self.k):
            recall = recall_score(true_ratings, predictions, k=k, threshold=self.threshold)

            if recall is not None:
                self.recall[i] += recall
                self.recall_n[i] += 1

            self.ndcg[i] += ndcg_score([true_ratings], [predictions], k=k)
            self.ndcg_n[i] += 1

    def calculate(self) -> Dict[str, float]:
        """
        Calculates the recommendation metrics.

        Returns:
            metrics: Dictionary containing the recommendation metrics.
        """

        metrics = {}

        for k, recall, n in zip(self.k, self.recall, self.recall_n):
            metrics[f"recall@{k}"] = recall * 100.0 / n if n > 0 else 0.0

        for k, ndcg, n in zip(self.k, self.ndcg, self.ndcg_n):
            metrics[f"ndcg@{k}"] = ndcg * 100.0 / n if n > 0 else 0.0

        return metrics

    def reset(self) -> None:
        """
        Resets the internal state of the evaluator.
        """

        self.recall = [0.0 for _ in self.k]
        self.recall_n = [0 for _ in self.k]

        self.ndcg = [0.0 for _ in self.k]
        self.ndcg_n = [0 for _ in self.k]


def recall_score(true_ratings: Tensor, predicted_ratings: Tensor, k: int, threshold: float) -> float:
    """
    Computes the recall score at k.

    Args:
        true_ratings: True ratings for the items.
        predicted_ratings: Predicted ratings for the items.
        k: The number of top items to consider.
        threshold: The rating threshold to consider an item relevant.

    Returns:
        recall: The recall score at k.
    """

    top_k_items = torch.topk(predicted_ratings, k).indices.cpu().numpy()

    # Calculate recall@k
    relevant_items = torch.where(true_ratings >= threshold)[0].cpu().numpy()

    if len(relevant_items) == 0:
        return None

    recall = len(set(relevant_items) & set(top_k_items)) / len(relevant_items)

    return recall
