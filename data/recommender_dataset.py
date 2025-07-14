from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


class RatingsRecommendationDataset(Dataset):
    """
    Dataset for predicting ratings, used to train the recommender system.

    Args:
        ratings_path (Path): The path to the ratings parquet file.
    """

    def __init__(self, ratings_path: Path) -> None:
        self.ratings = pd.read_parquet(ratings_path, engine="pyarrow")

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Retrieves a ratings recommendation sample.

        Args:
            idx (int): The index of the sample.

        Returns:
            feature_ids (Tensor): The feature item IDs.
            feature_ratings (Tensor): The feature item ratings.
            item_id (Tensor): The target item ID.
            rating (Tensor): The target item rating.
        """

        feature_ids, feature_ratings, item, rating = self.ratings.iloc[idx][["feature_ids", "feature_ratings", "item_id", "rating"]]

        return (
            torch.tensor(feature_ids, dtype=torch.int64),
            torch.tensor(feature_ratings, dtype=torch.float32),
            torch.tensor(item, dtype=torch.int64),
            torch.tensor(rating, dtype=torch.float32) / 5.0,
        )


def ratings_collate_fn(batch: List[Tuple[Tensor, Tensor, Tensor, Tensor]]) -> Tuple[Tuple[List[Tensor], List[Tensor], Tensor], Tensor]:
    """
    Collates a batch of ratings recommendation data.

    Args:
        batch (List[Tuple[Tensor, Tensor, Tensor, Tensor]]): The batch of data.

    Returns:
        rec_features (Tuple[List[Tensor], List[Tensor], Tensor]): User and item features.
        rec_targets (Tensor): Target ratings
    """
    feature_ids, feature_ratings, target_ids, target_ratings = zip(*batch)

    features = (feature_ids, feature_ratings, torch.stack(target_ids))
    targets = torch.stack(target_ratings)

    return features, targets


class UserRecommendationDataset(Dataset):
    """
    Dataset for user-centric recommendations, used during evaluation to predict ratings for all items for a given user.

    Args:
        ratings_path (Path): Path to the ratings parquet file.
        history_size (float): Fraction of the user's features to use as history for the recommendations.
    """

    def __init__(
        self,
        ratings_path: Path,
        history_size: float = 0.8,
    ) -> None:

        # Load & aggregate the ratings data by user
        ratings = pd.read_parquet(ratings_path, engine="pyarrow")

        # fmt: off
        self.ratings = ratings.groupby("user_id").agg({
            "feature_ids": "last",
            "feature_ratings": "last",
            "item_id": list,
            "rating": list,
            "timestamp": list
        }).reset_index()
        # fmt: on

        self.history_size = history_size

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        feature_ids, feature_ratings, items, ratings = self.ratings.iloc[idx][["feature_ids", "feature_ratings", "item_id", "rating"]]

        # Split into history and target slices
        history_length = int(len(items) * self.history_size)

        feature_ids = feature_ids[: history_length + 1]
        feature_ratings = feature_ratings[: history_length + 1]
        items = items[history_length:]
        ratings = ratings[history_length:]

        return (
            torch.tensor(feature_ids, dtype=torch.int64),
            torch.tensor(feature_ratings, dtype=torch.float32),
            torch.tensor(items, dtype=torch.int64),
            torch.tensor(ratings, dtype=torch.float32) / 5.0,
        )


def user_collate_fn(
    batch: List[Tuple[Tensor, Tensor, Tensor, Tensor]],
) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
    """
    Collates a batch of user recommendation data.

    Args:
        batch: The batch of data.

    Returns:
        features: User history features (IDs and ratings).
        targets: Target items and ratings.
    """

    feature_ids, feature_ratings, target_ids, target_ratings = zip(*batch)

    if len(feature_ids) != 1:
        raise NotImplementedError("User recommendation dataset only supports single user batches.")

    features = (feature_ids[0], feature_ratings[0])
    targets = (target_ids[0], target_ratings[0])

    return features, targets
