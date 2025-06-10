from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


class RatingsDataset(Dataset):
    """
    Dataset for ratings prediction, used to train the recommender system.

    Args:
        ratings (pd.DataFrame): The ratings dataframe.
    """

    def __init__(self, ratings_path: Path) -> None:
        self.ratings = pd.read_parquet(ratings_path, engine="pyarrow")

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Retrieves a ratings sample.

        Args:
            idx (int): The index of the sample.

        Returns:
            feature_ids (Tensor): The feature item IDs.
            feature_ratings (Tensor): The feature item ratings.
            item_id (Tensor): The target item ID.
            rating (Tensor): The target item rating.
        """

        feature_ids, feature_ratings, item_id, rating = self.ratings.iloc[idx][["feature_ids", "feature_ratings", "item_id", "rating"]]

        return (
            torch.tensor(feature_ids, dtype=torch.int64),
            torch.tensor(feature_ratings, dtype=torch.float32),
            torch.tensor(item_id, dtype=torch.int64),
            torch.tensor(rating / 5.0, dtype=torch.float32),
        )


def ratings_collate_fn(batch: List[Tuple[Tensor, Tensor, Tensor, Tensor]]) -> Tuple[Tuple[List[Tensor], List[Tensor], Tensor], Tensor]:
    """
    Collates a batch of ratings data.

    Args:
        batch (List[Tuple[Tensor, Tensor, Tensor, Tensor]]): The batch of data.

    Returns:
        rec_features (Tuple[List[Tensor], List[Tensor], Tensor]): User and item features.
        rec_targets (Tensor): Target ratings
    """
    feature_ids, feature_ratings, target_ids, target_ratings = zip(*batch)

    rec_features = (feature_ids, feature_ratings, torch.stack(target_ids))
    rec_targets = torch.stack(target_ratings)

    return rec_features, rec_targets
