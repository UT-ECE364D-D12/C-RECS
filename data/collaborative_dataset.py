import random
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


class CollaborativeDataset(Dataset):
    """
    Dataset for collaborative filtering.

    Args:
        ratings: Ratings dataframe
        requests: Requests dataframe
    """

    def __init__(self, ratings_path: Path, requests_path: Path) -> None:
        self.ratings = pd.read_parquet(ratings_path, engine="pyarrow")

        self.requests = pd.read_csv(requests_path).groupby("item_id").agg({"item_title": "first", "request": list})
        self.requests = self.requests.reset_index().set_index("item_id", drop=False).rename(columns={"request": "requests"})

        self.unique_item_ids = self.ratings["item_id"].unique()
        self.num_items = len(self.requests)

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tuple[str, Tensor], Tensor]:
        """
        Retrieves a collaborative filtering sample.

        Args:
            idx: Index of sample

        Returns:
            feature_ids: Feature item IDs
            feature_ratings: Feature item ratings
            item_id: Target item ID
            rating: Target item rating
            anchor: Anchor request and target item ID
            negative_item_id: Negative item ID
        """

        feature_ids, feature_ratings, item_id, rating = self.ratings.iloc[idx][["feature_ids", "feature_ratings", "item_id", "rating"]]

        anchor_requests = self.requests.loc[item_id]["requests"]

        anchor_request = random.choice(anchor_requests)

        negative_item_id = random.choice([i for i in self.unique_item_ids if i != item_id])

        return (
            torch.tensor(feature_ids, dtype=torch.int64),
            torch.tensor(feature_ratings, dtype=torch.float32),
            torch.tensor(item_id, dtype=torch.int64),
            torch.tensor(rating / 5.0, dtype=torch.float32),
            (anchor_request, torch.tensor(item_id, dtype=torch.int64)),
            torch.tensor(negative_item_id, dtype=torch.int64),
        )


def collaborative_collate_fn(
    batch: List[Tuple[Tensor, Tensor, Tensor, Tensor, Tuple[str, int], int]],
) -> Tuple[Tuple[List[Tensor], List[Tensor], Tensor], Tensor, Tuple[str, Tensor], Tensor]:
    """
    Collates a batch of collaborative data.

    Args:
        batch: Batch of data

    Returns:
        rec_features: User and item features
        rec_targets: Target ratings
        anchors: Anchor request and item ID
        negative_ids: Negative item IDs
    """

    # Unzip the batch
    feature_ids, feature_ratings, target_ids, target_ratings, anchors, negative_ids = zip(*batch)
    anchor_requests, anchor_ids = zip(*anchors)

    # Stack the tensors
    rec_features = (feature_ids, feature_ratings, torch.stack(target_ids))
    rec_targets = torch.stack(target_ratings)
    anchors = (anchor_requests, torch.stack(anchor_ids))
    negative_ids = torch.stack(negative_ids)

    return rec_features, rec_targets, anchors, negative_ids
