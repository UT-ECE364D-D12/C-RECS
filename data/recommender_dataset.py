from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class RecommenderDataset(Dataset):
    """
    Dataset for predicting ratings, used to train the recommender system.

    Args:
        ratings_path: The path to the ratings parquet file.
    """

    def __init__(self, ratings_path: Path) -> None:
        self.ratings = pd.read_parquet(ratings_path, engine="pyarrow")

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Retrieves a ratings recommendation sample.

        Args:
            idx: The index of the sample.

        Returns:
            feature_ids: The feature item IDs.
            feature_ratings: The feature item ratings.
            item_id: The target item ID.
            rating: The target item rating.
        """

        feature_ids, feature_ratings, item, rating = self.ratings.iloc[idx][["feature_ids", "feature_ratings", "item_id", "rating"]]

        return (
            torch.tensor(feature_ids, dtype=torch.int64),
            torch.tensor(feature_ratings, dtype=torch.float32),
            torch.tensor(item, dtype=torch.int64),
            torch.tensor(rating, dtype=torch.float32) / 5.0,
        )


def rec_collate_fn(batch: List[Tuple[Tensor, Tensor, Tensor, Tensor]]) -> Tuple[Tuple[List[Tensor], List[Tensor], Tensor], Tensor]:
    """
    Collates a batch of ratings recommendation data.

    Args:
        batch: The batch of data.

    Returns:
        rec_features: User and item features.
        rec_targets: Target ratings
    """
    feature_ids, feature_ratings, target_ids, target_ratings = zip(*batch)

    features = (feature_ids, feature_ratings, torch.stack(target_ids))
    targets = torch.stack(target_ratings)

    return features, targets


def build_rec_dataloaders(root: Path, batch_size: int, num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    """
    Build training and validation datasets for recommender training.

    Args:
        root: Path to the root directory of the dataset.
        batch_size: Batch size for the DataLoader.
        num_workers: Number of worker threads for DataLoader.

    Returns:
        train_loader: DataLoader for training dataset.
        val_loader: DataLoader for validation dataset.
    """

    train_dataset = RecommenderDataset(root / "train_ratings.parquet")

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=rec_collate_fn,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_dataset = RecommenderDataset(root / "val_ratings.parquet")

    val_dataloader = DataLoader(
        val_dataset,
        collate_fn=rec_collate_fn,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_dataloader, val_dataloader


class RecommenderEvalDataset(Dataset):
    """
    Dataset for user-centric recommendations, used during evaluation to predict ratings for all items for a given user.

    Args:
        ratings_path: Path to the ratings parquet file.
        history_frac: Fraction of the user's features to use as history for the recommendations.
    """

    def __init__(
        self,
        ratings_path: Path,
        history_frac: float = 0.8,
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

        self.history_frac = history_frac

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        feature_ids, feature_ratings, items, ratings = self.ratings.iloc[idx][["feature_ids", "feature_ratings", "item_id", "rating"]]

        # Split into history and target slices
        history_length = int(len(items) * self.history_frac)

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


def rec_eval_collate_fn(
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


def build_rec_eval_dataloader(root: Path, history_frac: float = 0.8, num_workers: int = 0, **_) -> DataLoader:
    """
    Build the evaluation DataLoader for user-centric recommendations.

    Args:
        root: Path to the root directory of the dataset.
        history_frac: Fraction of the user's features to use as history for the recommendations.
        num_workers: Number of worker threads for DataLoader.

    Returns:
        eval_dataloader: DataLoader for evaluation dataset.
    """

    eval_dataset = RecommenderEvalDataset(root / "val_ratings.parquet", history_frac=history_frac)

    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=rec_eval_collate_fn,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )

    return eval_dataloader
