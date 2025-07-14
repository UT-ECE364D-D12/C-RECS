import random
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class ContentDataset(Dataset):
    """
    Dataset for ratings & descriptions.

    Args:
        ratings_path: The path to the ratings data.
        descriptions_path: The path to the descriptions data.
    """

    def __init__(self, ratings_path: Path, descriptions_path: Path) -> None:
        self.ratings = pd.read_parquet(ratings_path, engine="pyarrow")

        self.descriptions = pd.read_csv(descriptions_path)

        self.unique_item_ids = self.ratings["item_id"].unique()
        self.num_items = len(self.descriptions)

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tuple[str, Tensor], Tuple[str, Tensor]]:
        """
        Retrieves a rating and description sample.

        Args:
            idx: The index of the sample.

        Returns:
            feature_ids: The feature item IDs.
            feature_ratings: The feature item ratings.
            item_id: The target item ID.
            rating: The target item rating.
            positive: The positive description and item ID.
            negative: The negative description and item ID.
        """

        # Obtain the features & targets for the recommender
        feature_ids, feature_ratings, item_id, rating = self.ratings.iloc[idx][["feature_ids", "feature_ratings", "item_id", "rating"]]

        # Obtain the positive description for the item
        positive_description = self.descriptions.loc[item_id]["description"]

        # Obtain a random description for the negative item
        negative_item_id = random.choice([i for i in self.unique_item_ids if i != item_id])

        negative_description = self.descriptions.loc[negative_item_id]["description"]

        return (
            torch.tensor(feature_ids, dtype=torch.int64),
            torch.tensor(feature_ratings, dtype=torch.float32),
            torch.tensor(item_id, dtype=torch.int64),
            torch.tensor(rating / 5.0, dtype=torch.float32),
            (positive_description, torch.tensor(item_id, dtype=torch.int64)),
            (negative_description, torch.tensor(negative_item_id, dtype=torch.int64)),
        )


def content_collate_fn(
    batch: List[Tuple[Tensor, Tensor, Tensor, Tensor, Tuple[str, Tensor], Tuple[str, Tensor]]],
) -> Tuple[Tuple[Tuple[List[Tensor], List[Tensor], Tensor], Tuple[str, Tensor], Tuple[str, Tensor]], Tensor]:
    """
    Collates a batch of ratings and descriptions data.

    Args:
        batch: The batch of data.

    Returns:
        features:
            rec_features: User features & target item ID.
            positive: Positive description and item ID.
            negative: Negative description and item ID.
        targets:
            rec_targets: Target ratings.
    """

    # Unzip the batch
    feature_ids, feature_ratings, target_ids, target_ratings, positives, negatives = zip(*batch)
    positive_descriptions, positive_ids = zip(*positives)
    negative_descriptions, negative_ids = zip(*negatives)

    # Stack the recommender tensors
    rec_features = (feature_ids, feature_ratings, torch.stack(target_ids))
    rec_targets = torch.stack(target_ratings)

    # Stack the positives and negatives
    positives = (positive_descriptions, torch.stack(positive_ids))
    negatives = (negative_descriptions, torch.stack(negative_ids))

    return (rec_features, positives, negatives), rec_targets


def build_content_dataloaders(root: Path, batch_size: int, num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    """
    Build training and validation datasets for content-based training.

    Args:
        root: Path to the root directory of the dataset.
        batch_size: Batch size for the DataLoader.
        num_workers: Number of worker threads for DataLoader.

    Returns:
        train_loader: DataLoader for training dataset.
        val_loader: DataLoader for validation dataset.
    """

    train_dataset = ContentDataset(root / "train_ratings.parquet", root / "descriptions.csv")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=content_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    # Use the same descriptions for validation because we care about
    # the performance of the recommender system, not the encoder.
    val_dataset = ContentDataset(root / "val_ratings.parquet", root / "descriptions.csv")

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=content_collate_fn,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )

    return train_dataloader, val_dataloader
