from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from model.layers import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron
from utils.misc import take_annotation_from


class DeepFM(nn.Module):
    """
    Deep Factorization Machine.

    Reference: H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.

    Args:
        num_items: Number of items
        embed_dim: Dimension of embeddings
        mlp_dims: Dimensions of MLP layers
        dropout: Dropout rate
        weights: Path to model weights
    """

    def __init__(self, num_items: int, embed_dim: int, mlp_dims: List[int], dropout: float, weights: str = None) -> None:
        super().__init__()

        self.num_items = num_items

        self.embedding = FeaturesEmbedding(num_items, embed_dim)
        self.linear = FeaturesLinear(num_items, output_dim=1)
        self.fm = FactorizationMachine()
        self.mlp = MultiLayerPerceptron(
            input_dim=2 * embed_dim,
            hidden_dims=mlp_dims,
            output_dim=1,
            dropout=dropout,
            norm_layer=nn.BatchNorm1d,
        )

        if weights is not None:
            self.load_state_dict(torch.load(weights, weights_only=True))

    def forward(self, features: Tuple[List[Tensor], List[Tensor], Tensor]) -> Tensor:
        """
        Predict item ratings for given user features.

        Args:
            features: User features, item features, and item IDs

        Returns:
            ratings: Predicted ratings
        """

        embeddings = self.embedding(features)

        linear_term = self.linear(features)

        fm_term = self.fm(embeddings)

        mlp_term = self.mlp(embeddings.flatten(1, 2))

        return torch.sigmoid(linear_term + fm_term + mlp_term).squeeze(1)

    @torch.no_grad()
    def predict(self, features: Tuple[Tensor, Tensor], *, mask: bool = True) -> Tensor:
        """
        Predict rating of every item for given user features.

        Args:
            features: User features and ratings
            mask: Whether to mask predictions for items user has already rated

        Returns:
            ratings: Predicted ratings
        """

        feature_ids, feature_ratings = features

        item_ids = torch.arange(self.num_items, device=feature_ids.device)
        feature_ids = feature_ids.repeat(self.num_items).split(len(feature_ids))
        feature_ratings = feature_ratings.repeat(self.num_items).split(len(feature_ratings))

        predicted_ratings = self.forward((feature_ids, feature_ratings, item_ids))

        if mask:
            predicted_ratings[feature_ids[0][1:]] = -1

        return predicted_ratings

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)
