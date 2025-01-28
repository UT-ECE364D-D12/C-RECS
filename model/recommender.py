from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from model.layers import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron


class DeepFM(nn.Module):
    """
    Deep Factorization Machine.

    Reference: H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.

    Args:
        num_items (int): Number of items.
        embed_dim (int): Dimension of the embeddings.
        mlp_dims (List[int]): Dimensions of the MLP layers.
        dropout (float): Dropout rate, optional.
        weights (str): Path to the model weights, optional.
    """

    def __init__(self, num_items: int, embed_dim: int, mlp_dims: List[int], dropout: float, weights: str = None) -> None:
        super().__init__()

        self.num_items = num_items

        self.embedding = FeaturesEmbedding(num_items, embed_dim)
        self.linear = FeaturesLinear(num_items, output_dim=1)
        self.fm = FactorizationMachine()
        self.mlp = MultiLayerPerceptron(input_dim=2 * embed_dim, hidden_dims=mlp_dims, output_dim=1, dropout=dropout)

        if weights is not None:
            self.load_state_dict(torch.load(weights, weights_only=True))

    def forward(self, features: Tuple[List[Tensor], List[Tensor], Tensor]) -> Tensor:
        """
        Predict the item ratings for the given user features, used during training.

        Args:
            features (Tuple[List[Tensor], List[Tensor], Tensor]): User features, item features, and item IDs.
        
        Returns:
            ratings (Tensor): Predicted ratings.
        """

        embeddings = self.embedding(features)

        linear_term = self.linear(features)

        fm_term = self.fm(embeddings)

        mlp_term = self.mlp(embeddings.flatten(1, 2))

        return torch.sigmoid(linear_term + fm_term + mlp_term).squeeze(1)

    def predict(self, features: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Predict the rating of every item for the given user features.

        Args:
            features (Tuple[Tensor, Tensor]): User features and ratings.
        
        Returns:
            ratings (Tensor): Predicted ratings.
        """

        feature_ids, feature_ratings = features

        with torch.no_grad():
            item_ids = torch.arange(self.num_items, device=feature_ids.device)
            feature_ids = feature_ids.repeat(self.num_items).split(len(feature_ids))
            feature_ratings = feature_ratings.repeat(self.num_items).split(len(feature_ratings))

            return self.forward((feature_ids, feature_ratings, item_ids))
