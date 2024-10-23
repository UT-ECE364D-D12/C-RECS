from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from model.layers import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron


class DeepFM(nn.Module):
    """
    A pytorch implementation of DeepFM.
    Reference: H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """
    def __init__(self, feature_dims: List[int], embed_dim: int, mlp_dims: List[int],  output_dim: int, dropout: float, weights: str = None):
        super().__init__()

        self.embedding = FeaturesEmbedding(feature_dims, embed_dim)
        
        self.linear = FeaturesLinear(feature_dims, output_dim=output_dim)
        
        self.fm = FactorizationMachine()

        self.mlp = MultiLayerPerceptron(input_dim=embed_dim, hidden_dims=mlp_dims, output_dim=output_dim, dropout=dropout)

        if weights is not None:
            self.load_state_dict(torch.load(weights, weights_only=True))

    def forward(self, features: Tuple[Tensor, Tensor]) -> Tensor:
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        user_embeddings = self.embedding(features)
        movie_embeddings = self.embedding.embedding.weight[:-1]

        features = self.linear(features) + self.fm(user_embeddings, movie_embeddings) + self.mlp(user_embeddings)
        
        return torch.sigmoid(features)