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
    def __init__(self, num_items: int, embed_dim: int, mlp_dims: List[int], dropout: float, weights: str = None):
        super().__init__()

        self.embedding = FeaturesEmbedding(num_items, embed_dim)
        
        self.linear = FeaturesLinear(num_items, output_dim=1)
        
        self.fm = FactorizationMachine()

        self.mlp = MultiLayerPerceptron(input_dim=2 * embed_dim, hidden_dims=mlp_dims, output_dim=1, dropout=dropout)

        if weights is not None:
            self.load_state_dict(torch.load(weights, weights_only=True))

    def forward(self, features: Tuple[List[Tensor], List[Tensor], Tensor]) -> Tensor:
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embeddings = self.embedding(features)

        linear_term = self.linear(features)

        fm_term = self.fm(embeddings)

        mlp_term = self.mlp(embeddings.flatten(1, 2))

        return torch.sigmoid(linear_term + fm_term + mlp_term).squeeze(1)