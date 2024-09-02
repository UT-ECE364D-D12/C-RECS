from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from model.layers import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron


class DeepFM(nn.Module):
    """
    A pytorch implementation of DeepFM.
    Reference: H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """
    def __init__(self, feature_dims: List[int], embed_dim: int, mlp_dims: List[int], dropout: float):
        super().__init__()
        self.linear = FeaturesLinear(feature_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(feature_dims, embed_dim)
        self.embed_output_dim = len(feature_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(input_dim=self.embed_output_dim, hidden_dims=mlp_dims, output_dim=1, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x_embed = self.embedding(x)

        x = self.linear(x) + self.fm(x_embed) + self.mlp(x_embed.view(-1, self.embed_output_dim))
        
        return torch.sigmoid(x.squeeze(1))