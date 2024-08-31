from typing import List

import torch.nn.functional as F
from torch import Tensor, nn

from model.layers import MultiLayerPerceptron


class RequestDecoder(nn.Module):
    def __init__(self, num_movies: int, embed_dim: int = 768, mlp_dims: List[int] = [32, 32], dropout: float = 0.5):
        super().__init__()
        self.mlp = MultiLayerPerceptron(embed_dim, mlp_dims, dropout, output_dim=num_movies)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``

        :return: Float tensor of size ``(batch_size, num_movies)``
        """
        return self.mlp(x)