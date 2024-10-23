from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims: List[int], output_dim: int):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))

    def forward(self, features: Tensor) -> Tensor:
        """
        :param features: Float tensor of size ``(batch_size, 2, num_features)``
        """
        feature_ids, feature_ratings = features

        num_features = [len(item_ids) for item_ids in feature_ids]

        # Flatten, embed, and scale (sum(num_features), output_dim)
        linear_embeddings: Tensor = self.fc(torch.cat(feature_ids)) * torch.cat(feature_ratings).unsqueeze(-1)

        # Aggregate the scaled weights per user (batch_size, output_dim)
        linear_embeddings = torch.stack([torch.sum(linear_embedding, dim=0) for linear_embedding in linear_embeddings.split(num_features)])

        return linear_embeddings + self.bias

class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()

        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, features: Tensor) -> Tensor:
        """
        :param features: Float tensor of size ``(batch_size, 2, num_features)``
        """
        feature_ids, feature_ratings = features

        num_features = [len(item_ids) for item_ids in feature_ids]

        # Flatten, embed, and scale (sum(num_features), embed_dim)
        embeddings: Tensor = self.embedding(torch.cat(feature_ids)) * torch.cat(feature_ratings).unsqueeze(-1)

        # Aggregate (batch_size, embed_dim)
        embeddings = torch.stack([torch.sum(embedding, dim=0) for embedding in embeddings.split(num_features)])

        return embeddings


class FactorizationMachine(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, user_embeddings: Tensor, movie_embeddings: Tensor) -> Tensor:
        """
        :param user_embeds: Float tensor of size ``(batch_size, embed_dim)``
        :param movie_embeds: Float tensor of size ``(batch_size, embed_dim)``
        :return: Interaction terms of shape ``(batch_size, num_movies)``
        """
        batch_size, num_movies, embed_dim = user_embeddings.shape[0], *movie_embeddings.shape

        # Square of the sum: Interact user embeddings with each movie embedding
        square_of_sum = torch.matmul(user_embeddings, movie_embeddings.T) ** 2

        # Sum of squares: Sum over the embedding dimension of the element-wise product of user embeddings and movie embeddings
        user_embeddings = user_embeddings.unsqueeze(1).expand(batch_size, num_movies, embed_dim)
        movie_embeddings = movie_embeddings.unsqueeze(0)

        sum_of_square = torch.sum((user_embeddings * movie_embeddings) ** 2, dim=-1)

        # FM Interaction
        interaction = 0.5 * (square_of_sum - sum_of_square)

        return interaction



class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers = list()
        for dim in hidden_dims:
            layers.append(torch.nn.Linear(input_dim, dim))
            layers.append(torch.nn.BatchNorm1d(dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = dim
        layers.append(torch.nn.Linear(input_dim, output_dim))
        
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
