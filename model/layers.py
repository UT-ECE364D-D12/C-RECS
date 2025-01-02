from math import sqrt
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FeaturesEmbedding(nn.Module):

    def __init__(self, num_items: int, embed_dim: int):
        super().__init__()

        self.num_items = num_items

        self.feature_embedding = nn.Embedding(num_items + 1, embed_dim)
        self.rating_embedding = nn.Embedding(10, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)

        self._initialize_weights()

    def forward(self, features: Tuple[List[Tensor], List[Tensor], Tensor]) -> Tensor:
        """
        :param features: (feature_ids, feature_ratings, item_ids)
        """
        feature_ids, feature_ratings, item_ids = features 

        # Embed user features
        num_features = [len(item_ids) for item_ids in feature_ids]

        rating_embeddings = self.rating_embedding(((torch.cat(feature_ratings) - 0.5) * 2).int().clamp(0, 9))

        user_embeddings: Tensor = self.feature_embedding(torch.cat(feature_ids)) * rating_embeddings

        user_embeddings = torch.stack([torch.sum(embedding, dim=0) for embedding in user_embeddings.split(num_features)])

        # Embed items
        item_embeddings = self.item_embedding(item_ids)

        return torch.stack((user_embeddings, item_embeddings), dim=1)

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.feature_embedding.weight.data, a=sqrt(5))
        nn.init.kaiming_uniform_(self.rating_embedding.weight.data, a=sqrt(5))
        nn.init.kaiming_uniform_(self.item_embedding.weight.data, a=sqrt(5))
    
    def __call__(self, *args) -> Tensor:
        return super().__call__(*args)

class FeaturesLinear(nn.Module):

    def __init__(self, num_items: int, output_dim: int):
        super().__init__()
        self.num_items = num_items
        
        self.user_fc = nn.Embedding(num_items + 1, output_dim)
        self.rating_fc = nn.Embedding(10, output_dim)
        self.item_fc = nn.Embedding(num_items, output_dim)

        self.bias = nn.Parameter(torch.zeros((output_dim,)))

        self._initialize_weights()

    def forward(self, features: Tuple[List[Tensor], List[Tensor], Tensor]) -> Tensor:
        """
        :param features: Float tensor of size ``(batch_size, 2, num_features)``
        """
        feature_ids, feature_ratings, item_ids = features 

        # Embed users
        num_features = [len(item_ids) for item_ids in feature_ids]

        rating_weights = self.rating_fc(((torch.cat(feature_ratings) - 0.5) * 2).int())

        user_weights: Tensor = self.user_fc(torch.cat(feature_ids)) * rating_weights

        user_weights = torch.stack([torch.sum(linear_embedding, dim=0) for linear_embedding in user_weights.split(num_features)])

        # Embed items
        item_weights = self.item_fc(item_ids) 

        return user_weights + item_weights + self.bias

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.user_fc.weight.data, a=sqrt(5))
        nn.init.kaiming_uniform_(self.rating_fc.weight.data, a=sqrt(5))
        nn.init.kaiming_uniform_(self.item_fc.weight.data, a=sqrt(5))
        nn.init.zeros_(self.bias.data)

class FactorizationMachine(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, embeddings: Tensor) -> Tensor:
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(embeddings, dim=1) ** 2

        sum_of_square = torch.sum(embeddings ** 2, dim=1)

        interaction = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)

        return interaction


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()

        layers = []

        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))

            input_dim = dim

        layers.append(nn.Linear(input_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

        self._initialize_weights()

    def forward(self, feature_embeddings: Tensor) -> Tensor:
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(feature_embeddings)
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data, a=sqrt(5))
                nn.init.zeros_(m.bias.data)
