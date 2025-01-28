from math import sqrt
from typing import List, Tuple

import torch
from torch import Tensor, nn

from utils.misc import take_annotation_from


class FeaturesEmbedding(nn.Module):
    """
    Embed user features and items.

    Args:
        num_items (int): Number of items.
        embed_dim (int): Embedding dimension.    
    """ 

    def __init__(self, num_items: int, embed_dim: int) -> None:
        super().__init__()

        self.num_items = num_items

        self.feature_embedding = nn.Embedding(num_items + 1, embed_dim)
        self.rating_embedding = nn.Embedding(10, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)

        self._initialize_weights()
    
    def forward(self, features: Tuple[List[Tensor], List[Tensor], Tensor]) -> Tensor:
        """
        Forward pass of the embedding layer.

        Args:
            features (Tuple[List[Tensor], List[Tensor], Tensor]): feature_ids, feature_ratings, item_ids
        
        Returns:
            embeddings (Tensor): User and item embeddings of shape (batch_size, 2, embed_dim).
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

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)

    def _initialize_weights(self):
        """
        Initialize the weights of the embedding layers.
        """

        nn.init.kaiming_uniform_(self.feature_embedding.weight.data, a=sqrt(5))
        nn.init.kaiming_uniform_(self.rating_embedding.weight.data, a=sqrt(5))
        nn.init.kaiming_uniform_(self.item_embedding.weight.data, a=sqrt(5))


class FeaturesLinear(nn.Module):
    """
    Linear layer for user features and items.

    Args:
        num_items (int): Number of items.
        output_dim (int): Output dimension.    
    """

    def __init__(self, num_items: int, output_dim: int) -> None:
        super().__init__()
        self.num_items = num_items

        self.user_fc = nn.Embedding(num_items + 1, output_dim)
        self.rating_fc = nn.Embedding(10, output_dim)
        self.item_fc = nn.Embedding(num_items, output_dim)

        self.bias = nn.Parameter(torch.zeros((output_dim,)))

        self._initialize_weights()

    def forward(self, features: Tuple[List[Tensor], List[Tensor], Tensor]) -> Tensor:
        """
        Forward pass of the linear layer.

        Args:
            features (Tuple[List[Tensor], List[Tensor], Tensor]): feature_ids, feature_ratings, item_ids
        
        Returns:
            weights (Tensor): Combination of the user and item weights of shape (batch_size, output_dim).
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

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)

    def _initialize_weights(self):
        """
        Initialize the weights of the linear layers.
        """

        nn.init.kaiming_uniform_(self.user_fc.weight.data, a=sqrt(5))
        nn.init.kaiming_uniform_(self.rating_fc.weight.data, a=sqrt(5))
        nn.init.kaiming_uniform_(self.item_fc.weight.data, a=sqrt(5))
        nn.init.zeros_(self.bias.data)


class FactorizationMachine(torch.nn.Module):
    """
    Factorization Machine that models interactions between features using latent factors.    
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, embeddings: Tensor) -> Tensor:
        """
        Forward pass of the Factorization Machine layer.

        Args:
            embeddings (Tensor): Float tensor of size (batch_size, num_fields, embed_dim).
        """

        square_of_sum = torch.sum(embeddings, dim=1) ** 2

        sum_of_square = torch.sum(embeddings**2, dim=1)

        interaction = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)

        return interaction

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)


class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer perceptron.

    Args:
        input_dim (int): Input dimension.
        hidden_dims (List[int]): List of the hidden dimensions.
        output_dim (int): Output dimension.
        dropout (float): Dropout rate, optional.
    """
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

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the multi-layer perceptron.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            y (Tensor): Output tensor of shape (batch_size, output_dim).
        """

        return self.mlp(x)

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)

    def _initialize_weights(self):
        """
        Initialize the weights of the multi-layer perceptron.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data, a=sqrt(5))
                nn.init.zeros_(m.bias.data)
