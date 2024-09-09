import torch
from torch import Tensor
from torchmetrics.functional import pairwise_cosine_similarity


def cosine_distance(x: Tensor, y: Tensor) -> Tensor:
    """
    Returns the cosine distance between two tensors.
    """
    return 1 - torch.cosine_similarity(x, y)

def pairwise_cosine_distance(x: Tensor, y: Tensor = None) -> Tensor:
    """
    Returns the pairwise cosine distance between two tensors.
    """
    return 1 - pairwise_cosine_similarity(x, y, zero_diagonal=False)