from typing import Tuple

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


class RatingsDataset(Dataset):
    def __init__(self, ratings: pd.DataFrame) -> None:
        self.ratings = ratings

    def __len__(self) -> int:
        return len(self.ratings)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        user_id, movie_id, rating = self.ratings.iloc[idx][["user_id", "movie_id", "rating"]]

        return torch.tensor([user_id - 1, movie_id - 1]), torch.tensor(rating / 5.0)
    
class RequestsDataset(Dataset):
    def __init__(self, requests: pd.DataFrame) -> None:
        self.requests = requests

    def __len__(self) -> int:
        return len(self.requests)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        request, movie_id = self.requests.iloc[idx][["encoded_request", "movie_id"]]

        return torch.tensor(request), torch.tensor(movie_id - 1)
    
def get_feature_sizes(ratings: pd.DataFrame) -> Tuple[int, ...]:
    return ratings["user_id"].nunique(), ratings["movie_id"].nunique()