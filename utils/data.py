import random
from typing import Tuple

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import BertTokenizer


class RatingsDataset(Dataset):
    def __init__(self, ratings: pd.DataFrame) -> None:
        self.ratings = ratings

    def __len__(self) -> int:
        return len(self.ratings)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        user_id, movie_id, rating = self.ratings.iloc[idx][["user_id", "movie_id", "rating"]]

        return torch.tensor([user_id - 1, movie_id - 1]), torch.tensor(rating / 5.0)
    
class DecoderDataset(Dataset):
    def __init__(self, requests: pd.DataFrame) -> None:
        self.requests = requests

    def __len__(self) -> int:
        return len(self.requests)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        request, movie_id = self.requests.iloc[idx][["encoded_request", "movie_id"]]

        return torch.tensor(request), torch.tensor(movie_id - 1)
    
class EncoderDataset(Dataset):
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        
        self.num_movies = len(self.data)
        self.num_samples_per_movie = len(self.data["request"].iloc[0])

    def __len__(self) -> int:
        return self.num_movies * self.num_samples_per_movie
    
    def __getitem__(self, idx: int) -> Tuple[Tuple[str, int], Tuple[str, int], Tuple[str, int]]:
        movie_idx, request_idx = divmod(idx, self.num_samples_per_movie)

        anchor_id, anchor_requests = self.data.iloc[movie_idx][["movie_id", "request"]]

        anchor_request = anchor_requests[request_idx]

        positive_request_idx = random.choice([i for i in range(self.num_samples_per_movie) if i != request_idx])

        positive_request = anchor_requests[positive_request_idx]

        negative_movie_idx = random.choice([i for i in range(self.num_movies) if i != movie_idx])

        negative_id, negative_requests = self.data.iloc[negative_movie_idx][["movie_id", "request"]]
        
        negative_request = random.choice(negative_requests)

        return ((anchor_request, anchor_id - 1), (positive_request, anchor_id - 1), (negative_request, negative_id - 1))

def get_feature_sizes(ratings: pd.DataFrame) -> Tuple[int, ...]:
    return ratings["user_id"].nunique(), ratings["movie_id"].nunique()