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
    def __init__(self, data: pd.DataFrame, tokenizer: BertTokenizer) -> None:
        self.data = data
        self.tokenizer = tokenizer
        
        self.num_movies = len(self.data)
        self.num_samples_per_movie = len(self.data["request"].iloc[0])

    def __len__(self) -> int:
        return self.num_movies * self.num_samples_per_movie
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        movie_idx, request_idx = divmod(idx, len(self.data))

        requests = self.data.iloc[movie_idx]["request"]

        anchor_request = requests[request_idx]

        positive_request_idx = random.choice([i for i in range(self.num_samples_per_movie) if i != request_idx])

        positive_request = requests[positive_request_idx]

        negative_movie_idx = random.choice([i for i in range(self.num_movies) if i != movie_idx])

        negative_request = random.choice(self.data.iloc[negative_movie_idx]["request"])

        return torch.tensor(anchor_request), torch.tensor(positive_request), torch.tensor(negative_request)

def get_feature_sizes(ratings: pd.DataFrame) -> Tuple[int, ...]:
    return ratings["user_id"].nunique(), ratings["movie_id"].nunique()