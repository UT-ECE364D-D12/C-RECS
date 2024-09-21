import random
from typing import Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
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
    
class EncoderDataset(Dataset):
    def __init__(self, ratings_data: pd.DataFrame, request_data: pd.DataFrame) -> None:
        self.ratings_data = ratings_data
        self.request_data = request_data

        self.unique_movie_ids = self.request_data["movie_id"].unique()
        self.movie_id_to_idx = {movie_id: i for i, movie_id in enumerate(self.unique_movie_ids)}

        self.num_movies = len(self.request_data)

    def __len__(self) -> int:
        return len(self.ratings_data)

    def __getitem__(self, idx: int) -> Tuple[Tuple[Tensor, int], Tuple[str, int], Tuple[str, int]]:
        user_id, movie_id, rating = self.ratings_data.iloc[idx][["user_id", "movie_id", "rating"]]

        anchor_requests = self.request_data.loc[movie_id]["requests"]

        anchor_request = random.choice(anchor_requests)
        
        negative_movie_id = random.choice([i for i in self.unique_movie_ids if i != movie_id])

        negative_requests = self.request_data.loc[negative_movie_id]["requests"]

        negative_request = random.choice(negative_requests)

        anchor_id = self.movie_id_to_idx[movie_id]
        negative_id = self.movie_id_to_idx[negative_movie_id]

        return torch.tensor([user_id - 1, anchor_id]), torch.tensor(rating / 5.0), (anchor_request, anchor_id), (negative_request, negative_id)

def train_test_split_requests(requests: pd.DataFrame, train_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:

    def split_request(row: pd.Series) -> pd.Series: 
        train_req, test_req = train_test_split(row['request'], train_size=train_size)

        return pd.Series([train_req, test_req])

    requests[['train_requests', 'test_requests']] = requests.apply(split_request, axis=1)

    train_requests = requests[['movie_id', 'movie_title', 'train_requests']].rename(columns={'train_requests': 'requests'})
    test_requests = requests[['movie_id', 'movie_title', 'test_requests']].rename(columns={'test_requests': 'requests'})

    return train_requests, test_requests

def get_feature_sizes(ratings: pd.DataFrame) -> Tuple[int, ...]:
    return ratings["user_id"].nunique(), ratings["movie_id"].nunique()