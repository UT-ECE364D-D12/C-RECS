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

        self.unique_user_ids = self.ratings_data["user_id"].unique()
        self.user_id_to_idx = {user_id: i for i, user_id in enumerate(self.unique_user_ids)}

        self.unique_movie_ids = self.ratings_data["movie_id"].unique()
        self.movie_id_to_index = {movie_id: i for i, movie_id in enumerate(self.unique_movie_ids)}

    def __len__(self) -> int:
        return len(self.ratings)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        user_id, movie_id, rating = self.ratings.iloc[idx][["user_id", "movie_id", "rating"]]

        user_id = self.user_id_to_idx[user_id]
        movie_id = self.movie_id_to_index[movie_id]

        return torch.tensor([user_id, movie_id]), torch.tensor(rating / 5.0)
    
class CollaborativeDataset(Dataset):
    def __init__(self, ratings_data: pd.DataFrame, request_data: pd.DataFrame) -> None:
        self.ratings_data = ratings_data
        self.request_data = request_data

        self.unique_user_ids = self.ratings_data["user_id"].unique()
        self.user_id_to_idx = {user_id: i for i, user_id in enumerate(self.unique_user_ids)}

        self.unique_movie_ids = self.ratings_data["movie_id"].unique()
        self.movie_id_to_index = {movie_id: i for i, movie_id in enumerate(self.unique_movie_ids)}

        self.num_movies = len(self.request_data)

    def __len__(self) -> int:
        return len(self.ratings_data)

    def __getitem__(self, idx: int) -> Tuple[Tuple[Tensor, int], Tuple[str, int], Tuple[str, int]]:
        user_id, movie_id, rating = self.ratings_data.iloc[idx][["user_id", "movie_id", "rating"]]
        
        user_id, movie_id = int(user_id), int(movie_id)

        anchor_requests = self.request_data.loc[movie_id]["requests"]

        anchor_request = random.choice(anchor_requests)
        
        negative_movie_id = random.choice([i for i in self.unique_movie_ids if i != movie_id])

        negative_requests = self.request_data.loc[negative_movie_id]["requests"]

        negative_request = random.choice(negative_requests)

        user_id = self.user_id_to_idx[user_id]
        anchor_id = self.movie_id_to_index[movie_id]
        negative_id = self.movie_id_to_index[negative_movie_id]

        return torch.tensor([user_id, anchor_id]), torch.tensor(rating / 5.0), (anchor_request, anchor_id), (negative_request, negative_id)
class ContentDataset(Dataset):
    def __init__(self, descriptions_data: pd.DataFrame, request_data: pd.DataFrame) -> None:
        self.descriptions_data = descriptions_data
        self.request_data = request_data

        self.unique_movie_ids = self.descriptions_data["movie_id"].unique()
        self.movie_id_to_index = {movie_id: i for i, movie_id in enumerate(self.unique_movie_ids)}

    def __len__(self) -> int:
        return len(self.descriptions_data)

    def __getitem__(self, idx: int) -> Tuple[Tuple[str, int], Tuple[str, int], Tuple[str, int]]:
        movie_id, positive_description = self.descriptions_data.iloc[idx][["movie_id", "description"]]

        anchor_requests = self.request_data.loc[movie_id]["requests"]

        anchor_request = random.choice(anchor_requests)
        
        negative_movie_id = random.choice([i for i in self.unique_movie_ids if i != movie_id])

        negative_requests = self.request_data.loc[negative_movie_id]["requests"]

        negative_request = random.choice(negative_requests)

        anchor_id = self.movie_id_to_index[movie_id]
        negative_id = self.movie_id_to_index[negative_movie_id]

        return (anchor_request, anchor_id), (positive_description, anchor_id), (negative_request, negative_id)

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