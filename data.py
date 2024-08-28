from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


class MovieLensDataset(Dataset):
    def __init__(self, path: str = "data/MovieLens 100k/u.data") -> None:
        self.ratings = pd.read_csv(path, sep="\t", header=None, names=["user_id", "movie_id", "rating", "timestamp"])

    def __len__(self) -> int:
        return len(self.ratings)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        user_id, movie_id, rating = self.ratings.iloc[idx][["user_id", "movie_id", "rating"]]

        return torch.tensor([user_id - 1, movie_id - 1]), torch.tensor(rating)