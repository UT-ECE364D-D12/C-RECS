import random
from typing import Callable, List, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class RatingsDataset(Dataset):
    def __init__(self, ratings: pd.DataFrame) -> None:
        self.ratings = ratings

    def __len__(self) -> int:
        return len(self.ratings)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        feature_ids, feature_ratings, item_id, rating = self.ratings.iloc[idx][["feature_ids", "feature_ratings", "item_id", "rating"]]

        return torch.tensor(feature_ids, dtype=torch.int64), torch.tensor(feature_ratings, dtype=torch.float32), torch.tensor(item_id, dtype=torch.int64), torch.tensor(rating / 5.0, dtype=torch.float32)
    
class CollaborativeDataset(Dataset):
    def __init__(self, ratings_data: pd.DataFrame, request_data: pd.DataFrame) -> None:
        self.ratings_data = ratings_data
        self.request_data = request_data

        self.unique_movie_ids = self.ratings_data["movie_id"].unique()
        self.num_movies = len(self.request_data)

    def __len__(self) -> int:
        return len(self.ratings_data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tuple[str, int], Tuple[str, int], int]:
        user_id, movie_id, rating = self.ratings_data.iloc[idx][["user_id", "movie_id", "rating"]]
        
        user_id, movie_id = int(user_id), int(movie_id)

        anchor_requests = self.request_data.loc[movie_id]["requests"]

        anchor_request = random.choice(anchor_requests)
        
        negative_movie_id = random.choice([i for i in self.unique_movie_ids if i != movie_id])

        return torch.tensor([user_id, movie_id]), torch.tensor(rating / 5.0), (anchor_request, movie_id), (negative_movie_id)
        
class ContentDataset(Dataset):
    def __init__(self, descriptions: pd.DataFrame, requests: pd.DataFrame) -> None:
        self.descriptions = descriptions
        self.requests = requests

        self.num_movies = len(self.requests)
        self.num_requests_per_movie = len(self.requests.iloc[0]["requests"])        

    def __len__(self) -> int:
        return self.num_movies * self.num_requests_per_movie

    def __getitem__(self, idx: int) -> Tuple[Tuple[str, int], Tuple[str, int], Tuple[str, int]]:
        """
        Returns an anchor, positive, and negative sample, each containing a description and an item id.
        """
        movie_idx, request_idx = divmod(idx, self.num_requests_per_movie)

        movie_id, anchor_requests = self.requests.iloc[movie_idx][["movie_id", "requests"]]

        anchor_request = anchor_requests[request_idx]

        positive_description = self.descriptions.loc[movie_id]["description"]
        
        negative_movie_idx = random.choice([i for i in range(self.num_movies) if i != movie_idx])

        negative_movie_id, negative_description = self.descriptions.iloc[negative_movie_idx][["movie_id", "description"]]

        return (anchor_request, movie_id), (positive_description, movie_id), (negative_description, negative_movie_id)
    
class DescriptionsDataset(Dataset):
    def __init__(self, descriptions: pd.DataFrame) -> None:
        self.descriptions = descriptions

        self.num_descriptions = len(self.descriptions)

    def __len__(self) -> int:
        return self.num_descriptions

    def __getitem__(self, idx: int) -> Tuple[int, str]:
        movie_id, description = self.descriptions.iloc[idx][["movie_id", "description"]]

        return movie_id, description
    
class SimulatorDataset(Dataset):
    def __init__(self, movies: pd.DataFrame, tokenizer: AutoTokenizer, prompt_generators: List[Callable]) -> None:
        self.movies = movies
        self.tokenizer = tokenizer
        self.prompt_generators = prompt_generators

    def __len__(self):
        return len(self.movies) * len(self.prompt_generators)

    def __getitem__(self, idx: int):
        prompt_idx, movie_idx = divmod(idx, len(self.movies))
        movie_id, movie_title = self.movies.iloc[movie_idx][["movie_id", "movie_title"]]

        # Generate the prompt for the movie
        prompt = self.prompt_generators[prompt_idx](movie_title)

        # Form prompt
        chat = [{"role": "user", "content": prompt}]

        # Apply the chat template
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        return movie_id, movie_title, prompt
    
def ratings_collate_fn(batch: List[Tuple[Tensor, Tensor, Tensor, Tensor]]) -> Tuple[Tuple[List[Tensor], List[Tensor], Tensor], Tensor]:
    feature_ids, feature_ratings, target_ids, target_ratings = zip(*batch)

    return (feature_ids, feature_ratings, torch.stack(target_ids)), torch.stack(target_ratings)

def train_test_split_ratings(ratings: pd.DataFrame, train_size: float = 0.8):
    """
    Split ratings into a train and test set. For a user with n ratings the first 
    floor(n * train_size) ratings are used for training and the rest for testing.
    """
    def split_user_ratings(user_ratings: pd.DataFrame):
        num_ratings = len(user_ratings)
        num_train_ratings = int(num_ratings * train_size)
        
        train_data = user_ratings.iloc[:num_train_ratings]
        test_data = user_ratings.iloc[num_train_ratings:]
        
        return train_data, test_data

    train_ratings = []
    test_ratings = []

    for _, user_ratings in ratings.groupby("user_id"):
        train_data, val_data = split_user_ratings(user_ratings)

        train_ratings.append(train_data)
        test_ratings.append(val_data)

    train_ratings = pd.concat(train_ratings)
    test_ratings = pd.concat(test_ratings)

    return train_ratings, test_ratings
    
def train_test_split_requests(requests: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:

    def split_request(row: pd.Series) -> pd.Series: 
        train_req, test_req = train_test_split(row['request'], **kwargs)

        return pd.Series([train_req, test_req])

    requests[['train_requests', 'test_requests']] = requests.apply(split_request, axis=1)

    train_requests = requests[['movie_id', 'movie_title', 'train_requests']].rename(columns={'train_requests': 'requests'})
    test_requests = requests[['movie_id', 'movie_title', 'test_requests']].rename(columns={'test_requests': 'requests'})

    return train_requests, test_requests

def get_feature_sizes(ratings: pd.DataFrame) -> Tuple[int, ...]:
    return (ratings["item_id"].nunique() + 1,)

def simulate(
    language_model: AutoModelForCausalLM,
    language_tokenizer: AutoTokenizer,
    split_string: str,
    dataloader: DataLoader,
    max_length: int = 2048,
    output_col = "request"
) -> pd.DataFrame:
    data = pd.DataFrame(columns=["movie_id", "movie_title", output_col])

    with torch.no_grad():
        for movie_ids, movie_titles, prompts in tqdm(dataloader, desc="Simulating", unit="batch"):
            # Tokenize input
            input_tokens = language_tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt").to(language_model.device)

            # Generate request
            output_tokens = language_model.generate(**input_tokens, max_new_tokens=max_length, do_sample=True, pad_token_id=language_tokenizer.eos_token_id)

            # Decode request
            batch_output = [language_tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True).split(split_string)[-1] for output in output_tokens]

            batch_output = pd.DataFrame({
                "movie_id": movie_ids,
                "movie_title": movie_titles,
                output_col: batch_output,
            })

            data = pd.concat([data, batch_output], ignore_index=True)

    return data