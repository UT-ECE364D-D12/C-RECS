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
    def __init__(self, ratings: pd.DataFrame, requests: pd.DataFrame) -> None:
        self.ratings = ratings
        self.requests = requests

        self.unique_item_ids = self.ratings["item_id"].unique()
        self.num_items = len(self.requests)

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tuple[str, Tensor], Tensor]:
        feature_ids, feature_ratings, item_id, rating = self.ratings.iloc[idx][["feature_ids", "feature_ratings", "item_id", "rating"]]

        anchor_requests = self.requests.loc[item_id]["requests"]

        anchor_request = random.choice(anchor_requests)
        
        negative_item_id = random.choice([i for i in self.unique_item_ids if i != item_id])

        return torch.tensor(feature_ids, dtype=torch.int64), torch.tensor(feature_ratings, dtype=torch.float32), torch.tensor(item_id, dtype=torch.int64), torch.tensor(rating / 5.0, dtype=torch.float32), (anchor_request, torch.tensor(item_id, dtype=torch.int64)), torch.tensor(negative_item_id, dtype=torch.int64)
        
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
        item_idx, request_idx = divmod(idx, self.num_requests_per_movie)

        item_id, anchor_requests = self.requests.iloc[item_idx][["item_id", "requests"]]

        anchor_request = anchor_requests[request_idx]

        positive_description = self.descriptions.loc[item_id]["description"]
        
        negative_item_idx = random.choice([i for i in range(self.num_movies) if i != item_idx])

        negative_item_id, negative_description = self.descriptions.iloc[negative_item_idx][["item_id", "description"]]

        return (anchor_request, item_id), (positive_description, item_id), (negative_description, negative_item_id)
    
class DescriptionsDataset(Dataset):
    def __init__(self, descriptions: pd.DataFrame) -> None:
        self.descriptions = descriptions

        self.num_descriptions = len(self.descriptions)

    def __len__(self) -> int:
        return self.num_descriptions

    def __getitem__(self, idx: int) -> Tuple[int, str]:
        item_id, description = self.descriptions.iloc[idx][["item_id", "description"]]

        return item_id, description
    
class SimulatorDataset(Dataset):
    def __init__(self, movies: pd.DataFrame, tokenizer: AutoTokenizer, prompt_generators: List[Callable]) -> None:
        self.movies = movies
        self.tokenizer = tokenizer
        self.prompt_generators = prompt_generators

    def __len__(self):
        return len(self.movies) * len(self.prompt_generators)

    def __getitem__(self, idx: int):
        prompt_idx, item_idx = divmod(idx, len(self.movies))
        item_id, item_title = self.movies.iloc[item_idx][["item_id", "item_title"]]

        # Generate the prompt for the movie
        prompt = self.prompt_generators[prompt_idx](item_title)

        # Form prompt
        chat = [{"role": "user", "content": prompt}]

        # Apply the chat template
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        return item_id, item_title, prompt
    
def ratings_collate_fn(batch: List[Tuple[Tensor, Tensor, Tensor, Tensor]]) -> Tuple[Tuple[List[Tensor], List[Tensor], Tensor], Tensor]:
    feature_ids, feature_ratings, target_ids, target_ratings = zip(*batch)

    return (feature_ids, feature_ratings, torch.stack(target_ids)), torch.stack(target_ratings)

def collaborative_collate_fn(batch: List[Tuple[Tensor, Tensor, Tensor, Tensor, Tuple[str, int], int]]) -> Tuple[Tuple[List[Tensor], List[Tensor], Tensor], Tensor, Tuple[str, Tensor], Tensor]:
    feature_ids, feature_ratings, target_ids, target_ratings, anchors, negative_ids = zip(*batch)
    anchor_requests, anchor_ids = zip(*anchors)

    return (feature_ids, feature_ratings, torch.stack(target_ids)), torch.stack(target_ratings), (anchor_requests, torch.stack(anchor_ids)), torch.stack(negative_ids)

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

    train_requests = requests[['item_id', 'item_title', 'train_requests']].rename(columns={'train_requests': 'requests'})
    test_requests = requests[['item_id', 'item_title', 'test_requests']].rename(columns={'test_requests': 'requests'})

    return train_requests, test_requests

def simulate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataloader: DataLoader,
    max_length: int = 64,
    output_column_name: str = "request"
) -> pd.DataFrame:
    data = pd.DataFrame(columns=["item_id", "item_title", output_column_name])

    with torch.no_grad():
        for item_ids, item_titles, prompts in tqdm(dataloader, desc="Simulating", unit="batch"):
            # Tokenize input
            batch_input_tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)

            # Generate response
            batch_output_tokens = model.generate(**batch_input_tokens, max_new_tokens=max_length, do_sample=True)

            # Decode response
            responses = [tokenizer.decode(output_tokens[len(input_tokens):], skip_special_tokens=True).strip('\"') for input_tokens, output_tokens in zip(batch_input_tokens["input_ids"], batch_output_tokens)]

            batch_output = pd.DataFrame({
                "item_id": item_ids,
                "item_title": item_titles,
                output_column_name: responses,
            })

            data = pd.concat([data, batch_output], ignore_index=True)

    return data