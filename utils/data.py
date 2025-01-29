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
    """
    Dataset for ratings prediction, used to train the recommender system.

    Args:
        ratings (pd.DataFrame): The ratings dataframe.    
    """

    def __init__(self, ratings: pd.DataFrame) -> None:
        self.ratings = ratings

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Retrieves a ratings sample.

        Args:
            idx (int): The index of the sample.

        Returns:
            feature_ids (Tensor): The feature item IDs.
            feature_ratings (Tensor): The feature item ratings.
            item_id (Tensor): The target item ID.
            rating (Tensor): The target item rating.        
        """

        feature_ids, feature_ratings, item_id, rating = self.ratings.iloc[idx][["feature_ids", "feature_ratings", "item_id", "rating"]]

        return (
            torch.tensor(feature_ids, dtype=torch.int64),
            torch.tensor(feature_ratings, dtype=torch.float32),
            torch.tensor(item_id, dtype=torch.int64),
            torch.tensor(rating / 5.0, dtype=torch.float32),
        )


class CollaborativeDataset(Dataset):
    """
    Dataset for collaborative filtering.

    Args:
        ratings (pd.DataFrame): The ratings dataframe.
        requests (pd.DataFrame): The requests dataframe. 
    """

    def __init__(self, ratings: pd.DataFrame, requests: pd.DataFrame) -> None:
        self.ratings = ratings
        self.requests = requests

        self.unique_item_ids = self.ratings["item_id"].unique()
        self.num_items = len(self.requests)

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tuple[str, Tensor], Tensor]:
        """
        Retrieves a collaborative filtering sample.

        Args:
            idx (int): The index of the sample.

        Returns:
            feature_ids (Tensor): The feature item IDs.
            feature_ratings (Tensor): The feature item ratings.
            item_id (Tensor): The target item ID.
            rating (Tensor): The target item rating.
            anchor (Tuple[str, Tensor]): The anchor request and target item ID.
            negative_item_id (Tensor): The negative item ID.
        """

        feature_ids, feature_ratings, item_id, rating = self.ratings.iloc[idx][["feature_ids", "feature_ratings", "item_id", "rating"]]

        anchor_requests = self.requests.loc[item_id]["requests"]

        anchor_request = random.choice(anchor_requests)

        negative_item_id = random.choice([i for i in self.unique_item_ids if i != item_id])

        return (
            torch.tensor(feature_ids, dtype=torch.int64),
            torch.tensor(feature_ratings, dtype=torch.float32),
            torch.tensor(item_id, dtype=torch.int64),
            torch.tensor(rating / 5.0, dtype=torch.float32),
            (anchor_request, torch.tensor(item_id, dtype=torch.int64)),
            torch.tensor(negative_item_id, dtype=torch.int64),
        )


class SimulatorDataset(Dataset):
    """
    Dataset for the user simulator.

    Args:
        items (pd.DataFrame): The items dataframe.
        tokenizer (AutoTokenizer): The tokenizer.
        prompt_generators (List[Callable]): The prompt generators which generate prompts given a item title.
    """

    def __init__(self, items: pd.DataFrame, tokenizer: AutoTokenizer, prompt_generators: List[Callable]) -> None:
        self.items = items
        self.tokenizer = tokenizer
        self.prompt_generators = prompt_generators

    def __len__(self) -> int:
        return len(self.items) * len(self.prompt_generators)

    def __getitem__(self, idx: int) -> Tuple[int, str, str]:
        """
        Retrieves an item and associated prompt.

        Args:
            idx (int): The index of the item.
        
        Returns:
            item_id (int): The item ID.
            item_title (str): The item title.
            prompt (str): The prompt.
        """
        
        prompt_idx, item_idx = divmod(idx, len(self.items))
        item_id, item_title = self.items.iloc[item_idx][["item_id", "item_title"]]

        # Generate the prompt for the movie
        prompt = self.prompt_generators[prompt_idx](item_title)

        # Form prompt
        chat = [{"role": "user", "content": prompt}]

        # Apply the chat template
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        return item_id, item_title, prompt


def ratings_collate_fn(batch: List[Tuple[Tensor, Tensor, Tensor, Tensor]]) -> Tuple[Tuple[List[Tensor], List[Tensor], Tensor], Tensor]:
    """
    Collates a batch of ratings data.

    Args:
        batch (List[Tuple[Tensor, Tensor, Tensor, Tensor]]): The batch of data.

    Returns:
        rec_features (Tuple[List[Tensor], List[Tensor], Tensor]): User and item features.
        rec_targets (Tensor): Target ratings
    """
    feature_ids, feature_ratings, target_ids, target_ratings = zip(*batch)

    rec_features = (feature_ids, feature_ratings, torch.stack(target_ids))
    rec_targets = torch.stack(target_ratings)

    return rec_features, rec_targets


def collaborative_collate_fn(
    batch: List[Tuple[Tensor, Tensor, Tensor, Tensor, Tuple[str, int], int]]
) -> Tuple[Tuple[List[Tensor], List[Tensor], Tensor], Tensor, Tuple[str, Tensor], Tensor]:
    """
    Collates a batch of collaborative data.

    Args:
        batch (List[Tuple[Tensor, Tensor, Tensor, Tensor, Tuple[str, int], int]]): The batch of data.

    Returns:
        rec_features (Tuple[List[Tensor], List[Tensor], Tensor]): User and item features.
        rec_targets (Tensor): Target ratings.
        anchors (Tuple[str, Tensor]): Anchor request and item ID.
        negative_ids (Tensor): Negative item IDs.
    """

    # Unzip the batch
    feature_ids, feature_ratings, target_ids, target_ratings, anchors, negative_ids = zip(*batch)
    anchor_requests, anchor_ids = zip(*anchors)

    # Stack the tensors
    rec_features = (feature_ids, feature_ratings, torch.stack(target_ids))
    rec_targets = torch.stack(target_ratings)
    anchors = (anchor_requests, torch.stack(anchor_ids))
    negative_ids = torch.stack(negative_ids)

    return rec_features, rec_targets, anchors, negative_ids


def train_test_split_ratings(ratings: pd.DataFrame, train_size: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ratings into a train and test set. For a user with n ratings the first
    floor(n * train_size) ratings are used for training and the rest for testing.

    Args:
        ratings (pd.DataFrame): The ratings dataframe.
        train_size (float): The fraction of ratings to use for training.
    
    Returns:
        train_ratings (pd.DataFrame): The training ratings.
        test_ratings (pd.DataFrame): The testing ratings.
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
    """
    Split requests into a train and test set.

    Args:
        requests (pd.DataFrame): The requests dataframe.
        kwargs: Additional arguments to pass to train_test_split.
    """

    def split_request(row: pd.Series) -> pd.Series:
        train_req, test_req = train_test_split(row["request"], **kwargs)

        return pd.Series([train_req, test_req])

    requests[["train_requests", "test_requests"]] = requests.apply(split_request, axis=1)

    train_requests = requests[["item_id", "item_title", "train_requests"]].rename(columns={"train_requests": "requests"})
    test_requests = requests[["item_id", "item_title", "test_requests"]].rename(columns={"test_requests": "requests"})

    return train_requests, test_requests


def simulate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataloader: DataLoader,
    max_length: int = 64,
    output_column_name: str = "request"
) -> pd.DataFrame:
    """
    Generate responses for a given prompt using a model.

    Args:
        model (AutoModelForCausalLM): Language model.
        tokenizer (AutoTokenizer): Tokenizer.
        dataloader (DataLoader): The data to generate responses for.
        max_length (int, optional): The maximum length of the generated response.
        output_column_name (str, optional): The name of the output column in the returned dataframe.
    
    Returns:
        data (pd.DataFrame): The generated responses.
    """
    data = pd.DataFrame(columns=["item_id", "item_title", output_column_name])

    with torch.no_grad():
        for item_ids, item_titles, prompts in tqdm(dataloader, desc="Simulating", unit="batch", dynamic_ncols=True):
            # Tokenize input
            batch_input_tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)

            # Generate response
            batch_output_tokens = model.generate(**batch_input_tokens, max_new_tokens=max_length, do_sample=True)

            # Decode response
            responses = [
                tokenizer.decode(output_tokens[len(input_tokens) :], skip_special_tokens=True).strip('"')
                for input_tokens, output_tokens in zip(batch_input_tokens["input_ids"], batch_output_tokens)
            ]

            batch_output = pd.DataFrame(
                {
                    "item_id": item_ids,
                    "item_title": item_titles,
                    output_column_name: responses,
                }
            )

            data = pd.concat([data, batch_output], ignore_index=True)

    return data
