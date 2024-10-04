import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import argparse
import gc
from typing import Callable, List

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, MistralForCausalLM

from utils.misc import build_language_model

from utils.data import SimulatorDataset, simulate
from utils.misc import build_language_model

MODEL_NAMES = ["mistralai/Mistral-7B-Instruct-v0.2", "google/gemma-7b-it"]
SPLIT_STRINGS = ["[/INST] ", "\nmodel\n"]
PROMPT_GENERATORS = [
    lambda movie: f"""You are interacting with a movie recommendation system. Your goal is to make a short request that will help the system to suggest the movie "{movie}" without mentioning its title, characters, or ANY plot elements. The response should instead use GENERAL characteristics like the genre, tone, and themes of the movie. Your request should be concise, sound conversational, and not be too enthusiastic. As an example, for the movie "Crazy Stupid Love" you should give a request like "I'm looking for a silly romantic comedy with a happy ending. Any suggestions?" Reply ONLY with the human-like request for a movie. DO NOT include any other text.""",
    lambda movie: f"""You are interacting with a movie recommendation system. Your goal is to make a short request that will help the system to suggest the movie "{movie}" without mentioning its title, characters, or ANY plot elements. The response should instead use GENERAL characteristics like the genre, tone, and themes of the movie. Your request should be extremely short, sound conversational, not be too enthusiastic, and use informal wording. As an example, for the movie "La La Land" you should give a request like "I want to watch a rom com." Reply ONLY with the human-like request for a movie. DO NOT include any other text.""",
    lambda movie: f"""You are interacting with a movie recommendation system. Your goal is to make a request that will guide the system to suggest the movie "{movie}" without directly referencing any specific plot points, names, or events. Focus on describing the mood, pacing, and type of experience the movie offers. Your request should sound curious, thoughtful, and provide some context about what you're in the mood for, but without being overly specific. For example, for the movie "Inception," you could say something like, "I'm in the mood for something mind-bending with a lot of twists. What do you recommend?" Reply ONLY with the human-like request for a movie. DO NOT include any other text.""",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default="data/ml-20m/movies.csv", help='Path to the dataset')
    parser.add_argument('--output_path', type=str, default="data/ml-20m/requests.csv", help='Path to save the requests')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the dataloader')

    args = parser.parse_args()

    data_path = args.dataset_path
    output_path = args.output_path
    batch_size = args.batch_size

    # Read in the data
    movies = pd.read_csv(data_path, header=0, names=["movie_id", "movie_title", "genres"])

    movies = movies[["movie_id", "movie_title"]]

    data = pd.DataFrame(columns=["movie_id", "movie_title", "request"])

    for model_name, split_string in zip(MODEL_NAMES, SPLIT_STRINGS):
        print(f"Loading {model_name}...")

        # Load the model and tokenizer
        language_model, language_tokenizer = build_language_model(model_name)

        # Create the dataset
        dataset = SimulatorDataset(movies, language_tokenizer, PROMPT_GENERATORS)

        # Create the dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        # Simulate the requests
        model_data = simulate(language_model, language_tokenizer, split_string, dataloader, output_col="request")

        data = pd.concat([data, model_data], ignore_index=True)

        del language_model
        del language_tokenizer

        gc.collect()
        torch.cuda.empty_cache()

    data = data.groupby("movie_id").agg({
    "movie_title": "first",
    "request": list,
    }).reset_index()
    data.set_index("movie_id", inplace=True, drop=False)

    data.to_csv(output_path, index=False)
