import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import argparse
import gc
from typing import Callable, List

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    MistralForCausalLM,
)

from utils.data import SimulatorDataset, simulate
from utils.misc import build_language_model

# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
# SPLIT_STRING = "[/INST] "

MODEL_NAME = "google/gemma-7b-it"
SPLIT_STRING = "\nmodel\n"

PROMPT = [lambda movie: f"""In one short paragraph, Introduce movie {movie} and describe its attributes precisely including but not limited to genre, director, actors, time period, country, character, plot/theme, mood/tone, critical acclaim/award"""]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default="data/ml-20m/movies.csv", help='Path to the dataset')
    parser.add_argument('--output_path', type=str, default="data/ml-20m/descriptions.csv", help='Path to save the descriptions')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the dataloader')

    args = parser.parse_args()

    data_path = args.dataset_path
    output_path = args.output_path
    batch_size = args.batch_size

    # Read in the data
    movies = pd.read_csv(data_path, header=0, names=["movie_id", "movie_title", "genres"])

    movies = movies[["movie_id", "movie_title"]]

    data = pd.DataFrame(columns=["movie_id", "movie_title", "description"])

    print(f"Loading {MODEL_NAME}...")

    # Load the model and tokenizer
    language_model, language_tokenizer = build_language_model(MODEL_NAME)

    # Create the dataset
    dataset = SimulatorDataset(movies, language_tokenizer, prompt_generators=PROMPT)

    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Simulate the requests
    model_data = simulate(language_model, language_tokenizer, SPLIT_STRING, dataloader, output_col="description")

    data = pd.concat([data, model_data], ignore_index=True)

    del language_model
    del language_tokenizer

    gc.collect()
    torch.cuda.empty_cache()

    data.to_csv(output_path, index=False)