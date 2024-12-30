import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import argparse
import gc

import pandas as pd
import torch
from torch.utils.data import DataLoader

from model.llm import build_language_model
from utils.data import SimulatorDataset, simulate

MODEL_NAMES = ["google/gemma-7b-it", "mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Meta-Llama-3.1-8B-Instruct"]
BATCH_SIZES = [16, 32, 32]

PROMPT_GENERATORS = [
    # Normal
    lambda movie: f"""You are interacting with a movie recommendation system. Your goal is to craft a casual and vague request that will help the system suggest the movie "{movie}" without mentioning its title, characters, or ANY plot elements. Focus on broad genres and simple preferences. For example, for the movie "The Godfather," you might say, "Looking for a classic crime drama." Reply ONLY with the human-like request for a movie. DO NOT include any other text.""",

    # Mood/Feeling
    lambda movie: f"""You are interacting with a movie recommendation system. Your goal is to describe the mood you're in, which should lead the system to suggest the movie "{movie}" without including any specific details. Keep it conversational and focus on your feelings or current vibe. For example, for the movie "The Notebook," you could say, "I'm in the mood for something romantic and emotional." Reply ONLY with the human-like request for a movie. DO NOT include any other text.""",

    # Country, Time Period, Critical Acclaim
    lambda movie: f"""You are interacting with a movie recommendation system. Your goal is to provide a request that reflects a niche preference while guiding the system to suggest the movie "{movie}" without mentioning any plot details. Emphasize aspects like country of origin, awards, critical acclaim, or the time period in which the movie was released (e.g., early 2000s, or use descriptors like new/old). For example, for the movie "Parasite," you could say, "Can you suggest a recent Oscar-winning foreign thriller?" Reply ONLY with the human-like request for a movie. DO NOT include any other text.""",

    # Style, Pacing, Theme
    lambda movie: f"""You are interacting with a movie recommendation system. Your goal is to make a precise yet intriguing request that will help the system suggest the movie "{movie}" without referring to its plot or characters. Highlight its pacing, storytelling style, or themes. For example, for the movie "Pulp Fiction," you could say, "I want something edgy and non-linear with a lot of dialogue." Reply ONLY with the human-like request for a movie. DO NOT include any other text.""",

    # Specific Actor
    lambda movie: f"""You are interacting with a movie recommendation system. Your goal is to create a request that will help the system recommend the movie "{movie}" without mentioning its title, characters, or plot. Instead, focus on requesting a movie featuring a specific actor from the movie. For example, for the movie "National Treasure," you might say, "I want to watch something with Nicolas Cage in it." Reply ONLY with the human-like request for a movie. DO NOT include any other text.""",

    # Similar Movie
    lambda movie: f"""You are interacting with a movie recommendation system. Your goal is to create a request that will guide the system to suggest the movie "{movie}" without mentioning its title, characters, or plot. Instead, base your request on a similar movie that shares key characteristics like genre, tone, or themes. For example, for the movie "The Dark Knight," you might say, "I'm looking for something like 'Batman Begins,' with a dark and gritty superhero vibe." Reply ONLY with the human-like request for a movie. DO NOT include any other text.""",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default="data/ml-20m/movies.csv", help='Path to the dataset')
    parser.add_argument('--output_path', type=str, default="data/ml-20m/requests.csv", help='Path to save the requests')

    args = parser.parse_args()

    data_path = args.dataset_path
    output_path = args.output_path

    # Read in the data
    movies = pd.read_csv(data_path, header=0, names=["item_id", "item_title", "genres"])

    movies = movies[["item_id", "item_title"]]

    data = pd.DataFrame(columns=["item_id", "item_title", "request"])

    for model_name, batch_size in zip(MODEL_NAMES, BATCH_SIZES):
        print(f"Loading {model_name}...")

        # Load the model and tokenizer
        language_model, language_tokenizer = build_language_model(model_name)

        # Create the dataset
        dataset = SimulatorDataset(movies, language_tokenizer, PROMPT_GENERATORS)

        # Create the dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        # Simulate the requests
        model_data = simulate(language_model, language_tokenizer, dataloader, output_column_name="request")

        data = pd.concat([data, model_data], ignore_index=True)

        del language_model
        del language_tokenizer

        gc.collect()
        torch.cuda.empty_cache()

    data.to_csv(output_path, index=False)