from sklearn.model_selection import train_test_split

from data.simulator_dataset import SimulatorDataset, simulate
from utils.misc import suppress_warnings

suppress_warnings()

import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import argparse
import gc
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from model.llm import build_language_model

MODEL_NAMES = ["meta-llama/Llama-3.3-70B-Instruct"]
BATCH_SIZES = [48]

# Prompts for generating requests
REQUEST_PROMPTS = [
    # Normal
    lambda movie: f"""You are interacting with a movie recommendation system. Your goal is to craft a casual and vague request that will help the system suggest the movie "{movie}" without mentioning its title, characters, or ANY plot elements. Focus on broad genres and simple preferences. For example, for the movie "The Godfather," you might say, "Looking for a classic crime drama." Reply ONLY with the human-like request for {movie}. DO NOT include any other text.""",
    # Mood/Feeling
    lambda movie: f"""You are interacting with a movie recommendation system. Your goal is to describe the mood you're in, which should lead the system to suggest the movie "{movie}" without including any specific details. Keep it conversational and focus on your feelings or current vibe. For example, for the movie "The Notebook," you could say, "I'm in the mood for something romantic and emotional." Reply ONLY with the human-like request for {movie}. DO NOT include any other text.""",
    # Country, Time Period, Critical Acclaim
    lambda movie: f"""You are interacting with a movie recommendation system. Your goal is to provide a request that reflects a niche preference while guiding the system to suggest the movie "{movie}" without mentioning any plot details. Emphasize aspects like country of origin, awards, critical acclaim, or the time period in which the movie was released (e.g., early 2000s, or use descriptors like new/old). For example, for the movie "Parasite," you could say, "Can you suggest a recent Oscar-winning foreign thriller?" Reply ONLY with the human-like request for {movie}. DO NOT include any other text.""",
    # Style, Pacing, Theme
    lambda movie: f"""You are interacting with a movie recommendation system. Your goal is to make a precise yet intriguing request that will help the system suggest the movie "{movie}" without referring to its plot or characters. Highlight its pacing, storytelling style, or themes. For example, for the movie "Pulp Fiction," you could say, "I want something edgy and non-linear with a lot of dialogue." Reply ONLY with the human-like request for {movie}. DO NOT include any other text.""",
    # Specific Actor
    lambda movie: f"""You are interacting with a movie recommendation system. Your goal is to create a request that will help the system recommend the movie "{movie}" without mentioning its title, characters, or plot. Instead, focus on requesting a movie featuring a specific actor from the movie. For example, for the movie "National Treasure," you might say, "I want to watch something with Nicolas Cage in it." Reply ONLY with the human-like request for {movie}. DO NOT include any other text.""",
    # Similar Movie
    lambda movie: f"""You are interacting with a movie recommendation system. Your goal is to create a request that will guide the system to suggest the movie "{movie}" without mentioning its title, characters, or plot. Instead, base your request on a similar movie that shares key characteristics like genre, tone, or themes. For example, for the movie "The Dark Knight," you might say, "I'm looking for something like 'Batman Begins,' with a dark and gritty superhero vibe." Reply ONLY with the human-like request for {movie}. DO NOT include any other text.""",
]

# Prompt for generating descriptions
DESCRIPTION_PROMPTS = [
    lambda movie: f"""You are interacting with a movie recommendation system. Your goal is to create a description for the movie "{movie}". Write a single concise paragraph that weaves together the genre, plot/theme, mood/tone, key cast, director, release year, country of origin, and critical acclaim.\nFor example, for the movie “Inception,” you might say:\n“Inception is a 2010 sci-fi thriller/heist from director Christopher Nolan that unfolds across mind-bending, multi-layered dreams in contemporary urban settings. Leonardo DiCaprio leads a stellar cast (with Joseph Gordon-Levitt, Ellen Page, Tom Hardy, and Marion Cotillard) as an expert “extractor” who assembles a team to plant an idea in a target’s subconscious, blending tense, cerebral suspense with blockbuster action. A UK/USA co-production shot with crisp, high-contrast visuals, it won four Academy Awards (Visual Effects, Sound Editing, Sound Mixing, and Cinematography).\nReply ONLY with the description for {movie}. DO NOT include any other text.""",
]


def main(data_root: Path, mode: str, max_length: int = 64, train_size: float = 0.8, val_size: float = 0.1) -> None:
    if mode == "requests":
        prompt_generators = REQUEST_PROMPTS
    elif mode == "descriptions":
        prompt_generators = DESCRIPTION_PROMPTS
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from 'requests' or 'descriptions'.")

    print(f"Generating {mode} with a maximum length of {max_length} tokens...")

    # Read in the item data
    movies = pd.read_csv(data_root / "movies.csv", header=0, names=["item_id", "item_title", "genres"])[["item_id", "item_title"]]

    data = pd.DataFrame(columns=["item_id", "item_title", "text"])

    for model_name, batch_size in zip(MODEL_NAMES, BATCH_SIZES):
        print(f"Loading {model_name}...")

        # Load the model and tokenizer
        model, tokenizer = build_language_model(model_name)

        # Create the dataset & dataloader
        dataset = SimulatorDataset(movies, tokenizer, prompt_generators)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=6)

        # Simulate the requests
        model_data = simulate(model, tokenizer, dataloader, max_length=max_length)

        data = pd.concat([data, model_data], ignore_index=True)

        del model
        del tokenizer

        gc.collect()
        torch.cuda.empty_cache()

    data.to_csv(data_root / f"{mode}s.csv", index=False)

    # Split the requests into train, validation, and test sets
    if mode == "requests":
        data = data.groupby("item_id").agg({"item_title": "first", "text": list}).reset_index()
        data.set_index("item_id", inplace=True, drop=False)

        test_size = 1 - train_size - val_size
        val_size = val_size / (train_size + val_size)

        def split_request(row: pd.Series) -> pd.Series:
            train_req, test_req = train_test_split(row["text"], test_size=test_size)
            train_req, val_req = train_test_split(train_req, test_size=val_size)

            return pd.Series([train_req, val_req, test_req])

        data[["train_text", "val_text", "test_text"]] = data.apply(split_request, axis=1)

        train_data = data[["item_id", "item_title", "train_text"]].rename(columns={"train_text": mode})
        val_data = data[["item_id", "item_title", "val_text"]].rename(columns={"val_text": mode})
        test_data = data[["item_id", "item_title", "test_text"]].rename(columns={"test_text": mode})

        # Flatten the requests
        train_data = train_data.explode(mode).reset_index(drop=True)
        val_data = val_data.explode(mode).reset_index(drop=True)
        test_data = test_data.explode(mode).reset_index(drop=True)

        train_data.to_csv(data_root / f"train_{mode}s.csv", index=False)
        val_data.to_csv(data_root / f"val_{mode}s.csv", index=False)
        test_data.to_csv(data_root / f"test_{mode}s.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path, help="Root directory of MovieLens data")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["requests", "descriptions"],
        help="Mode of simulation: 'requests' for user requests, 'descriptions' for movie descriptions",
    )

    parser.add_argument("--max_length", type=int, default=64, help="Maximum number of tokens to generate")
    parser.add_argument("--train_size", type=float, default=0.8, help="Proportion of data to use for training")
    parser.add_argument("--val_size", type=float, default=0.1, help="Proportion of data to use for validation")
    args = parser.parse_args()

    main(
        data_root=args.data_root,
        mode=args.mode,
        max_length=args.max_length,
        train_size=args.train_size,
        val_size=args.val_size,
    )
