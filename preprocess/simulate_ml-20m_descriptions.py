import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import argparse
import gc
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from preprocess.utils import SimulatorDataset, build_language_model, simulate

MODEL_NAMES = ["meta-llama/Llama-3.3-70B-Instruct"]
BATCH_SIZES = [48]

# Prompt for generating descriptions
PROMPTS = [
    lambda movie: f"""You are interacting with a movie recommendation system. Your goal is to create a description for the movie "{movie}". Write a single concise paragraph that weaves together the genre, plot/theme, mood/tone, key cast, director, release year, country of origin, and critical acclaim.\nFor example, for the movie “Inception,” you might say:\n“Inception is a 2010 sci-fi thriller/heist from director Christopher Nolan that unfolds across mind-bending, multi-layered dreams in contemporary urban settings. Leonardo DiCaprio leads a stellar cast (with Joseph Gordon-Levitt, Ellen Page, Tom Hardy, and Marion Cotillard) as an expert “extractor” who assembles a team to plant an idea in a target’s subconscious, blending tense, cerebral suspense with blockbuster action. A UK/USA co-production shot with crisp, high-contrast visuals, it won four Academy Awards (Visual Effects, Sound Editing, Sound Mixing, and Cinematography).\nReply ONLY with the description for {movie}. DO NOT include any other text.""",
]


def main(data_root: Path, max_length: int = 64) -> None:
    # Read in the item data
    movies = pd.read_csv(data_root / "movies.csv", header=0, names=["item_id", "item_title", "genres"])[["item_id", "item_title"]]

    data = pd.DataFrame(columns=["item_id", "item_title", "description"])

    for model_name, batch_size in zip(MODEL_NAMES, BATCH_SIZES):
        print(f"Loading {model_name}...")

        # Load the model and tokenizer
        model, tokenizer = build_language_model(model_name)

        # Create the dataset & dataloader
        dataset = SimulatorDataset(movies, tokenizer, PROMPTS)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=6)

        # Simulate the requests
        model_data = simulate(model, tokenizer, dataloader, max_length=max_length).rename(columns={"text": "description"})

        data = pd.concat([data, model_data], ignore_index=True)

        del model
        del tokenizer

        gc.collect()
        torch.cuda.empty_cache()

    data.to_csv(data_root / f"descriptions.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path, help="Root directory of MovieLens data")
    parser.add_argument("--max_length", type=int, default=64, help="Maximum number of tokens to generate")
    args = parser.parse_args()

    main(
        data_root=args.data_root,
        max_length=args.max_length,
    )
