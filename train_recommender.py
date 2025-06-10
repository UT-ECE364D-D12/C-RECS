import os
from pathlib import Path

import pandas as pd
import torch
import wandb
import yaml
from torch import optim
from torch.utils.data import DataLoader

from data.ratings_dataset import RatingsDataset, ratings_collate_fn
from model.recommender import DeepFM
from proccessor.recommender import train
from utils.criterion import RecommenderCriterion

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Load arguments from config file
args = yaml.safe_load(open("configs/recommender.yaml", "r"))

device = "cuda" if torch.cuda.is_available() else "cpu"

# Create datasets and dataloaders
data_root = Path(args["data_root"])

train_dataset = RatingsDataset(data_root / "train_ratings.parquet")
val_dataloader = RatingsDataset(data_root / "val_ratings.parquet")

train_dataloader = DataLoader(train_dataset, collate_fn=ratings_collate_fn, batch_size=args["batch_size"], num_workers=6, shuffle=True)
val_dataloader = DataLoader(val_dataloader, collate_fn=ratings_collate_fn, batch_size=args["batch_size"], num_workers=6)

# Create the model and optimizer
args["recommender"]["num_items"] = pd.read_csv(data_root / "movies.csv")["item_id"].nunique()

model = DeepFM(**args["recommender"]).to(device)

optimizer = optim.AdamW(model.parameters(), **args["optimizer"])

# Define the loss function
criterion = RecommenderCriterion()

# Begin logging & training
wandb.init(project="C-RECS", name=args["name"], tags=("Recommender",), config=args)

train(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_dataloader=train_dataloader,
    val_dataloder=val_dataloader,
    device=device,
    **args["train"],
)

wandb.finish()
