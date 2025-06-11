from pathlib import Path

import pandas as pd

from utils.lr import CosineAnnealingWarmRestarts
from utils.misc import suppress_warnings

suppress_warnings()

import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import random

import torch
import yaml
from torch import optim
from torch.utils.data import DataLoader, Subset

import wandb
from data.collaborative_dataset import CollaborativeDataset, collaborative_collate_fn
from model.crecs import CRECS
from model.encoder import build_classifier, build_expander
from proccessor.collaborative import train
from utils.criterion import CollaborativeCriterion
from utils.misc import set_random_seed

# Load arguments from config file
args = yaml.safe_load(open("configs/collaborative.yaml", "r"))

device = "cuda" if torch.cuda.is_available() else "cpu"

set_random_seed(args["random_seed"])

# Create datasets and dataloaders
data_root = Path(args["data_root"])

train_dataset = CollaborativeDataset(data_root / "train_ratings.parquet", data_root / "train_requests.csv")
val_dataset = CollaborativeDataset(data_root / "val_ratings.parquet", data_root / "val_requests.csv")
train_subset = Subset(train_dataset, random.sample(range(len(train_dataset)), k=len(val_dataset)))

train_dataloader = DataLoader(
    train_dataset,
    batch_size=(batch_size := args["batch_size"]),
    shuffle=True,
    collate_fn=collaborative_collate_fn,
    num_workers=8,
    drop_last=True,
)

train_subset_dataloader = DataLoader(
    train_subset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collaborative_collate_fn,
    num_workers=8,
    drop_last=True,
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=collaborative_collate_fn,
    num_workers=8,
    drop_last=True,
)

# Create the model
args["classifier"]["num_classes"] = (num_items := pd.read_csv(data_root / "movies.csv")["item_id"].nunique())
args["model"]["recommender"]["num_items"] = num_items

classifier = build_classifier(**args["classifier"]).to(device)

model = CRECS(classifier=classifier, **args["model"]).to(device)

expander = build_expander(**args["expander"]).to(device)

# Define the optimizer and lr scheduler
optimizer = optim.AdamW(
    [
        {"name": "encoder", "params": model.encoder.parameters(), **args["optimizer"]["encoder"]},
        {"name": "recommender", "params": model.recommender.parameters(), **args["optimizer"]["recommender"]},
        {"name": "classifier", "params": model.classifier.parameters(), **args["optimizer"]["classifier"]},
        {"name": "expander", "params": expander.parameters(), **args["optimizer"]["expander"]},
    ],
    **args["optimizer"]["all"],
)

scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, **args["scheduler"])

# Define the loss function
criterion = CollaborativeCriterion(expander=expander, **args["criterion"])

# Begin logging & training
wandb.init(project="C-RECS", name=args["name"], tags=("Encoder", "Collaborative"), config=args)

train(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    train_dataloader=train_dataloader,
    train_subset_dataloader=train_subset_dataloader,
    val_dataloader=val_dataloader,
    device=device,
    **args["train"],
)

wandb.finish()
