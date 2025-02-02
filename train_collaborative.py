from utils.lr import CosineAnnealingWarmRestarts
from utils.misc import suppress_warnings

suppress_warnings()

import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import random

import pandas as pd
import torch
import yaml
from torch import optim
from torch.utils.data import DataLoader, Subset

import wandb
from model.crecs import CRECS
from model.encoder import build_classifier, build_expander
from proccessor.collaborative import train
from utils.criterion import CollaborativeCriterion
from utils.data import CollaborativeDataset, collaborative_collate_fn, train_test_split_ratings, train_test_split_requests
from utils.misc import set_random_seed

# Load arguments from config file
args = yaml.safe_load(open("configs/collaborative.yaml", "r"))

device = "cuda" if torch.cuda.is_available() else "cpu"

set_random_seed(args["random_seed"])

# Load and split data
requests = pd.read_csv("data/ml-20m/requests.csv")
requests = requests.groupby("item_id").agg({"item_title": "first", "request": list}).reset_index()
requests.set_index("item_id", inplace=True, drop=False)

train_requests, test_requests = train_test_split_requests(requests, train_size=0.8)

ratings = pd.read_parquet("data/ml-20m/processed_ratings.parquet", engine="pyarrow")

train_ratings, test_ratings = train_test_split_ratings(ratings, train_size=0.8)

# Create datasets and dataloaders
train_dataset = CollaborativeDataset(train_ratings, train_requests)
test_dataset = CollaborativeDataset(test_ratings, test_requests)
train_subset = Subset(train_dataset, random.sample(range(len(train_dataset)), k=len(test_dataset)))

train_dataloader = DataLoader(
    train_dataset,
    batch_size=args["batch_size"],
    shuffle=True,
    collate_fn=collaborative_collate_fn,
    num_workers=8,
    drop_last=True,
)

train_subset_dataloader = DataLoader(
    train_subset,
    batch_size=args["batch_size"],
    shuffle=False,
    collate_fn=collaborative_collate_fn,
    num_workers=8,
    drop_last=True,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=args["batch_size"],
    collate_fn=collaborative_collate_fn,
    num_workers=8,
    drop_last=True,
)

# Create the model
classifier = build_classifier(num_classes=requests["item_id"].nunique(), **args["classifier"]).to(device)

args["model"]["recommender"]["num_items"] = requests["item_id"].nunique()
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
    **args["optimizer"]["all"]
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
    test_dataloader=test_dataloader,
    device=device,
    **args["train"]
)

wandb.finish()
