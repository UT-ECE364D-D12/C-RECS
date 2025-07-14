from pathlib import Path

import pandas as pd
import torch
import yaml
from torch import optim
from torch.utils.data import DataLoader

import wandb
from criterion.recommender_criterion import RecommenderCriterion
from data.recommender_dataset import RatingsRecommendationDataset, UserRecommendationDataset, ratings_collate_fn, user_collate_fn
from metrics.recommender_evaluator import RecommenderEvaluator
from model.recommender import DeepFM
from utils.lr import CosineAnnealingWarmRestarts
from utils.misc import set_random_seed
from utils.processor import train

# Load arguments from config file
args = yaml.safe_load(open("configs/recommender.yaml", "r"))

device = "cuda" if torch.cuda.is_available() else "cpu"

set_random_seed(args["random_seed"])

# Create datasets and dataloaders
data_root = Path(args["data_root"])

train_dataset = RatingsRecommendationDataset(data_root / "train_ratings.parquet")
val_dataset = RatingsRecommendationDataset(data_root / "val_ratings.parquet")
eval_dataset = UserRecommendationDataset(data_root / "val_ratings.parquet", history_size=0.8)

train_dataloader = DataLoader(train_dataset, collate_fn=ratings_collate_fn, batch_size=args["batch_size"], shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, collate_fn=ratings_collate_fn, batch_size=args["batch_size"], shuffle=False, num_workers=0)
eval_dataloader = DataLoader(eval_dataset, collate_fn=user_collate_fn, batch_size=1, shuffle=False, num_workers=0)

# Create the model and optimizer
args["recommender"]["num_items"] = pd.read_csv(data_root / "movies.csv")["item_id"].nunique()

model = DeepFM(**args["recommender"]).to(device)

optimizer = optim.AdamW([{"name": "recommender", "params": model.parameters()}], **args["optimizer"])

scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, **args["scheduler"])

# Define the loss function
criterion = RecommenderCriterion()

# Define the evaluation jobs
evaluator = RecommenderEvaluator(**args["evaluator"])

eval_jobs = [(model, evaluator, eval_dataloader)]

# Begin logging & training
wandb.init(project="C-RECS", name=args["name"], tags=("Recommender",), config=args)

train(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    eval_jobs=eval_jobs,
    device=device,
    **args["train"],
)

wandb.finish()
