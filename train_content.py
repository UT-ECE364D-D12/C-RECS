import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

from utils.misc import suppress_warnings

suppress_warnings()

from pathlib import Path

import pandas as pd
import torch
import yaml
from torch import optim
from torch.utils.data import DataLoader

import wandb
from criterion.joint_criterion import JointCriterion
from data.content_dataset import ContentDataset, content_collate_fn
from data.recommender_dataset import UserRecommendationDataset, user_collate_fn
from metrics.recommender_evaluator import RecommenderEvaluator
from model.crecs import CRECS
from model.encoder import build_classifier, build_expander
from utils.lr import CosineAnnealingWarmRestarts
from utils.misc import set_random_seed
from utils.processor import train

# Load arguments from config file
args = yaml.safe_load(open("configs/content.yaml", "r"))

device = "cuda" if torch.cuda.is_available() else "cpu"

set_random_seed(args["random_seed"])

# Create datasets and dataloaders
data_root = Path(args["data_root"])

train_dataset = ContentDataset(data_root / "train_ratings.parquet", data_root / "descriptions.csv")
val_dataset = ContentDataset(data_root / "val_ratings.parquet", data_root / "descriptions.csv")
rec_eval_dataset = UserRecommendationDataset(data_root / "val_ratings.parquet", history_size=0.8)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=(batch_size := args["batch_size"]),
    collate_fn=content_collate_fn,
    shuffle=True,
    num_workers=0,
    drop_last=True,
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=content_collate_fn,
    shuffle=False,
    num_workers=0,
    drop_last=True,
)

rec_eval_dataloader = DataLoader(
    rec_eval_dataset,
    batch_size=1,
    collate_fn=user_collate_fn,
    shuffle=False,
    num_workers=0,
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
criterion = JointCriterion(expander=expander, **args["criterion"])

# Define the evaluation jobs
rec_evaluator = RecommenderEvaluator(**args["evaluator"])

eval_jobs = [(model.recommender, rec_evaluator, rec_eval_dataloader)]

# Begin logging & training
wandb.init(project="C-RECS", name=args["name"], tags=("Encoder", "Content"), config=args)

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
