from utils.misc import suppress_warnings

suppress_warnings()

from pathlib import Path

import pandas as pd
import torch
import yaml
from torch import optim

import wandb
from criterion.joint_criterion import JointCriterion
from criterion.recommender_criterion import RecommenderCriterion
from data import build_content_dataloaders, build_rec_dataloaders, build_rec_eval_dataloader
from metrics.recommender_evaluator import RecommenderEvaluator
from model.crecs import CRECS
from model.encoder import build_classifier, build_expander
from model.recommender import DeepFM
from utils.lr import CosineAnnealingWarmRestarts
from utils.misc import set_random_seed
from utils.processor import train

args = yaml.safe_load(open("configs/recommender.yaml", "r"))

# Reproducibility
set_random_seed(args["random_seed"])

# Unified training script - can be used for recommender or content training
training_mode = str(args["training_mode"]).lower()

assert training_mode in ["recommender", "content"], f"Invalid training mode: {training_mode}"

# Create training and validation datasets
args["data"]["root"] = Path(args["data"]["root"])

if training_mode == "recommender":
    train_loader, val_loader = build_rec_dataloaders(**args["data"])
elif training_mode == "content":
    train_loader, val_loader = build_content_dataloaders(**args["data"])

# Create the model & optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_items = pd.read_csv(args["data"]["root"] / "movies.csv")["item_id"].nunique()

if training_mode == "recommender":
    args["recommender"]["num_items"] = num_items

    model = DeepFM(**args["recommender"]).to(device)
elif training_mode == "content":
    args["classifier"]["num_classes"] = num_items
    args["model"]["recommender"]["num_items"] = num_items

    classifier = build_classifier(**args["classifier"]).to(device)
    expander = build_expander(**args["expander"]).to(device)
    model = CRECS(classifier=classifier, **args["model"]).to(device)

# Create the optimizer
if training_mode == "recommender":
    optimizer = optim.AdamW([{"name": "recommender", "params": model.parameters()}], **args["optimizer"])
elif training_mode == "content":
    optimizer = optim.AdamW(
        [
            {"name": "encoder", "params": model.encoder.parameters(), **args["optimizer"]["encoder"]},
            {"name": "recommender", "params": model.recommender.parameters(), **args["optimizer"]["recommender"]},
            {"name": "classifier", "params": model.classifier.parameters(), **args["optimizer"]["classifier"]},
            {"name": "expander", "params": expander.parameters(), **args["optimizer"]["expander"]},
        ],
        **args["optimizer"]["all"],
    )

# Create the learning rate scheduler
scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, **args["scheduler"])

# Define the loss function
if training_mode == "recommender":
    criterion = RecommenderCriterion()
elif training_mode == "content":
    criterion = JointCriterion(expander=expander, **args["criterion"])


# Create evaluation jobs
if training_mode == "recommender":
    rec_eval_dataloader = build_rec_eval_dataloader(**args["data"])
    rec_eval = RecommenderEvaluator(**args["evaluator"])
    eval_jobs = [(model, rec_eval, rec_eval_dataloader)]
elif training_mode == "content":
    rec_eval_dataloader = build_rec_eval_dataloader(**args["data"])
    rec_eval = RecommenderEvaluator(**args["evaluator"])
    eval_jobs = [(model.recommender, rec_eval, rec_eval_dataloader)]

# Begin logging & training
wandb.init(project="C-RECS", name=args["name"], tags=(args["training_mode"],), config=args)

train(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    eval_jobs=eval_jobs,
    device=device,
    **args["train"],
)

wandb.finish()
