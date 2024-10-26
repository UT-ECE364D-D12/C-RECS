import os

import pandas as pd
import torch
import yaml
from torch import optim
from torch.utils.data import DataLoader

import wandb
from model.recommender import DeepFM
from proccessor.recommender import train
from utils.data import RatingsDataset, ratings_collate_fn, train_test_split_ratings
from utils.loss import RecommenderCriterion

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

args = yaml.safe_load(open("configs/recommender.yaml", "r"))

device = "cuda" if torch.cuda.is_available() else "cpu"

ratings = pd.read_hdf("data/ml-20m/processed_ratings.hdf")

train_ratings, test_ratings = train_test_split_ratings(ratings, train_size=0.8)

train_dataset, test_dataset = RatingsDataset(train_ratings), RatingsDataset(test_ratings)

train_dataloader, test_dataloader = DataLoader(train_dataset, collate_fn=ratings_collate_fn, batch_size=args["batch_size"], shuffle=True, num_workers=12), DataLoader(test_dataset, collate_fn=ratings_collate_fn, batch_size=args["batch_size"], num_workers=12)

model = DeepFM(num_items=ratings["item_id"].nunique(), **args["recommender"]).to(device)

optimizer = optim.AdamW(model.parameters(), **args["optimizer"])

criterion = RecommenderCriterion()

wandb.init(project="MovieLens", name=args["name"], tags=("Recommender",), config=args)

train(
    model=model,
    optimizer=optimizer,
    criterion=criterion, 
    train_dataloader=train_dataloader, 
    test_dataloader=test_dataloader,
    device=device,
    **args["train"],
)

wandb.finish()

os.makedirs("weights/recommender", exist_ok=True)
torch.save(model.state_dict(), "weights/recommender/deepfm.pt")