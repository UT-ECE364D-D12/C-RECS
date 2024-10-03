import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader

import wandb
from model.recommender import DeepFM
from utils.data import RatingsDataset, get_feature_sizes
from utils.loss import RecommenderCriterion
from utils.recommender import train

args = yaml.safe_load(open("configs/recommender.yaml", "r"))

device = "cuda" if torch.cuda.is_available() else "cpu"

ratings = pd.read_csv("data/ml-20m/ratings.csv", header=0, names=["user_id", "movie_id", "rating", "timestamp"])

train_ratings, test_ratings = train_test_split(ratings, train_size=0.8)

train_dataset, test_dataset = RatingsDataset(train_ratings), RatingsDataset(test_ratings)

train_dataloader, test_dataloader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True), DataLoader(test_dataset, batch_size=args["batch_size"])

model = DeepFM(feature_dims=get_feature_sizes(ratings), **args["recommender"]).to(device)

optimizer = optim.AdamW(model.parameters(), *args["optimizer"])

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

torch.save(model.state_dict(), "weights/recommender/deepfm.pt")