import os
import random

import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader, Subset

import wandb
from model.encoder import Encoder, build_classifier, build_expander
from model.recommender import DeepFM
from proccessor.collaborative import train
from utils.data import CollaborativeDataset, get_feature_sizes, train_test_split_requests
from utils.loss import JointCriterion
from utils.misc import set_random_seed

args = yaml.safe_load(open("configs/collaborative.yaml", "r"))

device = "cuda" if torch.cuda.is_available() else "cpu"

set_random_seed(args["random_seed"])

# Load requests
requests = pd.read_csv('data/ml-20m/requests.csv')
requests = requests.groupby("movie_id").agg({
    "movie_title": "first",
    "request": list,
}).reset_index()
requests.set_index("movie_id", inplace=True, drop=False)

ratings = pd.read_csv("data/ml-20m/ratings.csv", header=0, names=["user_id", "movie_id", "rating", "timestamp"]).astype({"user_id": int, "movie_id": int, "rating": float, "timestamp": int})

user_id_to_unique_id = {user_id: i for i, user_id in enumerate(ratings["user_id"].unique())}
item_id_to_unique_id = {movie_id: i for i, movie_id in enumerate(requests["movie_id"].unique())}

train_requests, test_requests = train_test_split_requests(requests, train_size=0.8)
train_ratings, test_ratings = train_test_split(ratings, train_size=0.8)

train_dataset = CollaborativeDataset(train_ratings, train_requests, user_id_to_unique_id, item_id_to_unique_id)
train_dataloader= DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=4, drop_last=True)

test_dataset = CollaborativeDataset(test_ratings, test_requests, user_id_to_unique_id, item_id_to_unique_id)
test_dataloader = DataLoader(test_dataset, batch_size=args["batch_size"], num_workers=4, drop_last=True)

subset_indices = random.sample(range(len(train_dataset)), k=len(test_dataset))
train_subset = Subset(train_dataset, subset_indices)
train_subset_dataloader = DataLoader(train_subset, batch_size=args["batch_size"], shuffle=False, num_workers=4, drop_last=True)

encoder = Encoder(**args["encoder"]).to(device)

recommender = DeepFM(feature_dims=get_feature_sizes(ratings), **args["recommender"]).to(device)

expander = build_expander(embed_dim=encoder.embed_dim, **args["expander"]).to(device)

classifier = build_classifier(embed_dim=encoder.embed_dim, num_classes=requests["movie_id"].nunique(), **args["classifier"]).to(device)

optimizer = optim.AdamW([
    {"params": encoder.parameters(), **args["optimizer"]["encoder"]},
    {"params": expander.parameters(), **args["optimizer"]["expander"]},
    {"params": classifier.parameters(), **args["optimizer"]["classifier"]},
    {"params": recommender.parameters(), **args["optimizer"]["recommender"]},
])

criterion = JointCriterion(expander=expander, **args["criterion"])

wandb.init(project="MovieLens", name=args["name"], tags=("Encoder", "Collaborative"), config=args)

train(
    encoder=encoder,
    classifier=classifier,
    recommender=recommender,
    optimizer=optimizer,
    criterion=criterion,
    train_dataloader=train_dataloader,
    train_subset_dataloader=train_subset_dataloader,
    test_dataloader=test_dataloader,
    device=device,
    **args["train"]
)

wandb.finish()

os.makedirs("weights/collaborative", exist_ok=True)
torch.save(recommender.state_dict(), "weights/collaborative/deepfm.pt")
torch.save(encoder.state_dict(), "weights/collaborative/encoder.pt")