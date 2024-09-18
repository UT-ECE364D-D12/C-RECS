import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader

import wandb
from model.recommender import DeepFM
from utils.data import RatingsDataset, get_feature_sizes
from utils.loss import RecommenderCriterion
from utils.processor import train

device = "cuda" if torch.cuda.is_available() else "cpu"

ratings = pd.read_csv("data/ml-100k/u.data", sep="\t", header=None, names=["user_id", "movie_id", "rating", "timestamp"])

train_ratings, test_ratings = train_test_split(ratings, train_size=0.8)

train_dataset, test_dataset = RatingsDataset(train_ratings), RatingsDataset(test_ratings)

train_dataloader, test_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True), DataLoader(test_dataset, batch_size=1024)

model = DeepFM(feature_dims=get_feature_sizes(ratings), embed_dim=768, mlp_dims=(768, 768), dropout=0.8).to(device)

optimizer = optim.AdamW(model.parameters(), lr=0.001)

criterion = RecommenderCriterion()

wandb.init(project="MovieLens", name="DeepFM Embed Dim=768", tags=("Recommender",), config={"model": "DeepFM", "optimizer": "AdamW", "lr": 0.001, "Dropout": 0.8, "loss": "MSE"})

train(
    model=model,
    optimizer=optimizer,
    criterion=criterion, 
    train_dataloader=train_dataloader, 
    test_dataloader=test_dataloader,
    max_epochs=10, 
    device=device
)

wandb.finish()

torch.save(model.state_dict(), "weights/recommender/pretrained-deepfm.pt")