
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader

import wandb
from model.encoder import Encoder, build_classifier, build_expander
from model.recommender import DeepFM
from utils.data import EncoderDataset, get_feature_sizes, train_test_split_requests
from utils.loss import EncoderRecommenderCriterion
from utils.processor import train_encoder

device = "cuda" if torch.cuda.is_available() else "cpu"

requests = pd.read_csv('data/requests.csv')

requests = requests.groupby("movie_id").agg({
    "movie_title": "first",
    "request": list,
}).reset_index()

train_requests, test_requests = train_test_split_requests(requests, train_size=0.8)

ratings = pd.read_csv("data/ml-100k/u.data", sep="\t", header=None, names=["user_id", "movie_id", "rating", "timestamp"])

train_ratings, test_ratings = train_test_split(ratings, train_size=0.8)

train_dataset, test_dataset = EncoderDataset(train_ratings, train_requests), EncoderDataset(test_ratings, test_requests)

train_dataloader, test_dataloader = DataLoader(train_dataset, batch_size=80, shuffle=True, num_workers=4, drop_last=True), DataLoader(test_dataset, batch_size=80, num_workers=4, drop_last=True)

encoder = Encoder().to(device)

recommender = DeepFM(feature_dims=get_feature_sizes(ratings), embed_dim=768, mlp_dims=(16, 16), dropout=0.8).to(device)

expander = build_expander(embed_dim=768, width=2).to(device)

classifier = build_classifier(embed_dim=768, num_classes=requests["movie_id"].nunique()).to(device)

optimizer = optim.AdamW(list(encoder.parameters()) + list(expander.parameters()) + list(classifier.parameters()) + list(recommender.parameters()), lr=0.0001)

loss_weights = {"triplet": 1.0, "mse": 1.0, "id": 0.5, "variance": 0.8, "invariance": 0.8, "covariance": 0.008}

criterion = EncoderRecommenderCriterion(expander, classifier, loss_weights=loss_weights)

wandb.init(project="MovieLens", name="Joint Training", tags=("Encoder",), config={"model": "Encoder", "optimizer": "AdamW", "lr": 0.0001, "loss_weights": loss_weights, "batch_size": 80})

train_encoder(
    encoder=encoder,
    recommender=recommender,
    optimizer=optimizer,
    criterion=criterion,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    max_epochs=10,
    device=device
)

wandb.finish()

torch.save(encoder.state_dict(), "weights/encoder/encoder.pt")