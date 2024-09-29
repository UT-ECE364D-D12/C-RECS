import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader

import wandb
from model.encoder import Encoder, build_classifier, build_expander
from model.recommender import DeepFM
from utils.data import CollaborativeDataset, get_feature_sizes, train_test_split_requests
from utils.loss import JointCriterion
from utils.misc import set_random_seed
from utils.processor import train_collaborative

args = yaml.safe_load(open("configs/collaborative.yaml", "r"))

device = "cuda" if torch.cuda.is_available() else "cpu"

set_random_seed(args["random_seed"])

requests = pd.read_csv('data/ml-20m/requests.csv')

requests = requests.groupby("movie_id").agg({
    "movie_title": "first",
    "request": list,
}).reset_index()

requests.set_index("movie_id", inplace=True, drop=False)

train_requests, test_requests = train_test_split_requests(requests, train_size=0.8)

ratings = pd.read_csv("data/ml-20m/ratings.csv", header=0, names=["user_id", "movie_id", "rating", "timestamp"]).astype({"user_id": int, "movie_id": int, "rating": float, "timestamp": int})

train_ratings, test_ratings = train_test_split(ratings, train_size=0.8)

train_ratings, test_ratings = train_ratings.reset_index(drop=True), test_ratings.reset_index(drop=True)

train_dataset, test_dataset = CollaborativeDataset(train_ratings, train_requests), CollaborativeDataset(test_ratings, test_requests)

train_dataloader, test_dataloader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=4, drop_last=True), DataLoader(test_dataset, batch_size=args["batch_size"], num_workers=4, drop_last=True)

encoder = Encoder(**args["encoder"]).to(device)

recommender = DeepFM(feature_dims=get_feature_sizes(ratings), **args["recommender"]).to(device)

expander = build_expander(embed_dim=encoder.embed_dim, width=2).to(device)

classifier = build_classifier(embed_dim=encoder.embed_dim, num_classes=requests["movie_id"].nunique()).to(device)

optimizer = optim.AdamW([
    {"params": encoder.parameters(), **args["optimizer"]["encoder"]},
    {"params": expander.parameters(), **args["optimizer"]["expander"]},
    {"params": classifier.parameters(), **args["optimizer"]["classifier"]},
    {"params": recommender.parameters(), **args["optimizer"]["recommender"]},
])

criterion = JointCriterion(loss_weights=args["loss_weights"], expander=expander, classifier=classifier)

wandb.init(project="MovieLens", name="ml-20m", tags=("Encoder", "Collaborative"), config=args)

train_collaborative(
    encoder=encoder,
    recommender=recommender,
    optimizer=optimizer,
    criterion=criterion,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    device=device,
    **args["train"]
)

wandb.finish()

torch.save(recommender.state_dict(), "weights/recommender/deepfm.pt")
torch.save(encoder.state_dict(), "weights/encoder/encoder.pt")