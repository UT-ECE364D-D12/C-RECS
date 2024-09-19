
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader

import wandb
from model.encoder import Encoder, build_classifier, build_expander
from model.recommender import DeepFM
from utils.data import EncoderDataset, get_feature_sizes, train_test_split_requests
from utils.loss import EncoderRecommenderCriterion
from utils.misc import set_random_seed
from utils.processor import train_encoder

args = yaml.safe_load(open("configs/encoder.yaml", "r"))

device = "cuda" if torch.cuda.is_available() else "cpu"

set_random_seed(args["random_seed"])

requests = pd.read_csv('data/ml-20m/requests.csv')

requests = requests.groupby("movie_id").agg({
    "movie_title": "first",
    "request": list,
}).reset_index()

train_requests, test_requests = train_test_split_requests(requests, train_size=0.8)

ratings = pd.read_csv("data/ml-20m/ratings.csv", header=0, names=["user_id", "movie_id", "rating", "timestamp"])

train_ratings, test_ratings = train_test_split(ratings, train_size=0.8)

train_dataset, test_dataset = EncoderDataset(train_ratings, train_requests), EncoderDataset(test_ratings, test_requests)

train_dataloader, test_dataloader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=4, drop_last=True), DataLoader(test_dataset, batch_size=args["batch_size"], num_workers=4, drop_last=True)

encoder = Encoder(**args["encoder"]).to(device)

recommender = DeepFM(feature_dims=get_feature_sizes(ratings), **args["recommender"]).to(device)

expander = build_expander(embed_dim=768, width=2).to(device)

classifier = build_classifier(embed_dim=768, num_classes=requests["movie_id"].nunique()).to(device)

optimizer = optim.AdamW([
    {"params": encoder.parameters(), **args["optimizer"]["encoder"]},
    {"params": expander.parameters(), **args["optimizer"]["expander"]},
    {"params": classifier.parameters(), **args["optimizer"]["classifier"]},
    {"params": recommender.parameters(), **args["optimizer"]["recommender"]},
])

criterion = EncoderRecommenderCriterion(expander, classifier, loss_weights=args["loss_weights"])

wandb.init(project="MovieLens", name="Pretrained Recommender ", tags=("Encoder",), config=args)

train_encoder(
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