import os
import random
import warnings

import pandas as pd
import torch
import yaml
from torch import optim
from torch.utils.data import DataLoader, Subset

import wandb
from model.encoder import Encoder, build_classifier, build_expander
from model.recommender import DeepFM
from proccessor.collaborative import train
from utils.data import CollaborativeDataset, collaborative_collate_fn, train_test_split_ratings, train_test_split_requests
from utils.loss import JointCriterion
from utils.misc import set_random_seed

warnings.filterwarnings("ignore")

args = yaml.safe_load(open("configs/collaborative.yaml", "r"))

device = "cuda" if torch.cuda.is_available() else "cpu"

set_random_seed(args["random_seed"])
 
requests = pd.read_csv('data/ml-20m/requests.csv')
requests = requests.groupby("item_id").agg({
    "item_title": "first",
    "request": list,
}).reset_index()
requests.set_index("item_id", inplace=True, drop=False)

train_requests, test_requests = train_test_split_requests(requests, train_size=0.8)

ratings = pd.read_hdf("data/ml-20m/processed_ratings.hdf")

train_ratings, test_ratings = train_test_split_ratings(ratings, train_size=0.8)

train_dataset = CollaborativeDataset(train_ratings, train_requests)
test_dataset = CollaborativeDataset(test_ratings, test_requests)
train_subset = Subset(train_dataset, random.sample(range(len(train_dataset)), k=len(test_dataset)))

train_dataloader= DataLoader(
    train_dataset, 
    batch_size=args["batch_size"], 
    shuffle=True,
    collate_fn=collaborative_collate_fn, 
    num_workers=6,
    drop_last=True
)

test_dataloader = DataLoader(
    test_dataset, 
    batch_size=args["batch_size"], 
    collate_fn=collaborative_collate_fn,
    num_workers=6, 
    drop_last=True
)

train_subset_dataloader = DataLoader(
    train_subset, 
    batch_size=args["batch_size"], 
    shuffle=False, 
    collate_fn=collaborative_collate_fn, 
    num_workers=6, 
    drop_last=True
)

encoder = Encoder(**args["encoder"]).to(device)

expander = build_expander(embed_dim=encoder.embed_dim, **args["expander"]).to(device)

classifier = build_classifier(embed_dim=encoder.embed_dim, num_classes=requests["item_id"].nunique(), **args["classifier"]).to(device)

recommender = DeepFM(num_items=ratings["item_id"].nunique(), **args["recommender"]).to(device)

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
torch.save(encoder.state_dict(), "weights/collaborative/encoder.pt")
torch.save(recommender.state_dict(), "weights/collaborative/deepfm.pt")

item_embeddings = recommender.embedding.item_embedding.weight

torch.save(item_embeddings, "weights/collaborative/item_embeddings.pt")