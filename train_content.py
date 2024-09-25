import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader

import wandb
from model.encoder import Encoder, build_classifier, build_expander
from utils.data import ContentDataset, train_test_split_requests
from utils.loss import EncoderCriterion
from utils.misc import set_random_seed
from utils.processor import train_content

args = yaml.safe_load(open("configs/content.yaml", "r"))

device = "cuda" if torch.cuda.is_available() else "cpu"

set_random_seed(args["random_seed"])

requests = pd.read_csv('data/ml-20m/requests.csv')

requests = requests.groupby("movie_id").agg({
    "movie_title": "first",
    "request": list,
}).reset_index()

requests.set_index("movie_id", inplace=True, drop=False)

train_requests, test_requests = train_test_split_requests(requests, train_size=0.8)

descriptions = pd.read_csv("data/ml-20m/descriptions.csv")

train_descriptions, test_descriptions = train_test_split(descriptions, train_size=0.8)

train_descriptions, test_descriptions = train_descriptions.reset_index(drop=True), test_descriptions.reset_index(drop=True)

train_dataset, test_dataset = ContentDataset(train_descriptions, train_requests), ContentDataset(test_descriptions, test_requests)

train_dataloader, test_dataloader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=4, drop_last=True), DataLoader(test_dataset, batch_size=args["batch_size"], num_workers=4, drop_last=True)

encoder = Encoder(**args["encoder"]).to(device)

expander = build_expander(embed_dim=768, width=2).to(device)

classifier = build_classifier(embed_dim=768, num_classes=requests["movie_id"].nunique()).to(device)

optimizer = optim.AdamW([
    {"params": encoder.parameters(), **args["optimizer"]["encoder"]},
    {"params": expander.parameters(), **args["optimizer"]["expander"]},
    {"params": classifier.parameters(), **args["optimizer"]["classifier"]},
])

criterion = EncoderCriterion(expander, classifier, loss_weights=args["loss_weights"])

wandb.init(project="MovieLens", name="ml-20m id", tags=("Encoder", "Content",), config=args)

train_content(
    encoder=encoder,
    optimizer=optimizer,
    criterion=criterion,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    device=device,
    **args["train"]
)

wandb.finish()

torch.save(encoder.state_dict(), "weights/encoder/encoder.pt")