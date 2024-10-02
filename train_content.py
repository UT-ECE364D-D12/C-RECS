import pandas as pd
import torch
import yaml
from torch import optim
from torch.utils.data import DataLoader

import wandb
from model.encoder import Encoder, build_classifier, build_expander
from utils.content import train_content
from utils.data import ContentDataset, DescriptionsDataset, train_test_split_requests
from utils.loss import EncoderCriterion
from utils.misc import set_random_seed

args = yaml.safe_load(open("configs/content.yaml", "r"))

device = "cuda" if torch.cuda.is_available() else "cpu"

set_random_seed(args["random_seed"])

# Load requests
requests = pd.read_csv('data/ml-20m/requests.csv')
requests = requests.groupby("movie_id").agg({
    "movie_title": "first",
    "request": list,
}).reset_index()
requests.set_index("movie_id", inplace=True, drop=False)

# Load descriptions
descriptions = pd.read_csv("data/ml-20m/descriptions.csv")
descriptions.set_index("movie_id", inplace=True, drop=False)

train_requests, test_requests = train_test_split_requests(requests, train_size=0.8)

train_dataset = ContentDataset(descriptions, train_requests)
test_dataset = ContentDataset(descriptions, test_requests)
descriptions_dataset = DescriptionsDataset(descriptions)

train_dataloader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=4)
descriptions_dataloader = DataLoader(descriptions_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=4)

encoder = Encoder(**args["encoder"]).to(device)

expander = build_expander(embed_dim=encoder.embed_dim, width=2).to(device)

classifier = build_classifier(embed_dim=encoder.embed_dim, num_classes=requests["movie_id"].nunique()).to(device)

optimizer = optim.AdamW([
    {"params": encoder.parameters(), **args["optimizer"]["encoder"]},
    {"params": expander.parameters(), **args["optimizer"]["expander"]},
    {"params": classifier.parameters(), **args["optimizer"]["classifier"]},
])

criterion = EncoderCriterion(expander, **args["criterion"])

wandb.init(project="MovieLens", name=args["name"], tags=("Encoder", "Content",), config=args)

train_content(
    encoder=encoder,
    classifier=classifier,
    optimizer=optimizer,
    criterion=criterion,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    descriptions_dataloader=descriptions_dataloader,
    device=device,
    **args["train"]
)

wandb.finish()

torch.save(encoder.state_dict(), "weights/encoder/encoder.pt")