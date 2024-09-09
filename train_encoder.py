
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader

import wandb
from model.encoder import Encoder, build_classifier, build_expander
from utils.data import EncoderDataset
from utils.loss import EncoderCriterion
from utils.processor import train_encoder

device = "cuda" if torch.cuda.is_available() else "cpu"

requests = pd.read_hdf('data/requests.h5')

request = requests.groupby("movie_id").agg({
    "movie_title": "first",
    "request": list,
    "encoded_request": list
}).reset_index()

train_requests, test_requests = train_test_split(request, train_size=0.8)

train_dataset, test_dataset = EncoderDataset(train_requests), EncoderDataset(test_requests)

train_dataloader, test_dataloader = DataLoader(train_dataset, batch_size=80, shuffle=True, num_workers=4, drop_last=True), DataLoader(test_dataset, batch_size=80, num_workers=4, drop_last=True)

encoder = Encoder().to(device)

expander = build_expander(embed_dim=768, width=2).to(device)

classifier = build_classifier(embed_dim=768, num_classes=requests["movie_id"].nunique()).to(device)

optimizer = optim.AdamW(list(encoder.parameters()) + list(expander.parameters()) + list(classifier.parameters()), lr=0.0001)

loss_weights = {"triplet": 1.0, "id": 0.5, "variance": 0.8, "invariance": 0.8, "covariance": 0.0008}

criterion = EncoderCriterion(expander, classifier, loss_weights=loss_weights)

wandb.init(project="MovieLens", name="VICReg Weight Decay", tags=("Encoder",), config={"model": "Encoder", "optimizer": "AdamW", "lr": 0.0001, "loss_weights": loss_weights, "batch_size": 80})

train_encoder(
    model=encoder,
    optimizer=optimizer,
    criterion=criterion,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    max_epochs=10,
)
