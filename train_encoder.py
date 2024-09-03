
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader

import wandb
from model.encoder import Encoder, build_expander
from utils.data import EncoderDataset
from utils.loss import EncoderCriterion
from utils.processor import train

device = "cuda" if torch.cuda.is_available() else "cpu"

requests = pd.read_hdf('data/requests.h5')

request = requests.groupby("movie_id").agg({
    "movie_title": "first",
    "request": list,
    "encoded_request": list
}).reset_index()

train_requests, test_requests = train_test_split(request, train_size=0.8)

train_dataset, test_dataset = EncoderDataset(train_requests), EncoderDataset(test_requests)

train_dataloader, test_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True), DataLoader(test_dataset, batch_size=1024)

encoder = Encoder().to(device)

expander = build_expander(embed_dim=768).to(device)

optimizer = optim.AdamW(list(encoder.parameters()) + list(expander.parameters()), lr=0.001)

loss_weights = {"triplet": 1.0, "variance": 0.0, "invariance": 0.0, "covariance": 0.0}

criterion = EncoderCriterion(expander, loss_weights=loss_weights)

wandb.init(project="MovieLens", name="No VICReg", tags=("Encoder",), config={"model": "Encoder", "optimizer": "AdamW", "lr": 0.001, "loss_weights": loss_weights})

train(
    model=encoder,
    optimizer=optimizer,
    criterion=criterion,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    max_epochs=10,
    device=device
)
