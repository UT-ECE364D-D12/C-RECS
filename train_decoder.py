import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader

import wandb
from model.layers import MultiLayerPerceptron
from utils.data import DecoderDataset
from utils.loss import DecoderCriterion
from utils.processor import train

device = "cuda" if torch.cuda.is_available() else "cpu"

requests = pd.read_hdf('data/requests.h5')

train_requests, test_requests = train_test_split(requests, train_size=0.8)

train_dataset, test_dataset = DecoderDataset(train_requests), DecoderDataset(test_requests)

train_dataloader, test_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True), DataLoader(test_dataset, batch_size=1024)

model = MultiLayerPerceptron(input_dim=requests["movie_id"].nunique(), hidden_dims=[32, 32], output_dim=requests["movie_id"].nunique(), dropout=0.8).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

criterion = DecoderCriterion()

wandb.init(project="MovieLens", name="Two Requests", tags=("Decoder",), config={"model": "DeepFM", "optimizer": "AdamW", "lr": 0.001, "Dropout": 0.8, "loss": "MSE"})

train(
    model=model,
    optimizer=optimizer,
    criterion=criterion, 
    train_dataloader=train_dataloader, 
    test_dataloader=test_dataloader,
    max_epochs=10, 
    device=device
)