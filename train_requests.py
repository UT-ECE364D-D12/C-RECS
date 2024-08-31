import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader

import wandb
from model.request import RequestDecoder
from utils.data import RequestsDataset, get_feature_sizes
from utils.loss import RequestCriterion
from utils.processor import train

device = "cuda" if torch.cuda.is_available() else "cpu"

requests = pd.read_hdf('data/requests.h5')

train_requests, test_requests = train_test_split(requests, train_size=0.8)

train_dataset, test_dataset = RequestsDataset(train_requests), RequestsDataset(test_requests)

train_dataloader, test_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True), DataLoader(test_dataset, batch_size=1024)

model = RequestDecoder(num_movies=requests["movie_id"].nunique()).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

criterion = RequestCriterion()

wandb.init(project="MovieLens", name="One Request", tags=("Requests"), config={"model": "DeepFM", "optimizer": "AdamW", "lr": 0.001, "Dropout": 0.8, "loss": "MSE"})

train(
    model=model,
    optimizer=optimizer,
    criterion=criterion, 
    train_dataloader=train_dataloader, 
    test_dataloader=test_dataloader,
    max_epochs=10, 
    device=device
)