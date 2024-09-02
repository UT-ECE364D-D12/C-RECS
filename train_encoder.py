
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader

import wandb
from model.encoder import build_encoder, build_expander
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

encoder, tokenizer = build_encoder(device=device)

expander = build_expander(embed_dim=768).to(device)

criterion = EncoderCriterion(expander)

