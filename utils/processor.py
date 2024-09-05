import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from model.encoder import Encoder
from utils.loss import Criterion, EncoderCriterion


def train_one_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: Criterion,
    dataloader: DataLoader,
    epoch: int,
    device: str = "cpu",
) -> None:
    criterion.reset_metrics()
    losses = {}

    model.train()

    for features, targets in tqdm(dataloader, desc=f"Training (Epoch {epoch})"):
        optimizer.zero_grad()

        features, targets = features.to(device), targets.to(device)

        predictions = model(features)

        batch_losses = criterion(predictions, targets)

        wandb.log({"Train": {"Loss": batch_losses}}, step=wandb.run.step + len(targets))

        losses = {k: losses.get(k, 0) + v.item() for k, v in batch_losses.items()}
        
        loss = batch_losses["overall"]

        loss.backward()
        optimizer.step()

def train_encoder_one_epoch(
    model: Encoder,
    optimizer: optim.Optimizer,
    criterion: EncoderCriterion,
    dataloader: DataLoader,
    epoch: int,
) -> None:
    criterion.reset_metrics()
    losses = {}

    model.train()

    for anchor, positive, negative in tqdm(dataloader, desc=f"Training (Epoch {epoch})"):
        optimizer.zero_grad()

        anchor_requests, anchor_ids = anchor 
        positive_requests, positive_ids = positive 
        negative_requests, negative_ids = negative 

        anchor_embeddings = model(anchor_requests)
        positive_embeddings = model(positive_requests)
        negative_embeddings = model(negative_requests)

        batch_losses = criterion((anchor_embeddings, anchor_ids), (positive_embeddings, positive_ids), (negative_embeddings, negative_ids))

        wandb.log({"Train": {"Loss": batch_losses}}, step=wandb.run.step + len(anchor_requests))

        losses = {k: losses.get(k, 0) + v.item() for k, v in batch_losses.items()}
        
        loss = batch_losses["overall"]

        loss.backward()
        optimizer.step()

    metrics = criterion.get_metrics()

    wandb.log({"Train": {"Metric": metrics}}, step=wandb.run.step)

def evaluate_one_epoch(
    model: nn.Module,
    criterion: Criterion,
    dataloader: DataLoader,
    epoch: int,
    device: str = "cpu",
) -> None:
    criterion.reset_metrics()
    losses = {}

    model.eval()

    with torch.no_grad():
        for features, targets in tqdm(dataloader, desc=f"Validation (Epoch {epoch})"):
            features, targets = features.to(device), targets.to(device)

            predictions = model(features)

            batch_losses = criterion(predictions, targets)

            losses = {k: losses.get(k, 0) + v.item() for k, v in batch_losses.items()}

        wandb.log({"Validation": {"Loss": losses}}, step=wandb.run.step)
        
def evaluate_encoder_one_epoch(
    model: nn.Module,
    criterion: Criterion,
    dataloader: DataLoader,
    epoch: int,
) -> None:
    criterion.reset_metrics()
    losses = {}

    model.eval()

    with torch.no_grad():
        for anchor, positive, negative in tqdm(dataloader, desc=f"Validation (Epoch {epoch})"):
            anchor_requests, anchor_ids = anchor 
            positive_requests, positive_ids = positive 
            negative_requests, negative_ids = negative 

            anchor_embeddings = model(anchor_requests)
            positive_embeddings = model(positive_requests)
            negative_embeddings = model(negative_requests)

            batch_losses = criterion((anchor_embeddings, anchor_ids), (positive_embeddings, positive_ids), (negative_embeddings, negative_ids))

            losses = {k: losses.get(k, 0) + v.item() for k, v in batch_losses.items()}

    metrics = criterion.get_metrics()

    wandb.log({"Validation": {"Loss": losses, "Metric": metrics}}, step=wandb.run.step)

def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: Criterion,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    max_epochs: int,
    device: str = "cpu",
) -> None:
    for epoch in range(max_epochs):
        train_one_epoch(model, optimizer, criterion, train_dataloader, epoch, device)
        
        evaluate_one_epoch(model, criterion, test_dataloader, epoch, device)

def train_encoder(
    model: Encoder,
    optimizer: optim.Optimizer,
    criterion: EncoderCriterion,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    max_epochs: int,
) -> None:
    for epoch in range(max_epochs):
        train_encoder_one_epoch(model, optimizer, criterion, train_dataloader, epoch)
        
        evaluate_encoder_one_epoch(model, criterion, test_dataloader, epoch)