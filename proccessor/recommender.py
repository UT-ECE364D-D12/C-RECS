import os
from typing import Dict, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from model.recommender import DeepFM
from utils.loss import RecommenderCriterion
from utils.misc import send_to_device


def train_one_epoch(
    model: DeepFM,
    optimizer: optim.Optimizer,
    criterion: RecommenderCriterion,
    dataloader: DataLoader,
    epoch: int,
    device: str = "cpu",
) -> None:
    losses = {}

    model.train()

    for features, targets in tqdm(dataloader, desc=f"Training (Epoch {epoch})", dynamic_ncols=True):
        optimizer.zero_grad()

        features, targets = send_to_device(features, device), send_to_device(targets, device)

        predictions = model(features)

        batch_losses = criterion(predictions, targets)

        wandb.log({"Train": {"Loss": batch_losses}}, step=wandb.run.step + len(predictions))

        losses = {k: losses.get(k, 0) + v.item() for k, v in batch_losses.items()}
        
        loss = batch_losses["overall"]

        loss.backward()
        optimizer.step()

def evaluate(
    model: nn.Module,
    criterion: RecommenderCriterion,
    dataloader: DataLoader,
    epoch: int,
    device: str = "cpu",
) -> Dict[str, float]:
    losses = {}

    model.eval()

    with torch.no_grad():
        for features, targets in tqdm(dataloader, desc=f"Validation (Epoch {epoch})", dynamic_ncols=True):
            features, targets = send_to_device(features, device), send_to_device(targets, device)

            predictions = model(features)

            batch_losses = criterion(predictions, targets)

            losses = {k: losses.get(k, 0) + v.item() for k, v in batch_losses.items()}

        losses = {k: v / len(dataloader) for k, v in losses.items()}

    return losses
        
def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: RecommenderCriterion,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    max_epochs: int,
    device: str = "cpu",
    output_dir: str = "weights/recommender",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    best_loss = float("inf")
    
    for epoch in range(max_epochs):
        train_one_epoch(model, optimizer, criterion, train_dataloader, epoch, device)

        # TODO: There is something weird going on with evaluation - GPU memory shoots up masssively and I intermittently get segfaults. 
        test_losses = evaluate(model, criterion, test_dataloader, epoch, device)

        wandb.log({"Validation": {"Loss": test_losses}}, step=wandb.run.step)

        torch.save(model.state_dict(), os.path.join(output_dir, "last.pt"))

        if (test_losses["overall"] < best_loss):
            best_loss = test_losses["overall"]
            torch.save(model.state_dict(), os.path.join(output_dir, "best.pt"))