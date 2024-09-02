from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from utils.loss import Criterion


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

def evaluate_one_epoch(
    model: nn.Module,
    criterion: Criterion,
    dataloader: DataLoader,
    epoch: int,
    device: str = "cpu",
) -> None:
    criterion.reset_metrics()
    losses = {}

    for features, targets in tqdm(dataloader, desc=f"Validation (Epoch {epoch})"):
        features, targets = features.to(device), targets.to(device)

        predictions = model(features)

        batch_losses = criterion(predictions, targets)

        losses = {k: losses.get(k, 0) + v.item() for k, v in batch_losses.items()}

    wandb.log({"Validation": {"Loss": losses}}, step=wandb.run.step)
        
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