import os
from typing import Dict

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from model.recommender import DeepFM
from utils.criterion import RecommenderCriterion
from utils.misc import send_to_device


def train_one_epoch(
    model: DeepFM,
    optimizer: optim.Optimizer,
    criterion: RecommenderCriterion,
    dataloader: DataLoader,
    epoch: int,
    device: str = "cpu",
) -> None:
    """
    Train a recommender model for one epoch.

    Args:
        model (DeepFM): The model to train.
        optimizer (optim.Optimizer): The optimizer.
        criterion (RecommenderCriterion): The loss function.
        dataloader (DataLoader): The training data.
        epoch (int): The current epoch.
        device (str, optional): The device to use.
    """

    losses = {}

    model.train()

    for features, targets in tqdm(dataloader, desc=f"Training (Epoch {epoch})", dynamic_ncols=True):
        torch.cuda.empty_cache()

        optimizer.zero_grad()

        features, targets = send_to_device(features, device), send_to_device(targets, device)

        predictions = model(features)

        batch_losses = criterion(predictions, targets)

        wandb.log({"Train": {"Loss": batch_losses}}, step=wandb.run.step + len(predictions))

        losses = {k: losses.get(k, 0) + v.item() for k, v in batch_losses.items()}

        loss = batch_losses["overall"]

        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: RecommenderCriterion,
    dataloader: DataLoader,
    epoch: int,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate a recommender model.

    Args:
        model (nn.Module): The model to evaluate.
        criterion (RecommenderCriterion): The loss function.
        dataloader (DataLoader): The data to evaluate.
        epoch (int): The current epoch.
        device (str, optional): The device to use.

    Returns:
        losses (Dict[str, float]): The average losses.
        metrics (Dict[str, float]): The average metrics.
    """

    losses = {}

    model.eval()

    for features, targets in tqdm(dataloader, desc=f"Validation (Epoch {epoch})", dynamic_ncols=True):
        torch.cuda.empty_cache()

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
    val_dataloder: DataLoader,
    max_epochs: int,
    device: str = "cpu",
    output_dir: str = "weights/recommender",
) -> None:
    """
    Train a recommender model.

    Args:
        model (nn.Module): The model to train.
        optimizer (optim.Optimizer): The optimizer.
        criterion (RecommenderCriterion): The loss function.
        train_dataloader (DataLoader): The training data.
        val_dataloader (DataLoader): The validation data.
        max_epochs (int): The number of epochs to train for.
        device (str, optional): The device to use.
        output_dir (str, optional): The directory to save the weights.
    """

    os.makedirs(output_dir, exist_ok=True)

    best_loss = float("inf")

    for epoch in range(max_epochs):
        train_one_epoch(model, optimizer, criterion, train_dataloader, epoch, device)

        # TODO: There is something weird going on with evaluation - GPU memory shoots up masssively and I intermittently get segfaults.
        val_losses = evaluate(model, criterion, val_dataloder, epoch, device)

        # Log the validation losses
        wandb.log({"Validation": {"Loss": val_losses}}, step=wandb.run.step)

        # Save the latest and best model weights
        torch.save(model.state_dict(), os.path.join(output_dir, "last.pt"))

        if val_losses["overall"] < best_loss:
            best_loss = val_losses["overall"]
            torch.save(model.state_dict(), os.path.join(output_dir, "best.pt"))
