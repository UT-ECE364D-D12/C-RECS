import os
from typing import Dict, Tuple

import torch
from torch import optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from model.crecs import CRECS
from utils.criterion import CollaborativeCriterion
from utils.logging import get_model_statistics
from utils.metric import get_id_metrics, get_reid_metrics
from utils.misc import send_to_device


def train_one_epoch(
    model: CRECS,
    optimizer: optim.Optimizer,
    criterion: CollaborativeCriterion,
    dataloader: DataLoader,
    epoch: int,
    max_grad_norm: float = None,
    grad_accumulation_steps: int = 1,
    device: str = "cpu",
    verbose: bool = True,
) -> None:
    """
    Train CRECS using collaborative filtering.

    Args:
        model (CRECS): The CRECS model to train.
        optimizer (optim.Optimizer): The optimizer.
        criterion (CollaborativeCriterion): The loss function.
        dataloader (DataLoader): The training data.
        epoch (int): The current epoch.
        max_grad_norm (float, optional): Clip value for the gradient norm.
        grad_accumulation_steps (int, optional): Number of steps to accumulate gradients.
        device (str, optional): The device to use.
        verbose (bool, optional): Whether to log the training progress.
    """

    losses = {}

    model.train()

    # Track epoch progress in the terminal
    if verbose:
        data = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Training (Epoch {epoch})", dynamic_ncols=True)
    else:
        data = enumerate(dataloader)

    for i, (rec_features, rec_targets, anchors, negative_ids) in data:
        # Unpack the batch & send it to the training device
        anchor_requests, anchor_ids = anchors

        rec_features, rec_targets = send_to_device(rec_features, device), send_to_device(rec_targets, device)
        anchor_ids, negative_ids = send_to_device(anchor_ids, device), send_to_device(negative_ids, device)

        # Forward pass
        rec_predictions, anchor, positive, negative = model(rec_features, anchor_requests, anchor_ids, negative_ids)

        # Compute losses
        batch_losses = criterion(rec_predictions, rec_targets, anchor, positive, negative)

        losses = {k: losses.get(k, 0) + v.item() for k, v in batch_losses.items()}

        # Log the gradient/parameter statistics and losses
        if verbose:
            modules = ["Encoder", "Recommender", "Classifier"]
            model_statistics = {module: get_model_statistics(model.__getattr__(module.lower())) for module in modules}

            wandb.log({"Train": {"Loss": batch_losses}, **model_statistics}, step=wandb.run.step + len(anchor_requests))
        
        # Backward pass
        loss = batch_losses["overall"]

        loss.backward()

        if (i + 1) % grad_accumulation_steps == 0:
            if max_grad_norm: clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

@torch.no_grad()
def evaluate(
    model: CRECS,
    criterion: CollaborativeCriterion,
    dataloader: DataLoader,
    epoch: int,
    device: str = "cpu",
    verbose: bool = True
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Evaluate CRECS using collaborative filtering.

    Args:
        model (CRECS): The CRECS model to evaluate.
        criterion (CollaborativeCriterion): The loss function.
        dataloader (DataLoader): The data to evaluate.
        epoch (int): The current epoch.
        device (str, optional): The device to use.
        verbose (bool, optional): Whether to log the evaluation progress.

    Returns:
        losses (Dict[str, float]): The losses of the model.
        metrics (Dict[str, float]): The metrics of the model.
    """

    model.eval()

    losses = {}

    predictions = []
    target_ids = []

    metrics = {}
    num_samples = 0

    gallery_embeddings = model.recommender.embedding.item_embedding.weight
    gallery_ids = torch.arange(len(gallery_embeddings)).to(device)

    data = tqdm(dataloader, desc=f"Validation (Epoch {epoch})", dynamic_ncols=True) if verbose else dataloader
    for rec_features, rec_targets, anchors, negative_ids in data:
        # Unpack the batch & send it to the evaluation device
        anchor_requests, anchor_ids = anchors

        rec_features, rec_targets = send_to_device(rec_features, device), send_to_device(rec_targets, device)
        anchor_ids, negative_ids = send_to_device(anchor_ids, device), send_to_device(negative_ids, device)

        # Forward pass
        rec_predictions, anchor, positive, negative = model(rec_features, anchor_requests, anchor_ids, negative_ids)

        # Compute losses
        batch_losses = criterion(rec_predictions, rec_targets, anchor, positive, negative)

        losses = {k: losses.get(k, 0) + v.item() for k, v in batch_losses.items()}

        # Compute reid metrics
        anchor_embeddings, anchor_logits, anchor_ids = anchor

        reid_metrics = get_reid_metrics((anchor_embeddings, anchor_ids), (gallery_embeddings, gallery_ids), device=device)

        num_samples += (batch_size := len(anchor_ids))
        metrics = {k: metrics.get(k, 0) + v * batch_size for k, v in reid_metrics.items()}

        # Save id predictions and targets for later evaluation
        batch_predictions = torch.stack(anchor_logits.softmax(dim=-1).max(dim=-1), dim=-1)

        predictions.append(batch_predictions.cpu())
        target_ids.append(anchor_ids.cpu())

    # Normalize losses and reid metrics
    losses = {k: v / len(dataloader) for k, v in losses.items()}

    metrics = {k: v / num_samples for k, v in metrics.items()}

    # Calculate id metrics
    predictions = torch.cat(predictions)
    target_ids = torch.cat(target_ids)

    metrics = {**metrics, **get_id_metrics(predictions, target_ids)}

    return losses, metrics


def train(
    model: CRECS,
    optimizer: optim.Optimizer,
    criterion: CollaborativeCriterion,
    train_dataloader: DataLoader,
    train_subset_dataloader: DataLoader,
    test_dataloader: DataLoader,
    max_epochs: int,
    max_grad_norm: float = 10.0,
    grad_accumulation_steps: int = 1,
    output_dir: str = "weights/collaborative",
    **kwargs,
) -> None:
    """
    Train CRECS using collaborative filtering.

    Args:
        model (CRECS): The CRECS model to train.
        optimizer (optim.Optimizer): The optimizer.
        criterion (CollaborativeCriterion): Loss function.
        train_dataloader (DataLoader): Data to train on.
        train_subset_dataloader (DataLoader): Subset of the training data to evaluate on.
        test_dataloader (DataLoader): Held-out data used for validation.
        max_epochs (int): The number of epochs to train for.
        max_grad_norm (float, optional): Clip value for the gradient norm.
        grad_accumulation_steps (int, optional): Number of steps to accumulate gradients.
        output_dir (str, optional): Directory to save the model weights.
        **kwargs: Additional training arguments.
    """
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    best_metric = float("-inf")

    for epoch in range(max_epochs):
        train_one_epoch(model, optimizer, criterion, train_dataloader, epoch, max_grad_norm, grad_accumulation_steps, **kwargs)

        _, train_metrics = evaluate(model, criterion, train_subset_dataloader, epoch, **kwargs)

        test_losses, test_metrics = evaluate(model, criterion, test_dataloader, epoch, **kwargs)

        # Log the training and validation metrics
        wandb.log({"Train": {"Metric": train_metrics}, "Validation": {"Loss": test_losses, "Metric": test_metrics}}, step=wandb.run.step)

        # Update the latest and best model weights
        torch.save(model.state_dict(), os.path.join(output_dir, "last.pt"))

        if test_metrics["reid_map"] > best_metric:
            best_metric = test_metrics["reid_map"]
            torch.save(model.state_dict(), os.path.join(output_dir, "best.pt"))
