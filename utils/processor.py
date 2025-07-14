import os
from typing import Dict, List, Tuple, Union

import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from criterion.base_criterion import Criterion
from metrics.content_evaluator import Evaluator
from model.crecs import CRECS
from model.recommender import DeepFM
from utils.logging import get_model_statistics
from utils.lr import CosineAnnealingWarmRestarts
from utils.misc import send_to_device

Model = Union[CRECS, DeepFM]
EvaluationJob = Tuple[Model, Evaluator, DataLoader]


def train(
    model: Model,
    optimizer: Optimizer,
    scheduler: CosineAnnealingWarmRestarts,
    criterion: Criterion,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    eval_jobs: List[EvaluationJob],
    num_epochs: int,
    output_dir: str = "weights/collaborative",
    max_grad_norm: float = None,
    **kwargs: Dict[str, any],
) -> None:
    """
    Train & evaluate a model.

    Args:
        model: The model to train.
        optimizer: The optimizer.
        criterion: The loss function.
        train_dataloader: The training data.
        val_dataloader: The validation data, for loss evaluation.
        eval_jobs: List of evaluation jobs.
        num_epochs: The number of epochs to train for.
        max_grad_norm: Clip value for the gradient norm.
        output_dir: Directory to save the model weights.
        kwargs: Additional training arguments:
            device: The device to use for training & evaluation
            verbose: Whether to log the training progress.
    """

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    best_loss = float("inf")

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, scheduler, criterion, train_dataloader, epoch, max_grad_norm, **kwargs)

        val_losses = evaluate_losses(model, criterion, val_dataloader, **kwargs)

        # Log the validation losses
        wandb.log({"Validation": {"Loss": val_losses}}, step=wandb.run.step)

        for eval_model, eval_calculator, eval_dataloader in eval_jobs:
            metrics = evaluate_metrics(eval_model, eval_calculator, eval_dataloader, **kwargs)

            # Log the evaluation metrics
            wandb.log({"Validation": {"Metrics": metrics}}, step=wandb.run.step)

        # Update the latest and best model weights
        torch.save(model.state_dict(), os.path.join(output_dir, "last.pt"))

        # TODO: Save based on metrics - maybe have a parameter to choose metric & fall back to loss
        if val_losses["overall"] < best_loss:
            best_loss = val_losses["overall"]

            torch.save(model.state_dict(), os.path.join(output_dir, "best.pt"))


def train_one_epoch(
    model: Model,
    optimizer: Optimizer,
    scheduler: CosineAnnealingWarmRestarts,
    criterion: Criterion,
    dataloader: DataLoader,
    epoch: int,
    max_grad_norm: float = None,
    device: str = "cpu",
    verbose: bool = True,
) -> None:
    """
    Train the model for one epoch.

    Args:
        model: The model to train.
        optimizer: The optimizer.
        criterion: The loss function.
        dataloader: The training data.
        epoch: The current epoch.
        max_grad_norm: Clip value for the gradient norm.
        device: The device to use.
        verbose: Whether to log the training progress.
    """

    losses = {}

    model.train()

    for batch in tqdm(dataloader, desc=f"Training (Epoch {epoch})", dynamic_ncols=True, disable=not verbose):
        optimizer.zero_grad()

        # Send the batch to the training device
        features, targets = send_to_device(batch, device)

        # Forward pass
        predictions = model(features)

        # Compute losses
        batch_losses = criterion(predictions, targets)

        losses = {k: losses.get(k, 0) + v.item() for k, v in batch_losses.items()}

        # Backward pass
        loss = batch_losses["overall"]

        loss.backward()

        if max_grad_norm:
            clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()

        # Obtain the gradient & parameter statistics
        modules = ["encoder", "recommender", "classifier"]
        model_statistics = {module: get_model_statistics(model.__getattr__(module)) for module in modules if hasattr(model, module)}

        # Update the learning rates
        learning_rates = scheduler.step()

        # Log the training progress
        if verbose:
            wandb.log({"Train": {"Loss": batch_losses}, **learning_rates, **model_statistics}, step=wandb.run.step + len(targets))


@torch.no_grad()
def evaluate_losses(
    model: Model,
    criterion: Criterion,
    dataloader: DataLoader,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Calculate the average losses of the model on the given data.

    Args:
        model: The model to evaluate.
        criterion: The loss function.
        dataloader: The evaluation data.
        epoch: The current epoch.
        device: The device to use.
        verbose: Whether to log progress.

    Returns:
        losses: The losses of the model.
    """

    model.eval()

    losses = {}

    for batch in tqdm(dataloader, desc=f"Evaluating Losses", dynamic_ncols=True, disable=not verbose):

        # Send the batch to the training device
        features, targets = send_to_device(batch, device)

        # Forward pass
        predictions = model(features)

        # Compute losses
        batch_losses = criterion(predictions, targets)

        # Update the running losses
        losses = {k: losses.get(k, 0) + v.item() for k, v in batch_losses.items()}

    # Normalize losses
    losses = {k: v / len(dataloader) for k, v in losses.items()}

    return losses


@torch.no_grad()
def evaluate_metrics(
    model: Model,
    evaluator: Evaluator,
    dataloader: DataLoader,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate the model on the given data.

    Args:
        model: The model to evaluate.
        evaluator: The metric calculator.
        dataloader: The evaluation data.
        device: The device to use.
        verbose: Whether to log progress.

    Returns:
        metrics: The evaluation metrics.
    """

    model.eval()
    evaluator.reset()

    for batch in tqdm(dataloader, desc=f"Evaluating Metrics", dynamic_ncols=True, disable=not verbose):
        # Send the batch to the training device
        features, targets = send_to_device(batch, device)

        # Forward pass
        predictions = model.predict(features)

        evaluator.update(predictions, targets)

    metrics = evaluator.calculate()

    return metrics
