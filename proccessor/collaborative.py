from typing import Dict, Tuple

import torch
from torch import optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from model.crecs import CRECS
from model.layers import MultiLayerPerceptron
from utils.loss import CollaborativeCriterion
from utils.metric import get_id_metrics, get_reid_metrics
from utils.misc import send_to_device


def train_one_epoch(
    model: CRECS,
    optimizer: optim.Optimizer,
    criterion: CollaborativeCriterion,
    dataloader: DataLoader,
    epoch: int,
    accumulation_steps: int = 1,
    device: str = "cpu",
    verbose: bool = True,
) -> None:
    losses = {}

    model.train()

    data = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Training (Epoch {epoch})") if verbose else enumerate(dataloader)
    for i, (rec_features, rec_targets, anchors, negative_ids) in data:
        anchor_requests, anchor_ids = anchors 

        rec_features, rec_targets = send_to_device(rec_features, device), send_to_device(rec_targets, device)
        anchor_ids, negative_ids = send_to_device(anchor_ids, device), send_to_device(negative_ids, device)

        rec_predictions, anchor, positive, negative = model(rec_features, anchor_requests, anchor_ids, negative_ids)

        batch_losses = criterion(rec_predictions, rec_targets, anchor, positive, negative)

        if verbose:
            wandb.log({"Train": {"Loss": batch_losses}}, step=wandb.run.step + len(anchor_requests))

        losses = {k: losses.get(k, 0) + v.item() for k, v in batch_losses.items()}
        
        loss = batch_losses["overall"]

        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

def evaluate(
    model: CRECS,
    criterion: CollaborativeCriterion,
    dataloader: DataLoader,
    epoch: int,
    device: str = "cpu",
    verbose: bool = True
) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.eval()

    losses = {}

    request_embeddings = []
    request_logits = []
    request_ids = []

    data = tqdm(dataloader, desc=f"Validation (Epoch {epoch})") if verbose else dataloader
    with torch.no_grad():
        for rec_features, rec_targets, anchors, negative_ids in data:
            anchor_requests, anchor_ids = anchors 

            rec_features, rec_targets = send_to_device(rec_features, device), send_to_device(rec_targets, device)
            anchor_ids, negative_ids = send_to_device(anchor_ids, device), send_to_device(negative_ids, device)

            rec_predictions, anchor, positive, negative = model(rec_features, anchor_requests, anchor_ids, negative_ids)

            batch_losses = criterion(rec_predictions, rec_targets, anchor, positive, negative)

            losses = {k: losses.get(k, 0) + v.item() for k, v in batch_losses.items()}

            anchor_request_embeddings, anchor_logits, anchor_ids = anchor

            request_embeddings.append(anchor_request_embeddings.cpu())
            request_logits.append(anchor_logits.cpu())
            request_ids.append(anchor_ids.cpu())
            
    losses = {k: v / len(dataloader) for k, v in losses.items()}
    
    request_embeddings = torch.cat(request_embeddings)
    request_logits = torch.cat(request_logits)
    request_ids = torch.cat(request_ids)

    item_embeddings = model.recommender.embedding.item_embedding.weight.cpu()
    item_ids = torch.arange(len(item_embeddings))

    reid_metrics = get_reid_metrics((request_embeddings, request_ids), (item_embeddings, item_ids))
    id_metrics = get_id_metrics(request_logits, request_ids)

    return losses, {**id_metrics, **reid_metrics}


def train(
    model: CRECS,
    optimizer: optim.Optimizer,
    criterion: CollaborativeCriterion,
    train_dataloader: DataLoader,
    train_subset_dataloader: DataLoader,
    test_dataloader: DataLoader,
    max_epochs: int,
    accumulation_steps: int = 1,
    **kwargs
) -> None:
    for epoch in range(max_epochs):
        train_one_epoch(model, optimizer, criterion, train_dataloader, epoch, accumulation_steps, **kwargs)

        _, train_metrics = evaluate(model, criterion, train_subset_dataloader, epoch, **kwargs)

        test_losses, test_metrics = evaluate(model, criterion, test_dataloader, epoch, **kwargs)

        wandb.log({"Train": {"Metric": train_metrics}, "Validation": {"Loss": test_losses, "Metric": test_metrics}}, step=wandb.run.step)