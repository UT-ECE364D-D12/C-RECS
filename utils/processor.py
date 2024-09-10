import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from model.encoder import Encoder
from model.recommender import DeepFM
from utils.loss import Criterion, EncoderRecommenderCriterion, RecommenderCriterion


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
    encoder: Encoder,
    recommender: DeepFM,
    optimizer: optim.Optimizer,
    criterion: EncoderRecommenderCriterion,
    dataloader: DataLoader,
    epoch: int,
    device: str = "cpu"
) -> None:
    criterion.reset_metrics()
    losses = {}

    encoder.train()

    for rec_features, rec_targets, positive, negative in tqdm(dataloader, desc=f"Training (Epoch {epoch})"):
        optimizer.zero_grad()
        
        rec_features, rec_targets = rec_features.to(device), rec_targets.to(device)

        rec_predictions = recommender(rec_features)

        positive_requests, positive_ids = positive 
        negative_requests, negative_ids = negative 

        positive_embeddings = encoder(positive_requests)
        negative_embeddings = encoder(negative_requests)

        anchor_embeddings, anchor_ids = recommender.embedding.embedding.weight[recommender.embedding.offsets[1] + positive_ids], positive_ids

        batch_losses = criterion(rec_predictions, rec_targets, (anchor_embeddings, anchor_ids), (positive_embeddings, positive_ids), (negative_embeddings, negative_ids))

        wandb.log({"Train": {"Loss": batch_losses}}, step=wandb.run.step + len(anchor_embeddings))

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
    encoder: Encoder,
    recommender: DeepFM,
    criterion: EncoderRecommenderCriterion,
    dataloader: DataLoader,
    epoch: int,
    device: str = "cpu",
) -> None:
    criterion.reset_metrics()
    losses = {}

    encoder.eval()

    with torch.no_grad():
        for rec_features, rec_targets, positive, negative in tqdm(dataloader, desc=f"Validation (Epoch {epoch})"):
            
            rec_features, rec_targets = rec_features.to(device), rec_targets.to(device)

            rec_predictions = recommender(rec_features)

            positive_requests, positive_ids = positive 
            negative_requests, negative_ids = negative 

            positive_embeddings = encoder(positive_requests)
            negative_embeddings = encoder(negative_requests)

            anchor_embeddings, anchor_ids = recommender.embedding.embedding.weight[recommender.embedding.offsets[1] + positive_ids], positive_ids

            batch_losses = criterion(rec_predictions, rec_targets, (anchor_embeddings, anchor_ids), (positive_embeddings, positive_ids), (negative_embeddings, negative_ids))

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
    encoder: Encoder,
    recommender: DeepFM,
    optimizer: optim.Optimizer,
    criterion: EncoderRecommenderCriterion,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    max_epochs: int,
    device: str = "cpu",
) -> None:
    for epoch in range(max_epochs):
        train_encoder_one_epoch(encoder, recommender, optimizer, criterion, train_dataloader, epoch, device)
        
        evaluate_encoder_one_epoch(encoder, recommender, criterion, test_dataloader, epoch, device)