from typing import Dict, Tuple

import torch
from torch import optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from model.encoder import Encoder
from model.layers import MultiLayerPerceptron
from model.recommender import DeepFM
from utils.loss import JointCriterion
from utils.metric import get_id_metrics, get_reid_metrics


def train_one_epoch(
    encoder: Encoder,
    classifier: MultiLayerPerceptron,
    recommender: DeepFM,
    optimizer: optim.Optimizer,
    criterion: JointCriterion,
    dataloader: DataLoader,
    epoch: int,
    accumulation_steps: int = 1,
    device: str = "cpu"
) -> None:
    losses = {}

    encoder.train()

    for i, (rec_features, rec_targets, anchor, negative) in tqdm(enumerate(dataloader), desc=f"Training (Epoch {epoch})", total=len(dataloader)):
        rec_features, rec_targets = rec_features.to(device), rec_targets.to(device)

        rec_predictions = recommender(rec_features)

        anchor_requests, anchor_ids = anchor 
        negative_requests, negative_ids = negative 

        anchor_embeddings = encoder(anchor_requests)
        positive_embeddings = recommender.embedding.embedding.weight[recommender.embedding.offsets[1] + anchor_ids]
        negative_embeddings = encoder(negative_requests)

        anchor_logits = classifier(anchor_embeddings)
        positive_logits = classifier(positive_embeddings)
        negative_logits = classifier(negative_embeddings)

        batch_losses = criterion(rec_predictions, rec_targets, (anchor_embeddings, anchor_logits, anchor_ids), (positive_embeddings, positive_logits, anchor_ids), (negative_embeddings, negative_logits, negative_ids))

        wandb.log({"Train": {"Loss": batch_losses}}, step=wandb.run.step + len(anchor_embeddings))

        losses = {k: losses.get(k, 0) + v.item() for k, v in batch_losses.items()}
        
        loss = batch_losses["overall"]

        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

def evaluate(
    encoder: Encoder,
    classifier: MultiLayerPerceptron,
    recommender: DeepFM,
    criterion: JointCriterion,
    dataloader: DataLoader,
    epoch: int,
    device: str = "cpu",
) -> Tuple[Dict[str, float], Dict[str, float]]:
    encoder.eval()

    losses = {}

    request_embeddings = []
    request_logits = []
    request_ids = []

    with torch.no_grad():
        for rec_features, rec_targets, anchor, negative in tqdm(dataloader, desc=f"Validation (Epoch {epoch})"):
            rec_features, rec_targets = rec_features.to(device), rec_targets.to(device)

            rec_predictions = recommender(rec_features)

            anchor_requests, anchor_ids = anchor 
            negative_requests, negative_ids = negative 

            anchor_embeddings = encoder(anchor_requests)
            positive_embeddings = recommender.embedding.embedding.weight[recommender.embedding.offsets[1] + anchor_ids]
            negative_embeddings = encoder(negative_requests)

            anchor_logits = classifier(anchor_embeddings)
            positive_logits = classifier(positive_embeddings)
            negative_logits = classifier(negative_embeddings)

            batch_losses = criterion(rec_predictions, rec_targets, (anchor_embeddings, anchor_logits, anchor_ids), (positive_embeddings, positive_logits, anchor_ids), (negative_embeddings, negative_logits, negative_ids))

            losses = {k: losses.get(k, 0) + v.item() for k, v in batch_losses.items()}

            request_embeddings.append(anchor_embeddings.cpu())
            request_logits.append(anchor_logits.cpu())
            request_ids.append(anchor_ids.cpu())
            
    losses = {k: v / len(dataloader) for k, v in losses.items()}
    
    request_embeddings = torch.cat(request_embeddings)
    request_logits = torch.cat(request_logits)
    request_ids = torch.cat(request_ids)

    item_embeddings = recommender.embedding.embedding.weight[recommender.embedding.offsets[1]:].cpu()
    item_ids = torch.arange(len(item_embeddings))

    reid_metrics = get_reid_metrics((request_embeddings, request_ids), (item_embeddings, item_ids))
    id_metrics = get_id_metrics(request_logits, request_ids)

    return losses, {**id_metrics, **reid_metrics}


def train(
    encoder: Encoder,
    classifier: MultiLayerPerceptron,
    recommender: DeepFM,
    optimizer: optim.Optimizer,
    criterion: JointCriterion,
    train_dataloader: DataLoader,
    train_subset_dataloader: DataLoader,
    test_dataloader: DataLoader,
    max_epochs: int,
    accumulation_steps: int = 1,
    device: str = "cpu",
) -> None:
    for epoch in range(max_epochs):
        train_one_epoch(encoder, classifier, recommender, optimizer, criterion, train_dataloader, epoch, accumulation_steps, device)

        _, train_metrics = evaluate(encoder, classifier, recommender, criterion, train_subset_dataloader, epoch, device)

        test_losses, test_metrics = evaluate(encoder, classifier, recommender, criterion, test_dataloader, epoch, device)

        wandb.log({"Train": {"Metric": train_metrics}, "Validation": {"Loss": test_losses, "Metric": test_metrics}}, step=wandb.run.step)