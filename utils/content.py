from typing import Dict, Tuple

import torch
from torch import optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from model.encoder import Encoder, MultiLayerPerceptron
from utils.loss import EncoderCriterion
from utils.metric import get_id_metrics, get_reid_metrics


def train_content_one_epoch(
    encoder: Encoder,
    classifier: MultiLayerPerceptron,
    optimizer: optim.Optimizer,
    criterion: EncoderCriterion,
    dataloader: DataLoader,
    epoch: int,
    accumulation_steps: int = 1,
    device: str = "cpu"
) -> None:
    losses = {}

    encoder.train()

    for i, (anchor, positive, negative) in tqdm(enumerate(dataloader), desc=f"Training (Epoch {epoch})", total=len(dataloader)):
        anchor_requests, anchor_ids = anchor 
        positive_descriptions, positive_ids = positive
        negative_requests, negative_ids = negative 

        anchor_embeddings = encoder(anchor_requests)
        positive_embeddings = encoder(positive_descriptions)
        negative_embeddings = encoder(negative_requests)

        anchor_logits = classifier(anchor_embeddings)
        positive_logits = classifier(positive_embeddings)
        negative_logits = classifier(negative_embeddings)

        batch_losses = criterion((anchor_embeddings, anchor_logits, anchor_ids), (positive_embeddings, positive_logits, positive_ids), (negative_embeddings, negative_logits, negative_ids))

        wandb.log({"Train": {"Loss": batch_losses}}, step=wandb.run.step + len(anchor_embeddings))

        losses = {k: losses.get(k, 0) + v.item() for k, v in batch_losses.items()}
        
        loss = batch_losses["overall"]

        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()


def evaluate_content_one_epoch(
    encoder: Encoder,
    classifier: MultiLayerPerceptron,
    criterion: EncoderCriterion,
    dataloader: DataLoader,
    descriptions_dataloader: DataLoader,
    epoch: int,
    device: str = "cpu",
) -> Tuple[Dict[str, float], Dict[str, float]]:
    losses = {}

    encoder.eval()

    request_embeddings = []
    request_logits = []
    request_ids = []

    with torch.no_grad():
        for anchor, positive, negative in tqdm(dataloader, desc=f"Validation (Epoch {epoch})"):
            anchor_requests, anchor_ids = anchor 
            positive_descriptions, positive_ids = positive
            negative_requests, negative_ids = negative 

            anchor_embeddings = encoder(anchor_requests)
            positive_embeddings = encoder(positive_descriptions)
            negative_embeddings = encoder(negative_requests)

            anchor_logits = classifier(anchor_embeddings)
            positive_logits = classifier(positive_embeddings)
            negative_logits = classifier(negative_embeddings)

            batch_losses = criterion((anchor_embeddings, anchor_logits, anchor_ids), (positive_embeddings, positive_logits, positive_ids), (negative_embeddings, negative_logits, negative_ids))

            losses = {k: losses.get(k, 0) + v.item() for k, v in batch_losses.items()}

            request_embeddings.append(anchor_embeddings.cpu())
            request_logits.append(anchor_logits.cpu())
            request_ids.append(anchor_ids.cpu())

    losses = {k: v / len(dataloader) for k, v in losses.items()}

    request_embeddings = torch.cat(request_embeddings)
    request_logits = torch.cat(request_logits)
    request_ids = torch.cat(request_ids)

    description_embeddings = []
    description_item_ids = []

    with torch.no_grad():
        for movie_ids, descriptions in tqdm(descriptions_dataloader):
            batch_embeddings = encoder(descriptions)

            description_embeddings.append(batch_embeddings.cpu())
            description_item_ids.append(movie_ids.cpu())

    description_embeddings = torch.cat(description_embeddings)
    description_item_ids = torch.cat(description_item_ids)

    reid_metrics = get_reid_metrics((request_embeddings, request_ids), (description_embeddings, description_item_ids))
    id_metrics = get_id_metrics(request_logits, request_ids)

    return losses, {**id_metrics, **reid_metrics}

def train_content(
    encoder: Encoder,
    classifier: MultiLayerPerceptron,
    optimizer: optim.Optimizer,
    criterion: EncoderCriterion,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    descriptions_dataloader: DataLoader,
    max_epochs: int,
    accumulation_steps: int = 1,
    device: str = "cpu",
) -> None:

    for epoch in range(max_epochs): 
        train_content_one_epoch(encoder, classifier, optimizer, criterion, train_dataloader, epoch, accumulation_steps, device)
        
        _, train_metrics = evaluate_content_one_epoch(encoder, classifier, criterion, train_dataloader, descriptions_dataloader, epoch, device)

        test_losses, test_metrics = evaluate_content_one_epoch(encoder, classifier, criterion, test_dataloader, descriptions_dataloader, epoch, device)

        wandb.log({"Train": {"Metric": train_metrics}, "Validation": {"Loss": test_losses, "Metric": test_metrics}}, step=wandb.run.step)