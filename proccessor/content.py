from typing import Dict, Tuple

import torch
from torch import Tensor, optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from model.encoder import Encoder, MultiLayerPerceptron
from utils.loss import EncoderCriterion
from utils.metric import get_id_metrics, get_reid_metrics


def train_one_epoch(
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

    for i, (anchor, positive, negative) in tqdm(enumerate(dataloader), desc=f"Training (Epoch {epoch})", total=len(dataloader), dynamic_ncols=True):
        anchor_text, anchor_ids = anchor 
        positive_text, positive_ids = positive
        negative_text, negative_ids = negative 

        anchor_embeddings = encoder(anchor_text)
        positive_embeddings = encoder(positive_text)
        negative_embeddings = encoder(negative_text)

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

def get_item_embeddings(
        encoder: Encoder,
        descriptions_dataloader: DataLoader,
        epoch: int,
        device: str = "cpu",
) -> Tuple[Tensor, Tensor]:
    item_embeddings = []
    item_ids = []

    with torch.no_grad():
        for item_ids, descriptions in tqdm(descriptions_dataloader, desc=f"Description Embeddings (Epoch {epoch})", dynamic_ncols=True):
            batch_embeddings = encoder(descriptions)

            item_embeddings.append(batch_embeddings.cpu())
            item_ids.append(item_ids.cpu())

    item_embeddings = torch.cat(item_embeddings)
    item_ids = torch.cat(item_ids)

    return item_embeddings, item_ids

def evaluate(
    encoder: Encoder,
    classifier: MultiLayerPerceptron,
    criterion: EncoderCriterion,
    dataloader: DataLoader,
    items: Tuple[Tensor, Tensor],
    epoch: int,
    device: str = "cpu",
) -> Tuple[Dict[str, float], Dict[str, float]]:
    encoder.eval()

    losses = {}

    request_embeddings = []
    request_logits = []
    request_ids = []

    with torch.no_grad():
        for anchor, positive, negative in tqdm(dataloader, desc=f"Evaluation (Epoch {epoch})", dynamic_ncols=True):
            anchor_text, anchor_ids = anchor 
            positive_text, positive_ids = positive
            negative_text, negative_ids = negative 

            anchor_embeddings = encoder(anchor_text)
            positive_embeddings = encoder(positive_text)
            negative_embeddings = encoder(negative_text)

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

    reid_metrics = get_reid_metrics((request_embeddings, request_ids), items)
    id_metrics = get_id_metrics(request_logits, request_ids)

    return losses, {**id_metrics, **reid_metrics}

def train(
    encoder: Encoder,
    classifier: MultiLayerPerceptron,
    optimizer: optim.Optimizer,
    criterion: EncoderCriterion,
    train_dataloader: DataLoader,
    train_subset_dataloader: DataLoader,
    test_dataloader: DataLoader,
    descriptions_dataloader: DataLoader,
    max_epochs: int,
    accumulation_steps: int = 1,
    device: str = "cpu",
) -> None:

    for epoch in range(max_epochs): 
        train_one_epoch(encoder, classifier, optimizer, criterion, train_dataloader, epoch, accumulation_steps, device)

        items = get_item_embeddings(encoder, descriptions_dataloader, epoch, device)
        
        _, train_metrics = evaluate(encoder, classifier, criterion, train_subset_dataloader, items, epoch, device)

        test_losses, test_metrics = evaluate(encoder, classifier, criterion, test_dataloader, items, epoch, device)

        wandb.log({"Train": {"Metric": train_metrics}, "Validation": {"Loss": test_losses, "Metric": test_metrics}}, step=wandb.run.step)