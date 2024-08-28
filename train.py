import wandb
from torch import nn, optim
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    dataloader: DataLoader,
    epoch: int,
    device: str = "cpu",
) -> None:
    pass

def evaluate_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    epoch: int,
    device: str = "cpu",
) -> None:
    pass

def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    max_epochs: int,
    device: str = "cpu",
) -> None:
    for epoch in range(max_epochs):
        train_one_epoch(model, optimizer, criterion, train_dataloader, epoch, device)
        
        evaluate_one_epoch(model, criterion, val_dataloader, epoch, device)