import random

import optuna
import pandas as pd
import torch
import yaml
from torch import optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model.encoder import Encoder, build_classifier, build_expander
from model.recommender import DeepFM
from proccessor.collaborative import evaluate, train_one_epoch
from utils.criterion import CollaborativeCriterion
from utils.data import CollaborativeDataset, collaborative_collate_fn, train_test_split_ratings, train_test_split_requests
from utils.misc import set_random_seed

args = yaml.safe_load(open("configs/collaborative.yaml", "r"))

device = "cuda" if torch.cuda.is_available() else "cpu"

set_random_seed(args["random_seed"])

requests = pd.read_csv("data/ml-20m/requests.csv")
requests = (
    requests.groupby("item_id")
    .agg(
        {
            "item_title": "first",
            "request": list,
        }
    )
    .reset_index()
)
requests.set_index("item_id", inplace=True, drop=False)

train_requests, test_requests = train_test_split_requests(requests, train_size=0.8)

ratings = pd.read_hdf("data/ml-20m/processed_ratings.hdf")

train_ratings, test_ratings = train_test_split_ratings(ratings, train_size=0.8)

train_dataset = CollaborativeDataset(train_ratings, train_requests)
test_dataset = CollaborativeDataset(test_ratings, test_requests)
train_subset = Subset(train_dataset, random.sample(range(len(train_dataset)), k=len(test_dataset)))

train_dataloader = DataLoader(
    train_dataset, batch_size=args["batch_size"], shuffle=True, collate_fn=collaborative_collate_fn, num_workers=6, drop_last=True
)

test_dataloader = DataLoader(
    test_dataset, batch_size=args["batch_size"], collate_fn=collaborative_collate_fn, num_workers=6, drop_last=True
)

train_subset_dataloader = DataLoader(
    train_subset, batch_size=args["batch_size"], shuffle=False, collate_fn=collaborative_collate_fn, num_workers=6, drop_last=True
)


def objective(trial: optuna.Trial) -> float:
    # Dropout
    args["encoder"]["hidden_dropout_prob"] = trial.suggest_float("encoder.dropout", 0.0, 1.0)
    args["classifier"]["dropout"] = trial.suggest_float("classifier.dropout", 0.0, 1.0)
    args["expander"]["dropout"] = trial.suggest_float("expander.dropout", 0.0, 1.0)
    args["recommender"]["dropout"] = trial.suggest_float("recommender.dropout", 0.0, 1.0)

    # Weight Decay
    args["optimizer"]["encoder"]["weight_decay"] = trial.suggest_float("encoder.weight_decay", 0.0, 0.1)
    args["optimizer"]["classifier"]["weight_decay"] = trial.suggest_float("classifier.weight_decay", 0.0, 0.1)
    args["optimizer"]["expander"]["weight_decay"] = trial.suggest_float("expander.weight_decay", 0.0, 0.1)
    args["optimizer"]["recommender"]["weight_decay"] = trial.suggest_float("recommender.weight_decay", 0.0, 0.1)

    # Loss Weights
    args["criterion"]["loss_weights"]["mse"] = trial.suggest_float("weights.mse", 0.0, 10.0)
    args["criterion"]["loss_weights"]["id"] = trial.suggest_float("weights.id", 0.0, 10.0)
    args["criterion"]["loss_weights"]["triplet"] = trial.suggest_float("weights.triplet", 0.0, 10.0)
    args["criterion"]["loss_weights"]["variance"] = trial.suggest_float("weights.variance", 0.0, 1.0)
    args["criterion"]["loss_weights"]["invariance"] = trial.suggest_float("weights.invariance", 0.0, 1.0)
    args["criterion"]["loss_weights"]["covariance"] = trial.suggest_float("weights.covariance", 0.0, 1.0)

    # Misc
    args["criterion"]["triplet_margin"] = trial.suggest_float("triplet_margin", 0.0, 2.0)
    args["criterion"]["focal_gamma"] = trial.suggest_float("focal_gamma", 0.0, 2.0)
    args["train"]["max_epochs"] = trial.suggest_int("max_epochs", 1, 10)

    encoder = Encoder(**args["encoder"]).to(device)

    expander = build_expander(embed_dim=encoder.embed_dim, **args["expander"]).to(device)

    classifier = build_classifier(embed_dim=encoder.embed_dim, num_classes=requests["item_id"].nunique(), **args["classifier"]).to(device)

    recommender = DeepFM(num_items=ratings["item_id"].nunique(), **args["recommender"]).to(device)

    optimizer = optim.AdamW(
        [
            {"params": encoder.parameters(), **args["optimizer"]["encoder"]},
            {"params": expander.parameters(), **args["optimizer"]["expander"]},
            {"params": classifier.parameters(), **args["optimizer"]["classifier"]},
            {"params": recommender.parameters(), **args["optimizer"]["recommender"]},
        ]
    )

    criterion = CollaborativeCriterion(expander=expander, **args["criterion"])

    try:
        for epoch in tqdm(range(args["train"]["max_epochs"]), desc=f"Trial {trial.number}", unit="epochs", dynamic_ncols=True):
            train_one_epoch(encoder, classifier, recommender, optimizer, criterion, train_dataloader, epoch, device=device, verbose=False)

            test_losses, test_metrics = evaluate(
                encoder, classifier, recommender, criterion, test_dataloader, epoch, device=device, verbose=False
            )
    except ValueError:  # Model training was too unstable
        return 1.0, 0, 0, 0, 0

    return test_losses["mse"], test_metrics["reid_map"], test_metrics["rank-1"], test_metrics["rank-5"], test_metrics["rank-10"]


if __name__ == "__main__":
    study = optuna.create_study(directions=["minimize", "maximize", "maximize", "maximize", "maximize"])
    study.enqueue_trial(
        {
            "encoder.dropout": args["encoder"]["hidden_dropout_prob"],
            "classifier.dropout": args["classifier"]["dropout"],
            "expander.dropout": args["expander"]["dropout"],
            "recommender.dropout": args["recommender"]["dropout"],
            "encoder.weight_decay": args["optimizer"]["encoder"]["weight_decay"],
            "classifier.weight_decay": args["optimizer"]["classifier"]["weight_decay"],
            "expander.weight_decay": args["optimizer"]["expander"]["weight_decay"],
            "recommender.weight_decay": args["optimizer"]["recommender"]["weight_decay"],
            "weights.mse": args["criterion"]["loss_weights"]["mse"],
            "weights.id": args["criterion"]["loss_weights"]["id"],
            "weights.triplet": args["criterion"]["loss_weights"]["triplet"],
            "weights.variance": args["criterion"]["loss_weights"]["variance"],
            "weights.invariance": args["criterion"]["loss_weights"]["invariance"],
            "weights.covariance": args["criterion"]["loss_weights"]["covariance"],
            "triplet_margin": args["criterion"]["triplet_margin"],
            "focal_gamma": args["criterion"]["focal_gamma"],
            "max_epochs": args["train"]["max_epochs"],
        }
    )
    study.optimize(objective, n_trials=100, gc_after_trial=True)

    for trial in study.best_trials:
        print(
            f"\nTrial #{trial.number} MSE: {trial.values[0]} mAP: {trial.values[1]} Rank-1: {trial.values[2]} Rank-5: {trial.values[3]} Rank-10: {trial.values[4]}"
        )
        for key, value in trial.params.items():
            print(f"{key}: {value}")
