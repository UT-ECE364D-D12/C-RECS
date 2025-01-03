import math
import os
import random
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics.functional import pairwise_cosine_similarity


# Modified from Pytorch to handle per-parameter group learning rates
class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: List[float] = [0], last_epoch: int = -1, verbose="deprecated"): 
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """Retrieve the learning rate of each parameter group."""
        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        elif self._step_count == 1 and self.last_epoch > 0:
            return [self.eta_min + (base_lr - self.eta_min[i]) * (1 + math.cos((self.last_epoch) * math.pi / self.T_max)) / 2 for i, (base_lr, group) in enumerate(zip(self.base_lrs, self.optimizer.param_groups))]
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group["lr"] + (base_lr - self.eta_min[i]) * (1 - math.cos(math.pi / self.T_max)) / 2 for i, (base_lr, group) in enumerate(zip(self.base_lrs, self.optimizer.param_groups))]
        
        return [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) * (group["lr"] - self.eta_min[i]) + self.eta_min[i] for i, group in enumerate(self.optimizer.param_groups)]

    def _get_closed_form_lr(self):
        return [self.eta_min[i] + (base_lr - self.eta_min[i]) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2 for i, base_lr in enumerate(self.base_lrs)]

def send_to_device(object: Union[Tensor, Dict, List, Tuple], device: str = "cpu") -> Union[Tensor, Dict, List, Tuple]:
    if isinstance(object, Tensor):
        return object.to(device)
    elif isinstance(object, dict):
        return {k: send_to_device(v, device) for k, v in object.items()}
    elif isinstance(object, list) or isinstance(object, tuple):
        return [send_to_device(element, device) for element in object]
    else:
        return object

def cosine_distance(x: Tensor, y: Tensor) -> Tensor:
    """
    Returns the cosine distance between two tensors.
    """
    return 1 - torch.cosine_similarity(x, y)

def pairwise_cosine_distance(x: Tensor, y: Tensor = None) -> Tensor:
    """
    Returns the pairwise cosine distance between two tensors.
    """
    return 1 - pairwise_cosine_similarity(x, y, zero_diagonal=False)

def set_random_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    
    os.environ["PYTHONHASHSEED"] = str(seed) 
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def suppress_warnings() -> None:
    """
    Suppress warnings.
    """
    import logging

    logging.getLogger("bitsandbytes").setLevel(logging.CRITICAL)
    logging.getLogger("transformers").setLevel(logging.CRITICAL)

    import warnings

    warnings.filterwarnings("ignore")

def get_model_statistics(model: nn.Module, norm_type: int = 2) -> dict:
    """
    Returns the parameter and gradient statistics of a model.
    """

    num_params = 0
    num_active_params = 0
    parameter_abs = 0.0
    parameter_norm = 0.0
    gradient_abs = 0.0
    gradient_norm = 0.0

    for param in model.parameters():
        # Parameter stats
        parameter_abs += param.detach().data.abs().sum().item()
        parameter_norm += param.detach().data.norm(norm_type).item() ** norm_type

        num_params += param.numel()

        # Gradient stats
        if param.grad is not None and param.requires_grad:
            gradient_abs += param.grad.detach().data.abs().sum().item()
            gradient_norm += param.grad.detach().data.norm(norm_type).item() ** norm_type

            num_active_params += param.numel()

    return {
        "Parameter": {
            "Abs": parameter_abs / num_params,
            "Norm": parameter_norm ** (1.0 / norm_type),
        },
        "Gradient": {
            "Abs": gradient_abs / num_active_params,
            "Norm": gradient_norm ** (1.0 / norm_type),
        }
    }