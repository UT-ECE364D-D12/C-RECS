import torchvision

torchvision.disable_beta_transforms_warning()

import logging

logging.getLogger("bitsandbytes").setLevel(logging.CRITICAL)

import math
import os
import random
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics.functional import pairwise_cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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


def build_language_model(model_name: str = "google/gemma-7b-it") -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        low_cpu_mem_usage=True, 
        quantization_config=config,
        attn_implementation="flash_attention_2"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

    return model, tokenizer