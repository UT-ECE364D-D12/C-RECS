import os
import random

import numpy as np
import torch
from torch import Tensor
from torchmetrics.functional import pairwise_cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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


def build_language_model(
    model_name: str = "google/gemma-7b-it",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
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
        attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = (
        tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    )

    return model, tokenizer
