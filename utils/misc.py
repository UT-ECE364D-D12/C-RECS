import os
import random
from typing import Callable, Dict, List, Tuple, TypeVar, Union

import numpy as np
import torch
from torch import Tensor
from torchmetrics.functional import pairwise_cosine_similarity
from typing_extensions import ParamSpec

T = TypeVar("T")
P = ParamSpec("P")


def take_annotation_from(this: Callable[P, T]) -> Callable[[Callable], Callable[P, T]]:
    def decorator(real_function: Callable[P, T]) -> Callable[P, T]:
        real_function.__doc__ = this.__doc__
        return real_function

    return decorator


def send_to_device(object: Union[Tensor, Dict, List, Tuple], device: str = "cpu") -> Union[Tensor, Dict, List, Tuple]:
    """
    Send all tensors in an object to a device.

    Args:
        object (Union[Tensor, Dict, List, Tuple]): Object containing tensors.
        device (str, optional): Device to send the tensors to.

    Returns:
        Union[Tensor, Dict, List, Tuple]: Object with tensors on the specified device.
    """

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

    Args:
        x (Tensor): First tensor.
        y (Tensor): Second tensor.
    """

    return 1 - torch.cosine_similarity(x, y)


def pairwise_cosine_distance(x: Tensor, y: Tensor = None) -> Tensor:
    """
    Returns the pairwise cosine distance between two tensors.
    If y is None, it will return the pairwise cosine distance of x with itself.

    Args:
        x (Tensor): First tensor.
        y (Tensor, optional): Second tensor.
    """

    return 1 - pairwise_cosine_similarity(x, y, zero_diagonal=False)


def set_random_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): Random seed.
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
    Suppress warnings to clean up the output.
    """

    import logging

    logging.getLogger("bitsandbytes").setLevel(logging.CRITICAL)
    logging.getLogger("transformers").setLevel(logging.CRITICAL)

    import warnings

    warnings.filterwarnings("ignore")
