from typing import Dict

from torch import nn


def get_model_statistics(model: nn.Module, norm_type: int = 2) -> Dict[str, Dict[str, float]]:
    """
    Get the parameter and gradient statistics of a model.

    Args:
        model (nn.Module): Model to get the statistics from.
        norm_type (int, optional): The type of norm to use, optional.

    Returns:
        stats Dict[str, Dict[str, float]]: Dictionary containing the parameter and gradient statistics.
    """

    num_params = 0
    num_active_params = 0
    parameter_mean = 0.0
    parameter_max = 0.0
    parameter_norm = 0.0
    gradient_mean = 0.0
    gradient_max = 0.0
    gradient_norm = 0.0

    for param in model.parameters():
        # Parameter stats
        parameter_mean += param.detach().data.abs().sum().item()
        parameter_max = max(parameter_max, param.detach().data.abs().max().item())
        parameter_norm += param.detach().data.norm(norm_type).item() ** norm_type

        num_params += param.numel()

        # Gradient stats
        if param.grad is not None and param.requires_grad:
            gradient_mean += param.grad.detach().data.abs().sum().item()
            gradient_max = max(gradient_max, param.grad.detach().data.abs().max().item())
            gradient_norm += param.grad.detach().data.norm(norm_type).item() ** norm_type
            num_active_params += param.numel()

    return {
        "Parameter": {
            "Norm": parameter_norm ** (1.0 / norm_type),
            "Mean": parameter_mean / num_params,
            "Max": parameter_max,
        },
        "Gradient": {
            "Norm": gradient_norm ** (1.0 / norm_type),
            "Mean": gradient_mean / num_active_params,
            "Max": gradient_max,
        },
    }
