import math
from typing import Dict, List, Union

import torch.optim as optim


class CosineAnnealingWarmRestarts:
    """
    A learning rate scheduler that implements the Cosine Annealing with Warm Restarts (Cyclical Learning Rates) algorithm.

    For each cycle, defined by the period, the learning rate is first increased from min_lr to max_lr in a linear fashion
    during the warmup_steps. Then, it is decayed from max_lr to min_lr using a cosine annealing schedule.

    Args:
        optimizer (optim.Optimizer): The optimizer to update the learning rates.
        period (int): The number of steps per cycle.
        min_lr (Union[float, List[float], Dict[str, float]]): The minimum learning rate for each parameter group.
        max_lr (Union[float, List[float], Dict[str, float]]): The maximum learning rate for each parameter group.
        warmup_steps (int, optional): The number of steps to linearly increase the learning rate.
        restart_decay (float, optional): The factor to decay max_lr after each cycle.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        period: int,
        min_lr: Union[float, List[float], Dict[str, float]],
        max_lr: Union[float, List[float], Dict[str, float]],
        warmup_steps: int = 0,
        restart_decay: float = 1.0,
    ) -> None:
        self.optimizer = optimizer
        self.period = period
        self.warmup_steps = warmup_steps
        self.restart_decay = restart_decay
        self.current_step = 0
        self.restart_count = 0

        # Assign a minimum and maximum learning rate for each parameter group
        self.min_lr = self._convert_to_list(min_lr, "min_lr")
        self.max_lr = self._convert_to_list(max_lr, "max_lr")

        # Validate that min_lr and max_lr are valid for each parameter group
        self._validate_lr_ranges()

        # Initialize learning rates for all parameter groups
        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr

    def step(self) -> Dict[str, float]:
        """
        Update learning rates for all parameter groups.

        Returns:
            Dict[str, float]: The learning rate for each parameter group.
        """
        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr

        self.current_step += 1

        # Check if a restart is needed
        if self.current_step >= self.period:
            self.current_step = 0
            self.restart_count += 1
            # Decay max_lr for each parameter group
            self.max_lr = [max_lr * self.restart_decay for max_lr in self.max_lr]

        return {
            f"{param_group.get('name', i)}.lr".capitalize(): param_group["lr"] for i, param_group in enumerate(self.optimizer.param_groups)
        }

    def get_lr(self) -> List[float]:
        """
        Calculate the learning rate for each parameter group.

        Returns:
            List[float]: The learning rate for each parameter group.
        """

        lrs = []
        for min_lr, max_lr in zip(self.min_lr, self.max_lr):
            if self.current_step < self.warmup_steps and self.warmup_steps > 0:
                # Warmup phase: linearly increase from min_lr to max_lr
                lr = min_lr + (max_lr - min_lr) * self.current_step / self.warmup_steps
            else:
                # Cosine annealing phase: decay from max_lr to min_lr
                steps_into_cycle = self.current_step - self.warmup_steps
                lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * steps_into_cycle / (self.period - self.warmup_steps)))
            lrs.append(lr)
        return lrs

    def _convert_to_list(self, lr: Union[float, List[float], Dict[str, float]], name: str) -> List[float]:
        """
        Convert a learning rate to a list of learning rates for each parameter group.

        Args:
            lr (Union[float, List[float], Dict[str, float]]): The learning rate(s) to convert.
            name (str): The name of the learning rate for error messages.

        Returns:
            List[float]: A list of learning rates for each parameter group.
        """

        if isinstance(lr, (int, float)):  # If lr is a single float, apply it to all parameter groups
            return [lr] * len(self.optimizer.param_groups)
        elif isinstance(lr, list):  # If lr is a list, ensure it matches the number of parameter groups
            assert len(lr) == len(self.optimizer.param_groups), f"Length of {name} must match the number of parameter groups."
            return lr
        elif isinstance(lr, dict):  # If lr is a dictionary, map parameter group names to learning rates
            lr_list = []
            for param_group in self.optimizer.param_groups:
                assert "name" in param_group, f"Parameter group must have a 'name' field when {name} is a dictionary."
                assert param_group["name"] in lr, f"Parameter group '{param_group['name']}' not found in {name}."
                lr_list.append(lr[param_group["name"]])
            return lr_list
        else:
            raise TypeError(f"{name} must be a float, a list of floats, or a dictionary mapping parameter group names to floats.")

    def _validate_lr_ranges(self) -> None:
        """
        Ensure that min_lr is less than or equal to max_lr for each parameter group.
        """

        for min_lr, max_lr in zip(self.min_lr, self.max_lr):
            assert min_lr <= max_lr, f"min_lr must be less than or equal to max_lr for each parameter group."
