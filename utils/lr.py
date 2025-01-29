import math
from typing import List, Union

from torch import optim


# TODO: Document, update per-param lr
class CosineAnnealingWarmup:
    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int, period: int, min_lr: float = 1e-6, max_lr: float = 1e-3) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = period
        self.min_lr = min_lr
        self.max_lr = max_lr

        self.current_step = 0

        lr = self.get_lr()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step(self) -> None:
        lr = self.get_lr()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        self.current_step += 1

    def get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            return self.min_lr + (self.max_lr - self.min_lr) * self.current_step / (self.warmup_steps - 1)
        else:
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * (self.current_step - self.warmup_steps) / (self.max_steps)))
