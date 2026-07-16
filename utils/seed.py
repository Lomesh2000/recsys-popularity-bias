"""Random seed and reproducibility utilities."""

import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_deterministic(deterministic: bool = True) -> None:
    """Set deterministic behavior for reproducibility."""
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def setup_reproducibility(seed: int = 42, deterministic: bool = True) -> None:
    """Setup full reproducibility environment."""
    set_seed(seed)
    set_deterministic(deterministic)
