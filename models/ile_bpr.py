"""ILE-BPR model with Item Loss Equalization."""

import torch
import torch.nn as nn
from typing import Optional
from .bpr import BPRModel


class ILEBPRModel(BPRModel):
    """BPR model with Item Loss Equalization.

    Extends base BPR with group-aware loss computation.
    The model itself is identical to BPR; ILE is applied in the loss function.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 128,
        init_std: float = 0.01
    ):
        super().__init__(n_users, n_items, embedding_dim, init_std)
        # Model architecture is identical to BPR
        # ILE is applied at the loss level
