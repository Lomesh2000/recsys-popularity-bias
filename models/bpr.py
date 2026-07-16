"""BPR (Bayesian Personalized Ranking) model."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class BPRModel(nn.Module):
    """Bayesian Personalized Ranking with matrix factorization.

    Paper: Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback", UAI 2009.

    Architecture:
        - User embeddings: P ∈ R^(n_users × embed_dim)
        - Item embeddings: Q ∈ R^(n_items × embed_dim)
        - Score: ŷ_ui = p_u^T q_i (dot product)
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 128,
        init_std: float = 0.01
    ):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        # Embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # Initialize
        self._init_weights(init_std)

    def _init_weights(self, std: float) -> None:
        """Initialize embeddings from normal distribution."""
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=std)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=std)

    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for a batch of triplets.

        Args:
            users: User IDs, shape (batch_size,)
            pos_items: Positive item IDs, shape (batch_size,)
            neg_items: Negative item IDs, shape (batch_size,)

        Returns:
            Tuple of (user_embeds, pos_embeds, neg_embeds)
        """
        user_embeds = self.user_embedding(users)      # (B, D)
        pos_embeds = self.item_embedding(pos_items)   # (B, D)
        neg_embeds = self.item_embedding(neg_items)   # (B, D)

        return user_embeds, pos_embeds, neg_embeds

    def predict(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Predict scores for user-item pairs.

        Args:
            users: User IDs, shape (n_pairs,)
            items: Item IDs, shape (n_pairs,)

        Returns:
            Scores, shape (n_pairs,)
        """
        user_embeds = self.user_embedding(users)  # (N, D)
        item_embeds = self.item_embedding(items)  # (N, D)

        scores = torch.sum(user_embeds * item_embeds, dim=1)  # (N,)
        return scores

    def get_user_embedding(self, user_id: int) -> torch.Tensor:
        """Get embedding for a single user."""
        return self.user_embedding.weight[user_id]

    def get_item_embedding(self, item_id: int) -> torch.Tensor:
        """Get embedding for a single item."""
        return self.item_embedding.weight[item_id]

    def get_all_item_embeddings(self) -> torch.Tensor:
        """Get all item embeddings."""
        return self.item_embedding.weight
