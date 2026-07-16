"""BPR loss implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking loss.

    For a triplet (u, i, j) where i is positive and j is negative:
        L_BPR = -log(σ(ŷ_ui - ŷ_uj))

    where σ is the sigmoid function and ŷ_ui = p_u^T q_i.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        user_embeds: torch.Tensor,
        pos_embeds: torch.Tensor,
        neg_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Compute BPR loss.

        Args:
            user_embeds: User embeddings, shape (batch_size, embed_dim)
            pos_embeds: Positive item embeddings, shape (batch_size, embed_dim)
            neg_embeds: Negative item embeddings, shape (batch_size, embed_dim)

        Returns:
            BPR loss scalar (mean over batch)
        """
        # Compute scores
        pos_scores = torch.sum(user_embeds * pos_embeds, dim=1)  # (B,)
        neg_scores = torch.sum(user_embeds * neg_embeds, dim=1)  # (B,)

        # BPR loss: -log(sigmoid(pos_score - neg_score))
        # Using log-sigmoid for numerical stability
        loss = -F.logsigmoid(pos_scores - neg_scores)  # (B,)

        return loss  # Return per-sample losses for ILE computation
