"""Item Loss Equalization (ILE) implementation."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Literal
from .bpr_loss import BPRLoss


class DistanceFunction:
    """Distance functions for ILE loss equalization.

    All functions measure disparity across group losses.
    Lower values indicate more equal (fairer) losses.
    """

    @staticmethod
    def std(group_losses: torch.Tensor, eps = 1e-8) -> torch.Tensor:
        """Standard deviation of group losses.

        Formula: sqrt(mean((L_g - mean(L_g))^2))

        Args:
            group_losses: Tensor of shape (n_groups,)
            eps: Small constant for numerical stability (will be cast to float)

        Returns:
            Standard deviation scalar
        """
        return torch.std(group_losses)

    @staticmethod
    def mad(group_losses: torch.Tensor, eps = 1e-8) -> torch.Tensor:
        """Mean absolute deviation of group losses.

        Formula: mean(|L_g - mean(L_g)|)

        Args:
            group_losses: Tensor of shape (n_groups,)
            eps: Small constant for numerical stability (will be cast to float)

        Returns:
            Mean absolute deviation scalar
        """
        mean_loss = torch.mean(group_losses)
        return torch.mean(torch.abs(group_losses - mean_loss))

    @staticmethod
    def entropy(group_losses: torch.Tensor, eps = 1e-8) -> torch.Tensor:
        """Entropy of normalized group losses.

        Formula: -sum(p_g * log(p_g)) where p_g = L_g / sum(L_g)

        Args:
            group_losses: Tensor of shape (n_groups,)
            eps: Small constant for numerical stability (will be cast to float)

        Returns:
            Entropy scalar. Lower entropy = more equal losses.
            Note: We return entropy directly (not negated), so minimizing 
            entropy maximizes equality (peaked distribution = equal losses).
        """
        # Robust: cast eps to float in case it comes as string from config
        eps = float(eps)

        # Normalize to probabilities
        probs = group_losses / (torch.sum(group_losses) + eps)
        probs = torch.clamp(probs, min=eps)  # Avoid log(0)

        entropy = -torch.sum(probs * torch.log(probs))
        return entropy


class ILELoss(nn.Module):
    """Item Loss Equalization loss.

    Implements Equation 2 from the paper:
        L* = L + λ * D({L_g | g ∈ G})

    where:
        - L is the average BPR loss over all training interactions (Eq. 1)
        - L_g is the average loss of items in group g (SIMPLE AVERAGE, per batch)
        - D is a distance function (STD, MAD, or ENT) from Table 1
        - λ controls the fairness-accuracy trade-off
        - G = {H, M, T} (Head, Mid, Tail item groups)

    NOTE: The paper specifies simple average per batch, NOT running/exponential average.
    We compute L_g as the mean BPR loss for group g within the CURRENT batch only.
    If a group has no samples in the batch, we use the global batch average as fallback.
    """

    def __init__(
        self,
        distance: Literal['std', 'mad', 'ent'] = 'std',
        lambda_ile: float = 0.25,
        eps: float = 1e-8,
        n_groups: int = 3
    ):
        super().__init__()

        self.bpr_loss = BPRLoss()
        self.distance_fn = distance
        self.lambda_ile = lambda_ile
        self.eps = float(eps)  # Ensure float, not string
        self.n_groups = n_groups

        # Map distance name to function
        self.distance_functions = {
            'std': DistanceFunction.std,
            'mad': DistanceFunction.mad,
            'ent': DistanceFunction.entropy
        }

        if distance not in self.distance_functions:
            raise ValueError(f"Unknown distance: {distance}. Choose from {list(self.distance_functions.keys())}")

    def forward(
        self,
        user_embeds: torch.Tensor,
        pos_embeds: torch.Tensor,
        neg_embeds: torch.Tensor,
        group_masks: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Compute ILE-augmented BPR loss.

        Args:
            user_embeds: User embeddings, shape (batch_size, embed_dim)
            pos_embeds: Positive item embeddings, shape (batch_size, embed_dim)
            neg_embeds: Negative item embeddings, shape (batch_size, embed_dim)
            group_masks: Dict mapping group name ('H', 'M', 'T') to boolean mask
                        Each mask has shape (batch_size,)

        Returns:
            Tuple of (total_loss, loss_info_dict)
            - total_loss: Scalar tensor
            - loss_info: Dict with 'bpr_loss', 'distance', 'ile_loss', 'group_losses'
        """
        # Compute per-sample BPR losses
        bpr_losses = self.bpr_loss(user_embeds, pos_embeds, neg_embeds)  # (B,)

        # Compute base BPR loss (mean over batch) = L in Eq. 2
        bpr_loss = torch.mean(bpr_losses)

        # Compute group losses L_g = mean BPR loss for items in group g
        # Paper: "L_g is the average loss of items in group g"
        group_losses = []
        group_loss_values = {}

        for group_name in ['H', 'M', 'T']:
            mask = group_masks[group_name]  # (B,)

            if mask.sum() > 0:
                # Simple average for this group in CURRENT batch (per paper)
                group_loss = torch.mean(bpr_losses[mask])
            else:
                # Fallback: use global batch average if group absent from batch
                group_loss = bpr_loss

            group_losses.append(group_loss)
            group_loss_values[group_name] = group_loss.item()

        group_losses_tensor = torch.stack(group_losses)  # (3,)

        # Compute distance D({L_g | g ∈ G}) per Table 1
        distance_fn = self.distance_functions[self.distance_fn]
        distance = distance_fn(group_losses_tensor, eps=self.eps)

        # Total loss: L* = L + λ * D  (Eq. 2)
        ile_term = self.lambda_ile * distance
        total_loss = bpr_loss + ile_term

        loss_info = {
            'bpr_loss': bpr_loss.item(),
            'distance': distance.item(),
            'ile_term': ile_term.item(),
            'total_loss': total_loss.item(),
            'group_losses': group_loss_values
        }

        return total_loss, loss_info
