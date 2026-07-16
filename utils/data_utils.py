"""Data processing utilities."""

import numpy as np
import torch
from typing import List, Tuple, Dict
from collections import defaultdict


def create_interaction_matrix(
    interactions: List[Tuple[int, int]],
    n_users: int,
    n_items: int
) -> np.ndarray:
    """Create binary interaction matrix."""
    matrix = np.zeros((n_users, n_items), dtype=np.float32)
    for u, i in interactions:
        matrix[u, i] = 1.0
    return matrix


def get_popularity_groups(
    item_interaction_counts: np.ndarray,
    head_ratio: float = 0.2,
    tail_ratio: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split items into Head, Mid, Tail groups by popularity.

    Args:
        item_interaction_counts: Array of interaction counts per item
        head_ratio: Fraction of items in Head group
        tail_ratio: Fraction of items in Tail group

    Returns:
        Tuple of (head_indices, mid_indices, tail_indices)
    """
    n_items = len(item_interaction_counts)
    sorted_indices = np.argsort(-item_interaction_counts)  # descending

    n_head = int(n_items * head_ratio)
    n_tail = int(n_items * tail_ratio)

    head_indices = sorted_indices[:n_head]
    tail_indices = sorted_indices[-n_tail:]
    mid_indices = sorted_indices[n_head:-n_tail]

    return head_indices, mid_indices, tail_indices


def create_user_negative_pool(
    train_interactions: Dict[int, List[int]],
    n_items: int
) -> Dict[int, List[int]]:
    """Create pool of negative items for each user."""
    all_items = set(range(n_items))
    negative_pool = {}

    for user, pos_items in train_interactions.items():
        pos_set = set(pos_items)
        negative_pool[user] = list(all_items - pos_set)

    return negative_pool


def sample_negatives(
    users: np.ndarray,
    train_dict: Dict[int, List[int]],
    n_items: int,
    n_negatives: int = 1
) -> np.ndarray:
    """Sample negative items for given users.

    Args:
        users: Array of user IDs
        train_dict: Dictionary mapping user -> positive items
        n_items: Total number of items
        n_negatives: Number of negatives per positive

    Returns:
        Array of negative item IDs
    """
    negatives = []
    for u in users:
        pos_items = set(train_dict[int(u)])
        neg_candidates = list(set(range(n_items)) - pos_items)

        if len(neg_candidates) < n_negatives:
            # If not enough negatives, sample with replacement
            neg = np.random.choice(neg_candidates, size=n_negatives, replace=True)
        else:
            neg = np.random.choice(neg_candidates, size=n_negatives, replace=False)

        negatives.extend(neg)

    return np.array(negatives)
