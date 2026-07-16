"""Recommendation evaluation metrics.

Implements the four metrics from the ILE paper:
- nDCG: Normalized Discounted Cumulative Gain (ranking quality)
- UPD: User Popularity Deviation (fairness)
- AD: Aggregate Diversity (fairness)
- EE: Equality of Exposure (fairness)
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from scipy.spatial.distance import jensenshannon


def compute_dcg(scores: np.ndarray) -> float:
    """Compute DCG for a single ranking.

    DCG = sum_{k=1}^{K} (2^rel_k - 1) / log2(k + 1)
    For binary relevance: rel_k ∈ {0, 1}, so 2^rel_k - 1 = rel_k

    Args:
        scores: Binary relevance scores at each position

    Returns:
        DCG value
    """
    positions = np.arange(1, len(scores) + 1)
    discounts = np.log2(positions + 1)
    return np.sum(scores / discounts)


def compute_ndcg(
    recommendations: Dict[int, List[int]],
    test_items: Dict[int, List[int]],
    k: int = 10
) -> float:
    """Compute nDCG@K averaged over users.

    Args:
        recommendations: Dict mapping user_id -> list of recommended item IDs (sorted by score)
        test_items: Dict mapping user_id -> list of test item IDs
        k: Cutoff for metric computation

    Returns:
        Average nDCG@K across users
    """
    ndcgs = []

    for user, recs in recommendations.items():
        if user not in test_items or len(test_items[user]) == 0:
            continue

        # Get top-k recommendations
        top_k = recs[:k]

        # Binary relevance: 1 if item is in test set, 0 otherwise
        relevance = np.array([1 if item in test_items[user] else 0 for item in top_k])

        # Compute DCG
        dcg = compute_dcg(relevance)

        # Compute ideal DCG (all test items at top positions)
        n_test = min(len(test_items[user]), k)
        ideal_relevance = np.zeros(k)
        ideal_relevance[:n_test] = 1
        idcg = compute_dcg(ideal_relevance)

        if idcg > 0:
            ndcgs.append(dcg / idcg)

    return np.mean(ndcgs) if ndcgs else 0.0


def compute_upd(
    recommendations: Dict[int, List[int]],
    train_items: Dict[int, List[int]],
    item_groups: Dict[int, str],
    k: int = 10
) -> float:
    """Compute User Popularity Deviation (UPD).

    Measures how well the distribution of (H, M, T) items in a user's
    profile matches the distribution in their recommendation list.
    Uses Jensen-Shannon Divergence (JSD).

    Args:
        recommendations: Dict mapping user_id -> recommended items
        train_items: Dict mapping user_id -> training items (profile)
        item_groups: Dict mapping item_id -> group ('H', 'M', 'T')
        k: Cutoff for recommendations

    Returns:
        Average UPD across users (lower is better, 0 = fairest)
    """
    jsds = []

    for user, recs in recommendations.items():
        if user not in train_items:
            continue

        # Profile distribution
        profile_items = train_items[user]
        profile_counts = {'H': 0, 'M': 0, 'T': 0}
        for item in profile_items:
            group = item_groups.get(item, 'M')
            profile_counts[group] += 1

        profile_total = sum(profile_counts.values())
        if profile_total == 0:
            continue

        profile_dist = np.array([
            profile_counts['H'] / profile_total,
            profile_counts['M'] / profile_total,
            profile_counts['T'] / profile_total
        ])

        # Recommendation distribution
        top_k = recs[:k]
        rec_counts = {'H': 0, 'M': 0, 'T': 0}
        for item in top_k:
            group = item_groups.get(item, 'M')
            rec_counts[group] += 1

        rec_total = sum(rec_counts.values())
        if rec_total == 0:
            continue

        rec_dist = np.array([
            rec_counts['H'] / rec_total,
            rec_counts['M'] / rec_total,
            rec_counts['T'] / rec_total
        ])

        # Jensen-Shannon Divergence
        jsd = jensenshannon(profile_dist, rec_dist)
        if not np.isnan(jsd):
            jsds.append(jsd)

    return np.mean(jsds) if jsds else 0.0


def compute_ad(
    recommendations: Dict[int, List[int]],
    n_items: int,
    k: int = 10
) -> float:
    """Compute Aggregate Diversity (AD).

    Percentage of items that appear at least once in recommendation lists.

    Args:
        recommendations: Dict mapping user_id -> recommended items
        n_items: Total number of items
        k: Cutoff for recommendations

    Returns:
        Aggregate diversity ratio [0, 1]
    """
    recommended_items = set()

    for recs in recommendations.values():
        recommended_items.update(recs[:k])

    return len(recommended_items) / n_items


def compute_ee(
    recommendations: Dict[int, List[int]],
    n_items: int,
    k: int = 10
) -> float:
    """Compute Equality of Exposure (EE).

    Measures how equal items are represented in recommendation lists.
    Exposure of item at position k: 1 / (1 + log(k))
    Uses 1 - Gini Index over exposure distribution.

    Args:
        recommendations: Dict mapping user_id -> recommended items
        n_items: Total number of items
        k: Cutoff for recommendations

    Returns:
        Equality of exposure [0, 1], 1 = fairest
    """
    # Compute exposure for each item
    exposure = np.zeros(n_items)

    for recs in recommendations.values():
        for pos, item in enumerate(recs[:k], start=1):
            exp = 1.0 / (1.0 + np.log(pos))
            exposure[item] += exp

    # Gini index
    def gini_index(x: np.ndarray) -> float:
        """Compute Gini index."""
        x = np.sort(x)
        n = len(x)
        cumsum = np.cumsum(x)
        return (2 * np.sum((np.arange(1, n + 1) * x))) / (n * cumsum[-1]) - (n + 1) / n

    if np.sum(exposure) == 0:
        return 0.0

    gini = gini_index(exposure)

    # Return 1 - Gini (so higher is better, consistent with paper)
    return 1.0 - gini


class Evaluator:
    """Evaluation engine for recommendation models."""

    def __init__(
        self,
        dataset,
        k: int = 10,
        device: str = 'cpu'
    ):
        self.dataset = dataset
        self.k = k
        self.device = device

        # Pre-compute item groups
        self.item_groups = {}
        for item in dataset.head_items:
            self.item_groups[item] = 'H'
        for item in dataset.tail_items:
            self.item_groups[item] = 'T'
        for item in dataset.mid_items:
            self.item_groups[item] = 'M'

    def evaluate(self, model) -> Dict[str, float]:
        """Evaluate model on all metrics.

        Args:
            model: Trained recommendation model

        Returns:
            Dictionary of metric values
        """
        model.eval()

        recommendations = {}
        all_items = torch.arange(self.dataset.n_items, device=self.device)

        with torch.no_grad():
            for user in range(self.dataset.n_users):
                # Skip users with no test items
                if user not in self.dataset.test_dict or len(self.dataset.test_dict[user]) == 0:
                    continue

                # Get items to rank: all items except training items
                train_items = set(self.dataset.train_dict.get(user, []))
                test_items = set(self.dataset.test_dict.get(user, []))

                # Items to rank: all non-training items
                candidate_items = [i for i in range(self.dataset.n_items) if i not in train_items]

                if len(candidate_items) == 0:
                    continue

                # Predict scores
                user_tensor = torch.tensor([user] * len(candidate_items), device=self.device)
                item_tensor = torch.tensor(candidate_items, device=self.device)

                scores = model.predict(user_tensor, item_tensor)

                # Get top-k recommendations
                _, top_indices = torch.topk(scores, min(self.k, len(candidate_items)))
                top_items = [candidate_items[i] for i in top_indices.cpu().numpy()]

                recommendations[user] = top_items

        # Compute metrics
        metrics = {}

        # nDCG
        metrics['nDCG'] = compute_ndcg(
            recommendations,
            self.dataset.test_dict,
            self.k
        )

        # UPD
        metrics['UPD'] = compute_upd(
            recommendations,
            self.dataset.train_dict,
            self.item_groups,
            self.k
        )

        # AD
        metrics['AD'] = compute_ad(
            recommendations,
            self.dataset.n_items,
            self.k
        )

        # EE
        metrics['EE'] = compute_ee(
            recommendations,
            self.dataset.n_items,
            self.k
        )

        return metrics
