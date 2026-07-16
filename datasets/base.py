"""Base dataset classes for recommendation."""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import os
import pickle


class BaseDataset:
    """Base class for recommendation datasets."""

    def __init__(
        self,
        data_dir: str = './data',
        dataset_name: str = 'movielens-1m',
        train_ratio: float = 0.8,
        min_interactions: int = 5,
        seed: int = 42
    ):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.train_ratio = train_ratio
        self.min_interactions = min_interactions
        self.seed = seed

        self.n_users = 0
        self.n_items = 0
        self.train_interactions = []  # List of (user, item) tuples
        self.test_interactions = []
        self.train_dict = defaultdict(list)  # user -> [items]
        self.test_dict = defaultdict(list)
        self.item_popularity = None  # interaction count per item

        # Popularity groups
        self.head_items = None
        self.mid_items = None
        self.tail_items = None

        # Mappings
        self.user_id_map = {}  # original -> internal
        self.item_id_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}

    def load(self) -> None:
        """Load and preprocess dataset."""
        raise NotImplementedError

    def split(self) -> None:
        """Split interactions into train/test."""
        np.random.seed(self.seed)

        # Group interactions by user
        user_items = defaultdict(list)
        for u, i in self.train_interactions:
            user_items[u].append(i)

        self.train_interactions = []
        self.test_interactions = []
        self.train_dict = defaultdict(list)
        self.test_dict = defaultdict(list)

        for user, items in user_items.items():
            if len(items) < self.min_interactions:
                # Put all in train if not enough interactions
                self.train_interactions.extend([(user, i) for i in items])
                self.train_dict[user] = items
                continue

            n_train = max(1, int(len(items) * self.train_ratio))
            np.random.shuffle(items)

            train_items = items[:n_train]
            test_items = items[n_train:]

            self.train_interactions.extend([(user, i) for i in train_items])
            self.test_interactions.extend([(user, i) for i in test_items])
            self.train_dict[user] = train_items
            self.test_dict[user] = test_items

        # Compute item popularity from training set
        self.item_popularity = np.zeros(self.n_items, dtype=np.int32)
        for u, i in self.train_interactions:
            self.item_popularity[i] += 1

        # Create popularity groups
        self._create_popularity_groups()

    def _create_popularity_groups(self) -> None:
        """Create Head/Mid/Tail item groups."""
        from utils.data_utils import get_popularity_groups

        self.head_items, self.mid_items, self.tail_items = get_popularity_groups(
            self.item_popularity,
            head_ratio=0.2,
            tail_ratio=0.2
        )

    def get_group_mask(self, items: np.ndarray) -> Dict[str, np.ndarray]:
        """Get boolean mask for items belonging to each group.

        Args:
            items: Array of item IDs

        Returns:
            Dictionary mapping group name to boolean mask
        """
        head_set = set(self.head_items)
        mid_set = set(self.mid_items)
        tail_set = set(self.tail_items)

        return {
            'H': np.array([i in head_set for i in items]),
            'M': np.array([i in mid_set for i in items]),
            'T': np.array([i in tail_set for i in items])
        }

    def get_item_group(self, item_id: int) -> str:
        """Get group label for an item."""
        if item_id in self.head_items:
            return 'H'
        elif item_id in self.tail_items:
            return 'T'
        else:
            return 'M'

    def save(self, path: str) -> None:
        """Save dataset state."""
        state = {
            'n_users': self.n_users,
            'n_items': self.n_items,
            'train_interactions': self.train_interactions,
            'test_interactions': self.test_interactions,
            'train_dict': dict(self.train_dict),
            'test_dict': dict(self.test_dict),
            'item_popularity': self.item_popularity,
            'head_items': self.head_items,
            'mid_items': self.mid_items,
            'tail_items': self.tail_items,
            'user_id_map': self.user_id_map,
            'item_id_map': self.item_id_map,
            'reverse_user_map': self.reverse_user_map,
            'reverse_item_map': self.reverse_item_map
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, path: str) -> None:
        """Load dataset state."""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.n_users = state['n_users']
        self.n_items = state['n_items']
        self.train_interactions = state['train_interactions']
        self.test_interactions = state['test_interactions']
        self.train_dict = defaultdict(list, state['train_dict'])
        self.test_dict = defaultdict(list, state['test_dict'])
        self.item_popularity = state['item_popularity']
        self.head_items = state['head_items']
        self.mid_items = state['mid_items']
        self.tail_items = state['tail_items']
        self.user_id_map = state['user_id_map']
        self.item_id_map = state['item_id_map']
        self.reverse_user_map = state['reverse_user_map']
        self.reverse_item_map = state['reverse_item_map']


class InteractionDataset(Dataset):
    """PyTorch Dataset for BPR training."""

    def __init__(
        self,
        interactions: List[Tuple[int, int]],
        n_items: int,
        n_negatives: int = 1,
        seed: int = 42
    ):
        self.interactions = interactions
        self.n_items = n_items
        self.n_negatives = n_negatives

        # Build user-item dictionary for negative sampling
        self.user_items = defaultdict(set)
        for u, i in interactions:
            self.user_items[u].add(i)

        np.random.seed(seed)

    def __len__(self) -> int:
        return len(self.interactions)

    def __getitem__(self, idx: int) -> Tuple[int, int, int]:
        """Get a single triplet (user, positive_item, negative_item)."""
        user, pos_item = self.interactions[idx]

        # Sample negative
        neg_item = self._sample_negative(user)

        return user, pos_item, neg_item

    def _sample_negative(self, user: int) -> int:
        """Sample a negative item for a user."""
        pos_items = self.user_items[user]

        while True:
            neg_item = np.random.randint(0, self.n_items)
            if neg_item not in pos_items:
                return neg_item
