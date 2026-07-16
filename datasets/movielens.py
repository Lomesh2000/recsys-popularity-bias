"""MovieLens dataset loader."""

import os
import pandas as pd
import numpy as np
from typing import Tuple
from .base import BaseDataset


class MovieLensDataset(BaseDataset):
    """MovieLens dataset (1M, 100K, etc.)."""

    def __init__(
        self,
        data_dir: str = './data',
        variant: str = '1m',
        **kwargs
    ):
        super().__init__(
            data_dir=data_dir,
            dataset_name=f'movielens-{variant}',
            **kwargs
        )
        self.variant = variant

    def load(self) -> None:
        """Load MovieLens data from ratings file."""
        data_path = os.path.join(self.data_dir, f'ml-{self.variant}')

        if self.variant == '1m':
            ratings_file = os.path.join(data_path, 'ratings.dat')
            ratings = pd.read_csv(
                ratings_file,
                sep='::',
                engine='python',
                names=['user_id', 'item_id', 'rating', 'timestamp'],
                encoding='latin-1'
            )
        elif self.variant == '100k':
            ratings_file = os.path.join(data_path, 'u.data')
            ratings = pd.read_csv(
                ratings_file,
                sep='\t',
                engine='python',
                names=['user_id', 'item_id', 'rating', 'timestamp']
            )
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

        # Use implicit feedback (all interactions)
        # Filter to minimum interactions
        user_counts = ratings['user_id'].value_counts()
        item_counts = ratings['item_id'].value_counts()

        valid_users = user_counts[user_counts >= self.min_interactions].index
        valid_items = item_counts[item_counts >= self.min_interactions].index

        ratings = ratings[
            ratings['user_id'].isin(valid_users) &
            ratings['item_id'].isin(valid_items)
        ]

        # Create ID mappings
        unique_users = sorted(ratings['user_id'].unique())
        unique_items = sorted(ratings['item_id'].unique())

        self.user_id_map = {u: i for i, u in enumerate(unique_users)}
        self.item_id_map = {i: j for j, i in enumerate(unique_items)}
        self.reverse_user_map = {i: u for u, i in self.user_id_map.items()}
        self.reverse_item_map = {j: i for i, j in self.item_id_map.items()}

        self.n_users = len(unique_users)
        self.n_items = len(unique_items)

        # Create interactions
        self.train_interactions = [
            (self.user_id_map[row['user_id']], self.item_id_map[row['item_id']])
            for _, row in ratings.iterrows()
        ]

        # Split into train/test
        self.split()
