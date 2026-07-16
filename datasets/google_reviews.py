"""Google Reviews dataset loader."""

import os
import pandas as pd
import numpy as np
from .base import BaseDataset


class GoogleReviewsDataset(BaseDataset):
    """Google Reviews (Local) dataset."""

    def __init__(
        self,
        data_dir: str = './data',
        **kwargs
    ):
        super().__init__(
            data_dir=data_dir,
            dataset_name='google-reviews',
            **kwargs
        )

    def load(self) -> None:
        """Load Google Reviews data.

        Assumes CSV file with columns: user_id, business_id, rating
        """
        data_path = os.path.join(self.data_dir, 'google-reviews')

        possible_files = [
            'google_reviews.csv',
            'reviews.csv',
            'ratings.csv',
            'google_local_ratings.csv'
        ]
        ratings_file = None

        for fname in possible_files:
            fpath = os.path.join(data_path, fname)
            if os.path.exists(fpath):
                ratings_file = fpath
                break

        if ratings_file is None:
            raise FileNotFoundError(
                f"Could not find Google Reviews data in {data_path}. "
                f"Expected one of: {possible_files}"
            )

        ratings = pd.read_csv(ratings_file)

        # Map column names
        if 'business_id' in ratings.columns and 'item_id' not in ratings.columns:
            ratings = ratings.rename(columns={'business_id': 'item_id'})

        # Use implicit feedback (all interactions with rating >= 3)
        if 'rating' in ratings.columns:
            ratings = ratings[ratings['rating'] >= 3]

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

        self.split()
