# datasets/__init__.py

from .base import BaseDataset, InteractionDataset
from .movielens import MovieLensDataset

__all__ = ['BaseDataset', 'InteractionDataset', 'MovieLensDataset']
