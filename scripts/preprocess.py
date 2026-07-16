"""Preprocess datasets for training."""

import os
import sys
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.movielens import MovieLensDataset
from datasets.goodreads import GoodreadsDataset
from datasets.google_reviews import GoogleReviewsDataset


DATASET_CLASSES = {
    'movielens-1m': MovieLensDataset,
    'movielens-100k': lambda **kwargs: MovieLensDataset(variant='100k', **kwargs),
    'goodreads': GoodreadsDataset,
    'google-reviews': GoogleReviewsDataset
}


def preprocess_dataset(name: str, data_dir: str = './data', seed: int = 42) -> None:
    """Preprocess a single dataset and save processed state."""
    print(f"\n{'='*60}")
    print(f"Preprocessing {name}")
    print(f"{'='*60}")

    if name not in DATASET_CLASSES:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(DATASET_CLASSES.keys())}")

    dataset_class = DATASET_CLASSES[name]
    dataset = dataset_class(data_dir=data_dir, seed=seed)

    try:
        dataset.load()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print(f"Please download the dataset first: python scripts/download_data.py --dataset {name}")
        return

    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"  Users: {dataset.n_users}")
    print(f"  Items: {dataset.n_items}")
    print(f"  Train interactions: {len(dataset.train_interactions):,}")
    print(f"  Test interactions: {len(dataset.test_interactions):,}")
    print(f"  Head items: {len(dataset.head_items)} ({100*len(dataset.head_items)/dataset.n_items:.1f}%)")
    print(f"  Mid items: {len(dataset.mid_items)} ({100*len(dataset.mid_items)/dataset.n_items:.1f}%)")
    print(f"  Tail items: {len(dataset.tail_items)} ({100*len(dataset.tail_items)/dataset.n_items:.1f}%)")

    # Save processed dataset
    save_name = name.replace('-', '_')
    save_path = os.path.join(data_dir, f'{save_name}_processed.pkl')
    os.makedirs(data_dir, exist_ok=True)
    dataset.save(save_path)

    print(f"\nSaved processed dataset to: {save_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Preprocess recommendation datasets')
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['all', 'movielens-1m', 'movielens-100k', 'goodreads', 'google-reviews'],
                       help='Dataset to preprocess')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory containing raw data')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    args = parser.parse_args()

    datasets = {
        'all': ['movielens-1m', 'goodreads', 'google-reviews'],
        'movielens-1m': ['movielens-1m'],
        'movielens-100k': ['movielens-100k'],
        'goodreads': ['goodreads'],
        'google-reviews': ['google-reviews']
    }[args.dataset]

    for name in datasets:
        try:
            preprocess_dataset(name, args.data_dir, args.seed)
        except Exception as e:
            print(f"ERROR processing {name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
