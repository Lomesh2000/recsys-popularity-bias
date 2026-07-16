"""Inference script to generate recommendations for a user.

Usage:
    python inference.py --checkpoint checkpoints/movielens-1m_std_0.25/best_model.pt --user_id 42 --top_k 10
"""

import os
import sys
import argparse
import torch

from utils.config import Config
from utils.logging_utils import get_device
from datasets.movielens import MovieLensDataset
from datasets.goodreads import GoodreadsDataset
from datasets.google_reviews import GoogleReviewsDataset
from models import BPRModel, ILEBPRModel


DATASET_CLASSES = {
    'movielens-1m': MovieLensDataset,
    'movielens-100k': lambda **kwargs: MovieLensDataset(variant='100k', **kwargs),
    'goodreads': GoodreadsDataset,
    'google-reviews': GoogleReviewsDataset
}


def parse_args():
    parser = argparse.ArgumentParser(description='Generate recommendations for a user')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--user_id', type=int, required=True,
                       help='Internal user ID (0-indexed)')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of recommendations to generate')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--data_dir', type=str, default='./data')
    return parser.parse_args()


def main():
    args = parse_args()

    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = Config(checkpoint['config'])

    # Load dataset
    dataset_class = DATASET_CLASSES[config.dataset.name]
    dataset = dataset_class(
        data_dir=args.data_dir,
        train_ratio=config.dataset.train_ratio,
        min_interactions=config.dataset.min_interactions,
        seed=config.seed
    )

    processed_path = os.path.join(
        args.data_dir,
        f"{config.dataset.name.replace('-', '_')}_processed.pkl"
    )

    if not os.path.exists(processed_path):
        print(f"ERROR: Processed dataset not found at {processed_path}")
        sys.exit(1)

    dataset.load_state(processed_path)

    # Validate user_id
    if args.user_id < 0 or args.user_id >= dataset.n_users:
        print(f"ERROR: user_id must be between 0 and {dataset.n_users - 1}")
        sys.exit(1)

    # Create and load model
    ModelClass = ILEBPRModel if config.loss.name == 'ile' else BPRModel
    model = ModelClass(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embedding_dim=config.model.embedding_dim,
        init_std=config.model.init_std
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Get user's training items
    train_items = set(dataset.train_dict.get(args.user_id, []))
    test_items = set(dataset.test_dict.get(args.user_id, []))

    # Generate scores for all non-training items
    candidate_items = [i for i in range(dataset.n_items) if i not in train_items]

    with torch.no_grad():
        user_tensor = torch.tensor([args.user_id] * len(candidate_items), device=device)
        item_tensor = torch.tensor(candidate_items, device=device)
        scores = model.predict(user_tensor, item_tensor)

    # Get top-k
    k = min(args.top_k, len(candidate_items))
    top_scores, top_indices = torch.topk(scores, k)
    top_items = [candidate_items[i] for i in top_indices.cpu().numpy()]
    top_scores = top_scores.cpu().numpy()

    # Build item group mapping
    item_groups = {}
    for item in dataset.head_items:
        item_groups[item] = 'H'
    for item in dataset.tail_items:
        item_groups[item] = 'T'
    for item in dataset.mid_items:
        item_groups[item] = 'M'

    # Print results
    print("\n" + "=" * 70)
    print(f"Top-{k} Recommendations for User {args.user_id}")
    print("=" * 70)
    print(f"{'Rank':>4} | {'Item ID':>8} | {'Group':>5} | {'Score':>10} | {'In Test?':>8}")
    print("-" * 70)

    for rank, (item, score) in enumerate(zip(top_items, top_scores), 1):
        group = item_groups.get(item, '?')
        in_test = "YES" if item in test_items else "no"
        print(f"{rank:>4} | {item:>8} | {group:>5} | {score:>10.4f} | {in_test:>8}")

    print("=" * 70)

    # Show user's profile distribution
    print(f"\nUser {args.user_id} Profile Distribution (Training Items):")
    profile_groups = {'H': 0, 'M': 0, 'T': 0}
    for item in train_items:
        g = item_groups.get(item, 'M')
        profile_groups[g] += 1

    total = sum(profile_groups.values())
    for g in ['H', 'M', 'T']:
        count = profile_groups[g]
        pct = 100 * count / total if total > 0 else 0
        bar = "█" * int(pct / 5)
        print(f"  {g}: {count:>5} ({pct:>5.1f}%) {bar}")

    # Show recommendation distribution
    print(f"\nRecommendation Distribution:")
    rec_groups = {'H': 0, 'M': 0, 'T': 0}
    for item in top_items:
        g = item_groups.get(item, 'M')
        rec_groups[g] += 1

    for g in ['H', 'M', 'T']:
        count = rec_groups[g]
        pct = 100 * count / k
        bar = "█" * int(pct / 5)
        print(f"  {g}: {count:>5} ({pct:>5.1f}%) {bar}")

    print()


if __name__ == '__main__':
    main()
