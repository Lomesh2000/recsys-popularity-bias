"""Evaluation script for trained models.

Usage:
    # Evaluate with default top_k (10)
    python evaluate.py --checkpoint checkpoints/movielens-1m_std_0.25/best_model.pt

    # Evaluate with custom top_k
    python evaluate.py --checkpoint checkpoints/movielens-1m_std_0.25/best_model.pt --top_k 5
    python evaluate.py --checkpoint checkpoints/movielens-1m_std_0.25/best_model.pt --top_k 20
"""

import os
import sys
import argparse
import torch

from utils.config import Config
from utils.logging_utils import setup_logger, get_device
from utils.seed import setup_reproducibility
from datasets.movielens import MovieLensDataset
from datasets.goodreads import GoodreadsDataset
from datasets.google_reviews import GoogleReviewsDataset
from models import BPRModel, ILEBPRModel
from metrics import Evaluator


DATASET_CLASSES = {
    'movielens-1m': MovieLensDataset,
    'movielens-100k': lambda **kwargs: MovieLensDataset(variant='100k', **kwargs),
    'goodreads': GoodreadsDataset,
    'google-reviews': GoogleReviewsDataset
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate trained recommendation model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate at k=10 (default, matches paper)
  python evaluate.py --checkpoint checkpoints/ile_std/best_model.pt

  # Evaluate at k=5
  python evaluate.py --checkpoint checkpoints/ile_std/best_model.pt --top_k 5

  # Evaluate at k=20
  python evaluate.py --checkpoint checkpoints/ile_std/best_model.pt --top_k 20
        """
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pt file)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Data directory')
    parser.add_argument('--top_k', type=int, default=None,
                       help='Cutoff for evaluation metrics (default: from config, typically 10)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load checkpoint
    device = get_device(args.device)
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = Config(checkpoint['config'])

    # Override top_k if provided
    if args.top_k is not None:
        config.set('evaluation.top_k', args.top_k)
        print(f"Using custom top_k: {args.top_k}")
    else:
        print(f"Using default top_k from config: {config.evaluation.top_k}")

    # Setup
    setup_reproducibility(config.seed, config.deterministic)
    logger = setup_logger('evaluate')

    logger.info("=" * 60)
    logger.info("Model Evaluation")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Dataset: {config.dataset.name}")
    logger.info(f"Evaluation top_k: {config.evaluation.top_k}")

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
        print("Run preprocessing first: python scripts/preprocess.py")
        sys.exit(1)

    dataset.load_state(processed_path)
    logger.info(f"Dataset loaded: {dataset.n_users} users, {dataset.n_items} items")

    # Create and load model
    ModelClass = ILEBPRModel if config.loss.name == 'ile' else BPRModel
    model = ModelClass(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embedding_dim=config.model.embedding_dim,
        init_std=config.model.init_std
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Model loaded: {ModelClass.__name__}")

    # Evaluate
    evaluator = Evaluator(dataset, k=config.evaluation.top_k, device=device)
    metrics = evaluator.evaluate(model)

    # Print results
    k = config.evaluation.top_k
    print("\n" + "=" * 60)
    print(f"Evaluation Results @ {k}")
    print("=" * 60)
    print(f"{'Metric':<15} {'Value':>10}")
    print("-" * 60)
    for metric, value in metrics.items():
        marker = ""
        if metric == 'nDCG':
            marker = "  (higher is better)"
        elif metric == 'UPD':
            marker = "  (lower is better)"
        elif metric in ('AD', 'EE'):
            marker = "  (higher is better)"
        print(f"{metric:<15} {value:>10.4f}{marker}")
    print("=" * 60)

    logger.info("Evaluation complete")


if __name__ == '__main__':
    main()
