"""Main training script for BPR and ILE-BPR models.

Usage:
    # Train base BPR on MovieLens-1M
    python train.py --dataset movielens-1m --distance none --lambda_ile 0.0

    # Train ILE-BPR with STD distance
    python train.py --dataset movielens-1m --distance std --lambda_ile 0.25

    # Train with custom early stopping
    python train.py --dataset movielens-1m --distance std --lambda_ile 0.25 \
        --early_stopping --patience 15 --early_stop_metric UPD --early_stop_mode min
"""

import os
import sys
import argparse
import torch

from utils.config import Config
from utils.seed import setup_reproducibility
from utils.logging_utils import setup_logger, get_device
from datasets.movielens import MovieLensDataset
from datasets.goodreads import GoodreadsDataset
from datasets.google_reviews import GoogleReviewsDataset
from models import BPRModel, ILEBPRModel
from trainers import BPRTrainer


DATASET_CLASSES = {
    'movielens-1m': MovieLensDataset,
    'movielens-100k': lambda **kwargs: MovieLensDataset(variant='100k', **kwargs),
    'goodreads': GoodreadsDataset,
    'google-reviews': GoogleReviewsDataset
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train BPR or ILE-BPR model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Base BPR
  python train.py --dataset movielens-1m --distance none

  # ILE-BPR with STD (paper: λ=0.25 for ML-1M)
  python train.py --dataset movielens-1m --distance std --lambda_ile 0.25

  # ILE-BPR with ENT (paper: λ=0.04 for ML-1M)
  python train.py --dataset movielens-1m --distance ent --lambda_ile 0.04

  # ILE-BPR with MAD (paper: λ=0.30 for ML-1M)
  python train.py --dataset movielens-1m --distance mad --lambda_ile 0.30

  # With early stopping on nDCG (maximize, patience=20)
  python train.py --dataset movielens-1m --distance std --lambda_ile 0.25 \
      --early_stopping --patience 20 --early_stop_metric nDCG --early_stop_mode max

  # With early stopping on UPD (minimize, patience=15)
  python train.py --dataset movielens-1m --distance std --lambda_ile 0.25 \
      --early_stopping --patience 15 --early_stop_metric UPD --early_stop_mode min
        """
    )
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Base configuration file')
    parser.add_argument('--dataset', type=str, default='movielens-1m',
                       choices=list(DATASET_CLASSES.keys()),
                       help='Dataset to use')
    parser.add_argument('--distance', type=str, default='std',
                       choices=['none', 'std', 'mad', 'ent'],
                       help='ILE distance function (none = base BPR)')
    parser.add_argument('--lambda_ile', type=float, default=None,
                       help='Weight for ILE term (λ)')
    parser.add_argument('--embedding_dim', type=int, default=None,
                       help='Embedding dimension (default: 128)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    parser.add_argument('--deterministic', action='store_true',
                       help='Enable deterministic mode')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name (used for checkpoint directory)')

    parser.add_argument('--top_k', type=int, default=None,
                       help='Cutoff for evaluation metrics (default: from config, typically 10)')

    # Early stopping arguments
    # Early stopping arguments
    parser.add_argument('--early_stopping', action='store_true',
                       help='Enable early stopping')
    parser.add_argument('--no_early_stopping', action='store_true',
                       help='Disable early stopping (overrides config)')
    parser.add_argument('--patience', type=int, default=None,
                       help='Early stopping patience (epochs to wait before stopping)')
    parser.add_argument('--early_stop_metric', type=str, default=None,
                       choices=['nDCG', 'UPD', 'AD', 'EE', 'loss'],
                       help='Metric to monitor for early stopping')
    parser.add_argument('--early_stop_mode', type=str, default=None,
                       choices=['min', 'max'],
                       help='Mode: min (stop when metric stops decreasing) or max (stop when metric stops increasing)')
    parser.add_argument('--min_delta', type=float, default=None,
                       help='Minimum change to qualify as an improvement')

    return parser.parse_args()


def main():
    args = parse_args()

    # Load base config
    config = Config.from_yaml(args.config)

    # Override with command line args
    if args.dataset:
        config.set('dataset.name', args.dataset)
    if args.distance is not None:
        if args.distance == 'none':
            config.set('loss.name', 'bpr')
            config.set('loss.distance', 'none')
            config.set('loss.lambda_ile', 0.0)
        else:
            config.set('loss.name', 'ile')
            config.set('loss.distance', args.distance)
    if args.lambda_ile is not None:
        config.set('loss.lambda_ile', args.lambda_ile)
    if args.embedding_dim is not None:
        config.set('model.embedding_dim', args.embedding_dim)
    if args.epochs is not None:
        config.set('training.epochs', args.epochs)
    if args.batch_size is not None:
        config.set('training.batch_size', args.batch_size)
    if args.learning_rate is not None:
        config.set('training.learning_rate', args.learning_rate)
    if args.seed is not None:
        config.set('seed', args.seed)
    if args.deterministic:
        config.set('deterministic', True)

    # Evaluation top_k override
    if args.top_k is not None:
        config.set('evaluation.top_k', args.top_k)

        # Early stopping overrides
    if args.early_stopping:
        config.set('training.early_stopping.enabled', True)
    if args.no_early_stopping:
        config.set('training.early_stopping.enabled', False)
    if args.patience is not None:
        config.set('training.early_stopping.patience', args.patience)
    if args.early_stop_metric is not None:
        config.set('training.early_stopping.metric', args.early_stop_metric)
    if args.early_stop_mode is not None:
        config.set('training.early_stopping.mode', args.early_stop_mode)
    if args.min_delta is not None:
        config.set('training.early_stopping.min_delta', args.min_delta)

    # Setup reproducibility
    setup_reproducibility(config.seed, config.deterministic)

    # Setup device
    device = get_device(args.device if args.device != 'auto' else 'auto')
    print(f"Using device: {device}")

    # Setup experiment name and directories
    if args.name:
        exp_name = args.name
    else:
        dist = config.loss.distance
        lam = config.loss.lambda_ile
        exp_name = f"{config.dataset.name}_{dist}_{lam}"

    checkpoint_dir = os.path.join(config.logging.checkpoint_dir, exp_name)
    config.set('logging.checkpoint_dir', checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_dir = config.logging.log_dir
    os.makedirs(log_dir, exist_ok=True)

    logger = setup_logger('train', log_dir, f'{exp_name}.log')
    logger.info("=" * 60)
    logger.info("ILE-BPR Training Started")
    logger.info("=" * 60)
    logger.info(f"Configuration: {config.to_dict()}")

    # Print early stopping config
    es_enabled = config.training.early_stopping.enabled
    logger.info(f"Early stopping: {'ENABLED' if es_enabled else 'DISABLED'}")
    if es_enabled:
        logger.info(f"  Patience: {config.training.early_stopping.patience}")
        logger.info(f"  Metric: {config.training.early_stopping.metric}")
        logger.info(f"  Mode: {config.training.early_stopping.mode}")
        if config.training.early_stopping.get('min_delta'):
            logger.info(f"  Min delta: {config.training.early_stopping.min_delta}")

    # Load dataset
    logger.info(f"Loading dataset: {config.dataset.name}")
    dataset_class = DATASET_CLASSES[config.dataset.name]
    dataset = dataset_class(
        data_dir=config.dataset.data_dir,
        train_ratio=config.dataset.train_ratio,
        min_interactions=config.dataset.min_interactions,
        seed=config.seed
    )

    # Try to load preprocessed, otherwise load raw
    processed_path = os.path.join(
        config.dataset.data_dir,
        f"{config.dataset.name.replace('-', '_')}_processed.pkl"
    )

    if os.path.exists(processed_path):
        logger.info(f"Loading preprocessed dataset from {processed_path}")
        dataset.load_state(processed_path)
    else:
        logger.info("Processing raw data...")
        dataset.load()
        dataset.save(processed_path)
        logger.info(f"Saved processed dataset to {processed_path}")

    logger.info(f"Dataset loaded: {dataset.n_users} users, {dataset.n_items} items")
    logger.info(f"Train: {len(dataset.train_interactions):,}, Test: {len(dataset.test_interactions):,}")

    # Create model
    if config.loss.name == 'ile':
        model = ILEBPRModel(
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            embedding_dim=config.model.embedding_dim,
            init_std=config.model.init_std
        )
    else:
        model = BPRModel(
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            embedding_dim=config.model.embedding_dim,
            init_std=config.model.init_std
        )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Parameters: {n_params:,}")
    logger.info(f"Embedding dim: {config.model.embedding_dim}")

    # Create trainer
    trainer = BPRTrainer(model, dataset, config, device, logger)

    # Train
    logger.info("Starting training...")
    history = trainer.train()

    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)

    # Print final results
    if history['val_metrics']:
        final_metrics = history['val_metrics'][-1]
        logger.info("Final Evaluation Metrics:")
        for metric, value in final_metrics.items():
            logger.info(f"  {metric:10s}: {value:.4f}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    if trainer.early_stopping and trainer.best_epoch > 0:
        print(f"Best model at epoch: {trainer.best_epoch}")
        print(f"Best {trainer.early_stop_metric}: {trainer.best_metric:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
