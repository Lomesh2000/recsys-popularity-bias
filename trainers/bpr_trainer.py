"""BPR and ILE-BPR trainer with enhanced early stopping."""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any
import numpy as np
from tqdm import tqdm

from models import BPRModel, ILEBPRModel
from losses import BPRLoss, ILELoss
from metrics import Evaluator
from utils.logging_utils import setup_logger


class EarlyStopping:
    """Early stopping handler with min_delta support.

    Monitors a metric and stops training when it stops improving.

    Args:
        patience: Number of epochs to wait before stopping
        metric: Metric name to monitor ('nDCG', 'UPD', 'AD', 'EE', 'loss')
        mode: 'min' to stop when metric stops decreasing, 'max' for increasing
        min_delta: Minimum change to qualify as an improvement
    """

    def __init__(
        self,
        patience: int = 20,
        metric: str = 'nDCG',
        mode: str = 'max',
        min_delta: float = 0.0
    ):
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta

        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.best_epoch = 0
        self.should_stop = False

        # Determine comparison operator
        if mode == 'min':
            self.is_better = lambda current, best: current < (best - min_delta)
        else:  # max
            self.is_better = lambda current, best: current > (best + min_delta)

    def step(self, metric_value: float, epoch: int) -> bool:
        """Check if metric improved.

        Args:
            metric_value: Current metric value
            epoch: Current epoch number

        Returns:
            True if this is the best so far, False otherwise
        """
        if self.is_better(metric_value, self.best_metric):
            self.best_metric = metric_value
            self.counter = 0
            self.best_epoch = epoch
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False

    def __repr__(self) -> str:
        return (f"EarlyStopping(patience={self.patience}, metric={self.metric}, "
                f"mode={self.mode}, min_delta={self.min_delta}, "
                f"best={self.best_metric:.4f}, counter={self.counter})")


class BPRTrainer:
    """Trainer for BPR and ILE-BPR models.

    Handles training loop, evaluation, checkpointing, logging, and early stopping.

    Args:
        model: PyTorch model (BPRModel or ILEBPRModel)
        dataset: Loaded dataset with train/test splits
        config: Configuration object
        device: torch.device
        logger: Optional logger instance
    """

    def __init__(
        self,
        model: nn.Module,
        dataset,
        config: Any,
        device: torch.device,
        logger=None
    ):
        self.model = model.to(device)
        self.dataset = dataset
        self.config = config
        self.device = device
        self.logger = logger or setup_logger('BPRTrainer')

        # Training config
        self.epochs = config.training.epochs
        self.batch_size = config.training.batch_size
        self.learning_rate = config.training.learning_rate
        self.weight_decay = config.training.get('weight_decay', 0.0)
        self.num_negatives = config.training.num_negatives

        # Loss config
        self.use_ile = config.loss.name == 'ile'
        self.distance = config.loss.get('distance', 'std')
        self.lambda_ile = config.loss.get('lambda_ile', 0.25)
        self.eps = config.loss.get('eps', 1e-8)

        # Setup loss
        if self.use_ile:
            self.criterion = ILELoss(
                distance=self.distance,
                lambda_ile=self.lambda_ile,
                eps=self.eps
            ).to(device)
        else:
            self.criterion = BPRLoss()

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Setup evaluator
        self.evaluator = Evaluator(
            dataset=dataset,
            k=config.evaluation.top_k,
            device=device
        )

        # Checkpointing
        self.checkpoint_dir = config.logging.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Early stopping
        self.early_stopping = None
        if config.training.early_stopping.enabled:
            self.early_stopping = EarlyStopping(
                patience=config.training.early_stopping.patience,
                metric=config.training.early_stopping.metric,
                mode=config.training.early_stopping.mode,
                min_delta=config.training.early_stopping.get('min_delta', 0.0)
            )
            self.early_stop_metric = config.training.early_stopping.metric
            self.early_stop_mode = config.training.early_stopping.mode

        self.best_epoch = 0
        self.best_metric = float('-inf') if (self.early_stopping and self.early_stop_mode == 'max') else float('inf')

        # History
        self.history = {
            'train_loss': [],
            'val_metrics': [],
            'epoch_times': []
        }

    def train(self) -> Dict[str, Any]:
        """Run full training loop.

        Returns:
            Training history dictionary with keys:
            - 'train_loss': list of average training losses per epoch
            - 'val_metrics': list of evaluation metric dicts
            - 'epoch_times': list of epoch durations
        """
        self.logger.info(f"Starting training for {self.epochs} epochs")
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        self.logger.info(f"Dataset: {self.dataset.dataset_name}")
        self.logger.info(f"Users: {self.dataset.n_users}, Items: {self.dataset.n_items}")
        self.logger.info(f"Train interactions: {len(self.dataset.train_interactions):,}")
        self.logger.info(f"Use ILE: {self.use_ile}")
        if self.use_ile:
            self.logger.info(f"Distance: {self.distance}, λ: {self.lambda_ile}")

        # Create dataloader
        from datasets.base import InteractionDataset

        train_dataset = InteractionDataset(
            interactions=self.dataset.train_interactions,
            n_items=self.dataset.n_items,
            n_negatives=self.num_negatives,
            seed=self.config.seed
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        self.logger.info(f"Batches per epoch: {len(train_loader)}")

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # Train one epoch
            train_loss, epoch_info = self._train_epoch(train_loader, epoch)
            epoch_time = time.time() - epoch_start
            self.history['epoch_times'].append(epoch_time)

            # Evaluate
            should_eval = (
                epoch % self.config.training.eval_freq == 0 or
                epoch == self.epochs or
                epoch == 1
            )

            if should_eval:
                metrics = self.evaluator.evaluate(self.model)

                # Log
                if self.use_ile and 'group_losses' in epoch_info:
                    gl = epoch_info['group_losses']
                    self.logger.info(
                        f"Epoch {epoch:3d}/{self.epochs} | "
                        f"Time: {epoch_time:5.2f}s | "
                        f"Loss: {train_loss:.4f} | "
                        f"BPR: {epoch_info['bpr_loss']:.4f} | "
                        f"ILE: {epoch_info['ile_term']:.4f} | "
                        f"Dist: {epoch_info['distance']:.4f} | "
                        f"L_H: {gl['H']:.4f} L_M: {gl['M']:.4f} L_T: {gl['T']:.4f} | "
                        f"nDCG: {metrics['nDCG']:.4f} | "
                        f"UPD: {metrics['UPD']:.4f} | "
                        f"AD: {metrics['AD']:.4f} | "
                        f"EE: {metrics['EE']:.4f}"
                    )
                else:
                    self.logger.info(
                        f"Epoch {epoch:3d}/{self.epochs} | "
                        f"Time: {epoch_time:5.2f}s | "
                        f"Loss: {train_loss:.4f} | "
                        f"nDCG: {metrics['nDCG']:.4f} | "
                        f"UPD: {metrics['UPD']:.4f} | "
                        f"AD: {metrics['AD']:.4f} | "
                        f"EE: {metrics['EE']:.4f}"
                    )

                # Save history
                self.history['train_loss'].append(train_loss)
                self.history['val_metrics'].append(metrics)

                # Save checkpoint periodically
                if epoch % self.config.training.save_freq == 0:
                    self._save_checkpoint(epoch, metrics)

                # Early stopping check
                if self.early_stopping is not None:
                    # Get metric value (use train_loss if metric is 'loss')
                    if self.early_stop_metric == 'loss':
                        current_metric = train_loss
                    else:
                        current_metric = metrics[self.early_stop_metric]

                    is_best = self.early_stopping.step(current_metric, epoch)

                    if is_best:
                        self.best_metric = current_metric
                        self.best_epoch = epoch
                        self._save_checkpoint(epoch, metrics, is_best=True)
                        self.logger.info(
                            f"  New best {self.early_stop_metric}: {current_metric:.4f} "
                            f"(saved as best_model.pt)"
                        )
                    else:
                        self.logger.info(
                            f"  {self.early_stop_metric}: {current_metric:.4f} "
                            f"(best: {self.best_metric:.4f} at epoch {self.best_epoch}, "
                            f"patience: {self.early_stopping.counter}/{self.early_stopping.patience})"
                        )

                    if self.early_stopping.should_stop:
                        self.logger.info(
                            f"Early stopping triggered at epoch {epoch}. "
                            f"Best {self.early_stop_metric}: {self.best_metric:.4f} "
                            f"at epoch {self.best_epoch}"
                        )
                        break
            else:
                self.history['train_loss'].append(train_loss)
                if self.use_ile and 'group_losses' in epoch_info:
                    gl = epoch_info['group_losses']
                    self.logger.info(
                        f"Epoch {epoch:3d}/{self.epochs} | "
                        f"Time: {epoch_time:5.2f}s | "
                        f"Loss: {train_loss:.4f} | "
                        f"BPR: {epoch_info['bpr_loss']:.4f} | "
                        f"ILE: {epoch_info['ile_term']:.4f} | "
                        f"L_H: {gl['H']:.4f} L_M: {gl['M']:.4f} L_T: {gl['T']:.4f}"
                    )
                else:
                    self.logger.info(
                        f"Epoch {epoch:3d}/{self.epochs} | "
                        f"Time: {epoch_time:5.2f}s | "
                        f"Loss: {train_loss:.4f}"
                    )

        # Training complete
        total_time = sum(self.history['epoch_times'])
        self.logger.info(f"Training completed in {total_time:.2f}s ({total_time/60:.2f}min)")

        if self.early_stopping:
            self.logger.info(
                f"Best model: epoch {self.best_epoch}, "
                f"{self.early_stop_metric}={self.best_metric:.4f}"
            )

        return self.history

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> tuple[float, dict]:
        """Train for one epoch.

        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number

        Returns:
            Tuple of (average training loss, loss_info dict with group losses)
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        # Track group losses across batches for epoch-level reporting
        epoch_group_losses = {'H': [], 'M': [], 'T': []}
        epoch_bpr_losses = []
        epoch_ile_terms = []
        epoch_distances = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

        for batch_idx, (users, pos_items, neg_items) in enumerate(pbar):
            users = users.to(self.device)
            pos_items = pos_items.to(self.device)
            neg_items = neg_items.to(self.device)

            # Forward pass
            user_embeds, pos_embeds, neg_embeds = self.model(users, pos_items, neg_items)

            # Compute loss
            if self.use_ile:
                # Get group masks for positive items
                group_masks = self.dataset.get_group_mask(pos_items.cpu().numpy())
                group_masks_tensor = {
                    k: torch.tensor(v, dtype=torch.bool, device=self.device)
                    for k, v in group_masks.items()
                }

                loss, loss_info = self.criterion(
                    user_embeds, pos_embeds, neg_embeds, group_masks_tensor
                )

                # Track group losses
                for g in ['H', 'M', 'T']:
                    if g in loss_info.get('group_losses', {}):
                        epoch_group_losses[g].append(loss_info['group_losses'][g])
                epoch_bpr_losses.append(loss_info.get('bpr_loss', 0))
                epoch_ile_terms.append(loss_info.get('ile_term', 0))
                epoch_distances.append(loss_info.get('distance', 0))
            else:
                bpr_losses = self.criterion(user_embeds, pos_embeds, neg_embeds)
                loss = torch.mean(bpr_losses)
                loss_info = {'bpr_loss': loss.item()}
                epoch_bpr_losses.append(loss.item())

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # Update progress bar
            if self.use_ile and 'ile_term' in loss_info:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'bpr': f"{loss_info['bpr_loss']:.4f}",
                    'ile': f"{loss_info['ile_term']:.4f}"
                })
            else:
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0

        # Compile epoch-level loss info
        epoch_info = {
            'bpr_loss': np.mean(epoch_bpr_losses) if epoch_bpr_losses else 0.0,
            'group_losses': {
                g: np.mean(v) if v else 0.0 
                for g, v in epoch_group_losses.items()
            },
            'distance': np.mean(epoch_distances) if epoch_distances else 0.0,
            'ile_term': np.mean(epoch_ile_terms) if epoch_ile_terms else 0.0,
        }

        return avg_loss, epoch_info

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch
            metrics: Current evaluation metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config.to_dict()
        }

        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        else:
            path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')

        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.info(f"Checkpoint loaded: {path}")
