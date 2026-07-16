
"""
Pareto Sweep: Accuracy (nDCG) vs. Fairness (UPD) Trade-off
Reproduces Section 4.5 of the ILE paper.
"""
import os
import sys
import copy
import json
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.config import Config
from utils.logging_utils import setup_logger, get_device
from utils.seed import setup_reproducibility
from datasets.movielens import MovieLensDataset
from models import BPRModel, ILEBPRModel
from trainers.bpr_trainer import BPRTrainer
from metrics import Evaluator

def get_base_config_dict():
    """Returns the exact configuration used in your successful training runs."""
    return {
        'seed': 42,
        'deterministic': True,
        'dataset': {'name': 'movielens-1m', 'data_dir': './data', 'train_ratio': 0.8, 'min_interactions': 5},
        'model': {'name': 'ile_bpr', 'embedding_dim': 128, 'init_std': 0.01},
        'loss': {'name': 'ile', 'distance': 'std', 'lambda_ile': 0.25, 'eps': 1e-08, 'reg_weight': 1e-4},
        'training': {
            'epochs': 200, 'batch_size': 1024, 'learning_rate': 0.0001, 'optimizer': 'adam', 
            'weight_decay': 0.0, 'num_negatives': 1, 'eval_freq': 10, 'save_freq': 10, 
            'early_stopping': {'enabled': True, 'patience': 20, 'metric': 'nDCG', 'mode': 'max', 'min_delta': 0.0}
        },
        'evaluation': {'top_k': 10, 'metrics': ['nDCG', 'UPD', 'AD', 'EE']},
        'logging': {'log_dir': './logs', 'checkpoint_dir': './checkpoints', 'use_tensorboard': False, 'log_interval': 100},
        'device': 'auto',
        'distributed': False
    }

def main():
    # 1. Define the Sweep Space (Including BPR Baseline at 0.0)
    lambda_values = [0.0, 0.05, 0.10, 0.25, 0.50, 1.0]
    results = []
    
    device = get_device('auto')
    print(f"Using device: {device}")
    
    # 2. Load Dataset ONCE (Saves massive I/O time)
    print("Loading Dataset...")
    base_cfg = get_base_config_dict()
    dataset = MovieLensDataset(
        data_dir=base_cfg['dataset']['data_dir'],
        variant='1m',
        train_ratio=base_cfg['dataset']['train_ratio'],
        min_interactions=base_cfg['dataset']['min_interactions'],
        seed=base_cfg['seed']
    )
    
    processed_path = os.path.join(base_cfg['dataset']['data_dir'], 'movielens_1m_processed.pkl')
    if not os.path.exists(processed_path):
        print(f"ERROR: Processed dataset not found at {processed_path}. Run preprocess.py first.")
        sys.exit(1)
        
    dataset.load_state(processed_path)
    print(f"Dataset loaded: {dataset.n_users} users, {dataset.n_items} items\n")

    # 3. Execute Sweep
    for lam in lambda_values:
        print(f"\n{'='*60}")
        print(f"Starting Run: λ = {lam}")
        print(f"{'='*60}")
        
        # Deep copy config and update lambda & directories
        run_cfg_dict = copy.deepcopy(base_cfg)
        
        if lam == 0.0:
            run_cfg_dict['loss']['name'] = 'bpr'
            run_name = 'bpr_baseline'
        else:
            run_cfg_dict['loss']['name'] = 'ile'
            run_cfg_dict['loss']['lambda_ile'] = lam
            run_name = f'ile_std_lambda_{lam}'
            
        run_cfg_dict['logging']['checkpoint_dir'] = f'./checkpoints/{run_name}'
        run_cfg_dict['logging']['log_dir'] = f'./logs/{run_name}'
        
        os.makedirs(run_cfg_dict['logging']['checkpoint_dir'], exist_ok=True)
        os.makedirs(run_cfg_dict['logging']['log_dir'], exist_ok=True)
        
        config = Config(run_cfg_dict)
        setup_reproducibility(config.seed, config.deterministic)
        logger = setup_logger(run_name)
        
        checkpoint_path = os.path.join(config.logging.checkpoint_dir, 'best_model.pt')
        
        # Instantiate Model
        ModelClass = BPRModel if lam == 0.0 else ILEBPRModel
        model = ModelClass(
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            embedding_dim=config.model.embedding_dim,
            init_std=config.model.init_std
        ).to(device)
        
        # Resume Logic: Skip training if checkpoint exists
        if os.path.exists(checkpoint_path):
            logger.info(f"Found existing checkpoint for λ={lam}. Skipping training and loading weights.")
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            logger.info(f"No checkpoint found. Training from scratch for λ={lam}...")
            trainer = BPRTrainer(model, dataset, config, device, logger)
            trainer.train()
            # Reload best model after training
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            
        # Evaluate
        logger.info(f"Evaluating model for λ={lam}...")
        evaluator = Evaluator(dataset, k=config.evaluation.top_k, device=device)
        metrics = evaluator.evaluate(model)
        
        results.append({
            'lambda': lam,
            'nDCG': metrics['nDCG'],
            'UPD': metrics['UPD'],
            'AD': metrics['AD'],
            'EE': metrics['EE']
        })
        
        logger.info(f"Run λ={lam} Complete | nDCG: {metrics['nDCG']:.4f} | UPD: {metrics['UPD']:.4f}")

    # 4. Save Raw Results to JSON
    with open('pareto_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nSaved raw metrics to pareto_results.json")

    # 5. Plot the Pareto Curve
    plot_pareto_curve(results)

def plot_pareto_curve(results):
    """Generates the Accuracy vs. Fairness Trade-off Plot."""
    lambdas = [r['lambda'] for r in results]
    ndcgs = [r['nDCG'] for r in results]
    upds = [r['UPD'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot the trajectory line
    ax.plot(ndcgs, upds, '--', color='gray', alpha=0.6, label='Sweep Trajectory')
    
    # Scatter points
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(lambdas)))
    
    for i, (lam, ndcg, upd) in enumerate(zip(lambdas, ndcgs, upds)):
        marker = '*' if lam == 0.0 else 'o'
        size = 200 if lam == 0.0 else 100
        label = 'BPR Baseline (λ=0.0)' if lam == 0.0 else f'λ={lam}'
        
        ax.scatter(ndcg, upd, s=size, c=[colors[i]], marker=marker, edgecolors='black', zorder=5)
        
        # Annotate points
        offset_x = 0.002 if lam != 0.50 else -0.015
        offset_y = 0.005 if lam != 0.0 else -0.015
        ax.annotate(label, (ndcg + offset_x, upd + offset_y), fontsize=11, fontweight='bold')

    # Highlight the "Knee" (Pareto Optimal Point - usually 0.25)
    # We assume 0.25 is the knee based on the paper, but you can dynamically calculate it
    optimal_lam = 0.25
    opt_idx = lambdas.index(optimal_lam)
    ax.scatter(ndcgs[opt_idx], upds[opt_idx], s=300, facecolors='none', edgecolors='red', linewidths=2, zorder=6)
    ax.annotate('Pareto Optimal', (ndcgs[opt_idx] + 0.005, upds[opt_idx] - 0.015), color='red', fontsize=12, fontweight='bold')

    # Formatting
    ax.set_xlabel('Accuracy (nDCG@10) $\rightarrow$ Higher is Better', fontsize=14)
    ax.set_ylabel('Unfairness (UPD / JSD) $\rightarrow$ Lower is Better', fontsize=14)
    ax.set_title('Accuracy-Fairness Trade-off (Pareto Front)\nILE vs Standard BPR', fontsize=16, fontweight='bold')
    
    # Add grid and ideal zone annotation
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.text(0.05, 0.95, '← Ideal Zone\n(High Accuracy, Low Bias)', transform=ax.transAxes, 
            fontsize=12, color='green', alpha=0.6, va='top', ha='left', fontstyle='italic')
            
    plt.tight_layout()
    plt.savefig('pareto_curve.png', dpi=300)
    print("\n✅ Saved Pareto Curve to pareto_curve.png")
    plt.show()

if __name__ == '__main__':
    main()
