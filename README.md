# Item Loss Equalization (ILE) for BPR

Production-quality implementation of **"Correcting Popularity Bias in Recommender Systems via Item Loss Equalization"** by Juno Prent and Masoud Mansoury (ACM RecSys 2024 Workshop).

## Overview

This repository implements Item Loss Equalization (ILE), an in-processing approach that corrects popularity bias in recommender systems by equalizing training loss across item popularity groups (Head, Mid, Tail).

### Key Features
- **BPR Base Model**: Bayesian Personalized Ranking with matrix factorization
- **ILE Loss**: Three distance functions (STD, ENT, MAD) for loss equalization
- **Fairness Metrics**: UPD, AD, EE as described in the paper
- **Datasets**: MovieLens-1M, Goodreads, Google Reviews
- **Reproducibility**: Full seed control, deterministic execution, checkpointing

## Installation

```bash
# Clone repository
git clone <repo-url>
cd ile-bpr

# Create environment
conda env create -f environment.yml
conda activate ile-bpr

# Or use pip
pip install -r requirements.txt
```

## Dataset Preparation

```bash
# Download and preprocess all datasets
python scripts/download_data.py --dataset all
python scripts/preprocess.py --dataset all

# Or individually
python scripts/download_data.py --dataset movielens-1m
python scripts/preprocess.py --dataset movielens-1m
```

## Training

```bash
# Train ILE-BPR with STD distance on MovieLens-1M
python train.py --config configs/default.yaml     --dataset movielens-1m     --distance std     --lambda_ile 0.25

# Train base BPR (no ILE)
python train.py --config configs/default.yaml     --dataset movielens-1m     --distance none     --lambda_ile 0.0

# Train with different distance functions
python train.py --dataset movielens-1m --distance ent --lambda_ile 0.04
python train.py --dataset movielens-1m --distance mad --lambda_ile 0.3
```

## Evaluation

```bash
# Evaluate a trained model
python evaluate.py --checkpoint checkpoints/movielens-1m_std_0.25/best_model.pt

# Run full benchmark across all methods
python scripts/benchmark.py --dataset movielens-1m
```

## Inference

```bash
# Generate recommendations for a user
python inference.py --checkpoint checkpoints/movielens-1m_std_0.25/best_model.pt     --user_id 42     --top_k 10
```

## Expected Results (MovieLens-1M)

| Method | nDCG↑ | UPD↓ | AD↑ | EE↑ |
|--------|-------|------|-----|-----|
| BPR | ~0.269 | ~0.121 | ~0.423 | ~0.096 |
| ILE (STD, λ=0.25) | ~0.226 | ~0.067 | ~0.488 | ~0.130 |
| ILE (ENT, λ=0.04) | ~0.224 | ~0.070 | ~0.525 | ~0.151 |
| ILE (MAD, λ=0.30) | ~0.241 | ~0.093 | ~0.444 | ~0.103 |

## Repository Structure

```
ile-bpr/
├── configs/          # Configuration files
├── data/             # Downloaded datasets
├── datasets/         # Dataset loaders and preprocessors
├── models/           # BPR and ILE-BPR models
├── losses/           # BPR loss and ILE distance functions
├── metrics/          # Evaluation metrics (nDCG, UPD, AD, EE)
├── trainers/         # Training loops
├── evaluation/       # Evaluation engine
├── utils/            # Utilities (config, seed, logging)
├── scripts/          # Data download and benchmark scripts
├── tests/            # Unit tests
├── train.py          # Main training script
├── evaluate.py       # Evaluation script
└── inference.py      # Inference script
```

## Reproducibility

All experiments use fixed random seeds by default. Set `seed` in config or via CLI:

```bash
python train.py --seed 42 --deterministic
```

## Citation

```bibtex
@inproceedings{prent2024ile,
  title={Correcting Popularity Bias in Recommender Systems via Item Loss Equalization},
  author={Prent, Juno and Mansoury, Masoud},
  booktitle={ACM RecSys 2024 Workshop on Recommender Systems for Sustainability and Social Good},
  year={2024}
}
```

## License

MIT License
