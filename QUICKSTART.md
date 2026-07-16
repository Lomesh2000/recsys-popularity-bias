# QUICKSTART GUIDE - ILE-BPR Implementation

## Complete Running Sequence

### Step 0: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

---

### Step 1: Download Dataset

```bash
# Download MovieLens-1M (automatic)
python scripts/download_data.py --dataset movielens-1m

# This downloads to ./data/ml-1m/
```

---

### Step 2: Preprocess Dataset

```bash
# Preprocess MovieLens-1M
python scripts/preprocess.py --dataset movielens-1m

# This creates: ./data/movielens_1m_processed.pkl
```

---

### Step 3: Run Unit Tests (Optional but Recommended)

```bash
# Test all components
python tests/test_losses.py
python tests/test_models.py
python tests/test_metrics.py
```

---

### Step 4: Train Models

#### 4a. Train Base BPR (Baseline)
```bash
python train.py     --dataset movielens-1m     --distance none     --lambda_ile 0.0     --epochs 200     --name bpr_baseline
```

#### 4b. Train ILE-BPR with STD Distance
```bash
python train.py     --dataset movielens-1m     --distance std     --lambda_ile 0.25     --epochs 200     --name ile_std
```

#### 4c. Train ILE-BPR with ENT Distance
```bash
python train.py     --dataset movielens-1m     --distance ent     --lambda_ile 0.04     --epochs 200     --name ile_ent
```

#### 4d. Train ILE-BPR with MAD Distance
```bash
python train.py     --dataset movielens-1m     --distance mad     --lambda_ile 0.30     --epochs 200     --name ile_mad
```

---

### Step 5: Evaluate Trained Model

```bash
# Evaluate the best checkpoint
python evaluate.py     --checkpoint checkpoints/ile_std/best_model.pt
```

---

### Step 6: Generate Recommendations

```bash
# Get top-10 recommendations for user 42
python inference.py     --checkpoint checkpoints/ile_std/best_model.pt     --user_id 42     --top_k 10
```

---

## Expected Output

### Training Output
```
================================================================================
ILE-BPR Training Started
================================================================================
Using device: cuda
Dataset loaded: 6040 users, 3706 items
Train: 800,167, Test: 200,042
Model: ILEBPRModel
Parameters: 1,249,408
Starting training...
Epoch 10/200 | Time: 45.23s | Loss: 0.3421 | nDCG: 0.2156 | UPD: 0.0892 | AD: 0.4234 | EE: 0.0987
...
Epoch 200/200 | Time: 44.89s | Loss: 0.1234 | nDCG: 0.2263 | UPD: 0.0674 | AD: 0.4884 | EE: 0.1300
================================================================================
Training Complete!
================================================================================
Final Evaluation Metrics:
  nDCG      : 0.2263
  UPD       : 0.0674
  AD        : 0.4884
  EE        : 0.1300
```

### Evaluation Output
```
============================================================
Evaluation Results
============================================================
Metric                Value
------------------------------------------------------------
nDCG                  0.2263  (higher is better)
UPD                   0.0674  (lower is better)
AD                    0.4884  (higher is better)
EE                    0.1300  (higher is better)
============================================================
```

---

## Full Benchmark (Reproduce Paper Results)

Run all methods on MovieLens-1M:

```bash
#!/bin/bash
# benchmark.sh

DATASET="movielens-1m"
EPOCHS=200

# Base BPR
python train.py --dataset $DATASET --distance none --lambda_ile 0.0 --epochs $EPOCHS --name bpr

# ILE-STD
python train.py --dataset $DATASET --distance std --lambda_ile 0.25 --epochs $EPOCHS --name ile_std

# ILE-ENT
python train.py --dataset $DATASET --distance ent --lambda_ile 0.04 --epochs $EPOCHS --name ile_ent

# ILE-MAD
python train.py --dataset $DATASET --distance mad --lambda_ile 0.30 --epochs $EPOCHS --name ile_mad

# Evaluate all
echo "=== BPR ==="
python evaluate.py --checkpoint checkpoints/bpr/best_model.pt

echo "=== ILE-STD ==="
python evaluate.py --checkpoint checkpoints/ile_std/best_model.pt

echo "=== ILE-ENT ==="
python evaluate.py --checkpoint checkpoints/ile_ent/best_model.pt

echo "=== ILE-MAD ==="
python evaluate.py --checkpoint checkpoints/ile_mad/best_model.pt
```

---

## Hyperparameters from Paper

| Dataset | Method | λ | Distance | Epochs |
|---------|--------|---|----------|--------|
| MovieLens-1M | ILE-STD | 0.25 | std | 200 |
| MovieLens-1M | ILE-ENT | 0.04 | ent | 200 |
| MovieLens-1M | ILE-MAD | 0.30 | mad | 200 |
| Goodreads | ILE-STD | 0.30 | std | 175 |
| Goodreads | ILE-ENT | 0.02 | ent | 175 |
| Goodreads | ILE-MAD | 0.40 | mad | 175 |
| Google Reviews | ILE-STD | 0.25 | std | 175 |
| Google Reviews | ILE-ENT | 0.03 | ent | 175 |
| Google Reviews | ILE-MAD | 0.50 | mad | 175 |

---

## File Structure After Running

```
ile-bpr/
├── data/
│   ├── ml-1m/                    # Downloaded raw data
│   └── movielens_1m_processed.pkl # Preprocessed data
├── checkpoints/
│   ├── bpr/
│   │   ├── best_model.pt         # Best model checkpoint
│   │   └── checkpoint_epoch_10.pt  # Periodic checkpoints
│   ├── ile_std/
│   │   └── best_model.pt
│   ├── ile_ent/
│   │   └── best_model.pt
│   └── ile_mad/
│       └── best_model.pt
├── logs/
│   ├── bpr.log
│   ├── ile_std.log
│   └── ...
└── [source files...]
```

---

## Troubleshooting

### "Dataset not found"
Run `python scripts/download_data.py --dataset movielens-1m` first.

### "Processed dataset not found"
Run `python scripts/preprocess.py --dataset movielens-1m` after downloading.

### CUDA out of memory
Reduce batch size: `--batch_size 512` or `--batch_size 256`

### Slow training
- Use GPU: `--device cuda`
- Reduce evaluation frequency in config
- Use smaller dataset: `--dataset movielens-100k`
