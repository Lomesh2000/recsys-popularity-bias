# Implementation Assumptions

This document describes all assumptions made during the implementation of the ILE-BPR paper.

## Paper Reference
- **Title**: Correcting Popularity Bias in Recommender Systems via Item Loss Equalization
- **Authors**: Juno Prent, Masoud Mansoury
- **Venue**: ACM RecSys 2024 Workshop on Recommender Systems for Sustainability and Social Good

## Assumptions and Inferences

### 1. BPR Base Model Architecture
- **Assumption**: Standard matrix factorization BPR with user and item embeddings.
- **Justification**: The paper cites Rendle et al. [25] for BPR, which is the canonical matrix factorization formulation.
- **Details**: 
  - Embedding dimension: 128 (from paper's experimental setup)
  - No bias terms (not mentioned in paper, standard BPR doesn't use them)
  - Dot product for scoring: ŷ_ui = p_u^T q_i

### 2. Negative Sampling Strategy
- **Assumption**: Uniform random negative sampling, 1 negative per positive interaction.
- **Justification**: This is the standard BPR implementation. The paper does not specify the negative sampling ratio.
- **Details**: Negatives are sampled from items not interacted with by the user.

### 3. Item Grouping
- **Assumption**: Items grouped by total interaction count: top 20% = Head, bottom 20% = Tail, middle 60% = Mid.
- **Justification**: Section 3 explicitly states: "Head items (H) are the 20% of the most popular items, Tail items (T) are the 20% of the least popular items, and Mid items (M) are the rest."
- **Details**: Grouping is computed once from the training set before training begins.

### 4. Group Loss Computation (L_g)
- **Assumption**: L_g is the average BPR loss for all triplets in the current batch where the positive item i belongs to group g.
- **Justification**: The paper states ILE "minimizes the disparity of loss values across different item groups during the training process." Computing per-batch is standard for mini-batch SGD.
- **Alternative considered**: Per-epoch computation would require a full forward pass and is computationally expensive.

### 5. Distance Function D Implementation
- **STD**: Standard deviation of group losses. Formula: sqrt(mean((L_g - mean(L_g))²))
- **MAD**: Mean absolute deviation. Formula: mean(|L_g - mean(L_g)|)
- **ENT**: Entropy of normalized group losses. Formula: -Σ p_g * log(p_g) where p_g = L_g / sum(L_g)
- **Justification**: Table 1 in the paper provides these formulas. For ENT, normalization is necessary since losses don't naturally sum to 1.

### 6. Loss Aggregation
- **Assumption**: The final loss is L* = L + λ * D, where L is the mean BPR loss over the batch.
- **Justification**: Equation 2 in the paper: L* = L + λ × D({L_g | g ∈ G})

### 7. Optimizer and Hyperparameters
- **Assumption**: Adam optimizer with learning rate 1e-4.
- **Justification**: Section 4.4 states: "learning rate 10^-4 and embedding size 128."
- **Inferred**: Adam is standard; paper doesn't specify optimizer but RecBole defaults to Adam.

### 8. Train/Test Split
- **Assumption**: Random 80/20 per-user split (leave-some-out).
- **Justification**: Section 4.4: "randomly divide users' profiles into training and test sets, with 80% allocated to the training set and 20% to the test set."

### 9. Evaluation Protocol
- **Assumption**: For each user, rank all non-training items. Compute metrics@10.
- **Justification**: Standard recommendation evaluation. Paper mentions "recommendation list of size 10."

### 10. Dataset Preprocessing
- **Assumption**: Implicit feedback only (binary interactions). No rating thresholding for MovieLens.
- **Justification**: Paper uses implicit feedback scenarios (Section 3).

### 11. Running Statistics for Group Losses
- **Assumption**: We maintain running averages of group losses for stability when a group has few samples in a batch.
- **Justification**: In early training or with small batches, some groups may have 0-1 samples, making per-batch L_g noisy.

### 12. Time Complexity
- **Assumption**: O(batch_size) overhead for ILE per batch.
- **Justification**: Computing group averages and distance is linear in batch size.

## Deviations from Paper

1. **Evaluation metrics implementation**: The paper uses RecBole for experiments. We implement metrics from scratch following standard definitions, which may have minor differences in edge case handling.

2. **Google Reviews dataset**: The paper's URL for this dataset may be outdated. We use the standard version available.

3. **Goodreads dataset**: The paper references a GitHub repository. We assume standard Goodreads book interactions format.
