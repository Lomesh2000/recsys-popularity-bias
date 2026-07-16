"""Unit tests for models.

Run with: python tests/test_models.py
"""

import torch
from models import BPRModel, ILEBPRModel


def test_bpr_model():
    """Test BPR model forward pass."""
    print("Testing BPRModel...")

    n_users = 100
    n_items = 50
    embed_dim = 32
    batch_size = 16

    model = BPRModel(n_users, n_items, embed_dim)

    users = torch.randint(0, n_users, (batch_size,))
    pos_items = torch.randint(0, n_items, (batch_size,))
    neg_items = torch.randint(0, n_items, (batch_size,))

    user_embeds, pos_embeds, neg_embeds = model(users, pos_items, neg_items)

    assert user_embeds.shape == (batch_size, embed_dim)
    assert pos_embeds.shape == (batch_size, embed_dim)
    assert neg_embeds.shape == (batch_size, embed_dim)

    print(f"  Output shapes: {user_embeds.shape}")
    print("  PASSED\n")


def test_ile_bpr_model():
    """Test ILE-BPR model."""
    print("Testing ILEBPRModel...")

    n_users = 100
    n_items = 50
    embed_dim = 32
    batch_size = 16

    model = ILEBPRModel(n_users, n_items, embed_dim)

    users = torch.randint(0, n_users, (batch_size,))
    pos_items = torch.randint(0, n_items, (batch_size,))
    neg_items = torch.randint(0, n_items, (batch_size,))

    user_embeds, pos_embeds, neg_embeds = model(users, pos_items, neg_items)

    assert user_embeds.shape == (batch_size, embed_dim)

    print(f"  Output shapes: {user_embeds.shape}")
    print("  PASSED\n")


def test_predict():
    """Test prediction method."""
    print("Testing predict method...")

    n_users = 10
    n_items = 20
    embed_dim = 16

    model = BPRModel(n_users, n_items, embed_dim)

    n_pairs = 50
    users = torch.randint(0, n_users, (n_pairs,))
    items = torch.randint(0, n_items, (n_pairs,))

    scores = model.predict(users, items)

    assert scores.shape == (n_pairs,)
    assert scores.dtype == torch.float32

    print(f"  Score shape: {scores.shape}")
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print("  PASSED\n")


def test_model_parameters():
    """Test model parameter counts."""
    print("Testing model parameters...")

    n_users = 1000
    n_items = 500
    embed_dim = 128

    model = BPRModel(n_users, n_items, embed_dim)
    n_params = sum(p.numel() for p in model.parameters())

    expected = n_users * embed_dim + n_items * embed_dim
    assert n_params == expected, f"Expected {expected} params, got {n_params}"

    print(f"  Total parameters: {n_params:,}")
    print(f"  Expected: {expected:,}")
    print("  PASSED\n")


if __name__ == '__main__':
    print("=" * 60)
    print("Running Model Tests")
    print("=" * 60 + "\n")

    test_bpr_model()
    test_ile_bpr_model()
    test_predict()
    test_model_parameters()

    print("=" * 60)
    print("All model tests passed!")
    print("=" * 60)
