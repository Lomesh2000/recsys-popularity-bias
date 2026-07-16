"""Unit tests for loss functions.

Run with: python tests/test_losses.py
"""

import torch
import numpy as np
from losses import BPRLoss, ILELoss, DistanceFunction


def test_bpr_loss():
    """Test BPR loss computation."""
    print("Testing BPRLoss...")

    criterion = BPRLoss()

    batch_size = 32
    embed_dim = 64

    user_embeds = torch.randn(batch_size, embed_dim)
    pos_embeds = torch.randn(batch_size, embed_dim)
    neg_embeds = torch.randn(batch_size, embed_dim)

    losses = criterion(user_embeds, pos_embeds, neg_embeds)

    assert losses.shape == (batch_size,), f"Expected (32,), got {losses.shape}"
    assert torch.all(losses > 0), "BPR losses should be positive"

    mean_loss = torch.mean(losses)
    assert mean_loss.dim() == 0, "Mean loss should be scalar"

    print(f"  Shape: {losses.shape}")
    print(f"  Range: [{losses.min():.4f}, {losses.max():.4f}]")
    print("  PASSED\n")


def test_distance_functions():
    """Test distance functions."""
    print("Testing DistanceFunction...")

    # Equal losses -> distance should be 0 (for STD and MAD)
    equal_losses = torch.tensor([1.0, 1.0, 1.0])

    std_equal = DistanceFunction.std(equal_losses)
    mad_equal = DistanceFunction.mad(equal_losses)
    ent_equal = DistanceFunction.entropy(equal_losses)

    assert std_equal.item() == 0.0, f"STD of equal values should be 0, got {std_equal.item()}"
    assert mad_equal.item() == 0.0, f"MAD of equal values should be 0, got {mad_equal.item()}"

    # Unequal losses -> distance should be > 0
    unequal_losses = torch.tensor([0.5, 1.0, 2.0])

    std_unequal = DistanceFunction.std(unequal_losses)
    mad_unequal = DistanceFunction.mad(unequal_losses)
    ent_unequal = DistanceFunction.entropy(unequal_losses)

    assert std_unequal.item() > 0, "STD of unequal values should be > 0"
    assert mad_unequal.item() > 0, "MAD of unequal values should be > 0"

    print(f"  STD(equal): {std_equal:.4f}, STD(unequal): {std_unequal:.4f}")
    print(f"  MAD(equal): {mad_equal:.4f}, MAD(unequal): {mad_unequal:.4f}")
    print(f"  ENT(equal): {ent_equal:.4f}, ENT(unequal): {ent_unequal:.4f}")
    print("  PASSED\n")


def test_ile_loss():
    """Test ILE loss computation."""
    print("Testing ILELoss...")

    batch_size = 32
    embed_dim = 64

    for distance in ['std', 'mad', 'ent']:
        criterion = ILELoss(distance=distance, lambda_ile=0.25)

        user_embeds = torch.randn(batch_size, embed_dim)
        pos_embeds = torch.randn(batch_size, embed_dim)
        neg_embeds = torch.randn(batch_size, embed_dim)

        # Create group masks (equal split)
        group_masks = {
            'H': torch.tensor([i < batch_size // 3 for i in range(batch_size)]),
            'M': torch.tensor([batch_size // 3 <= i < 2 * batch_size // 3 for i in range(batch_size)]),
            'T': torch.tensor([i >= 2 * batch_size // 3 for i in range(batch_size)])
        }

        loss, loss_info = criterion(user_embeds, pos_embeds, neg_embeds, group_masks)

        assert loss.dim() == 0, "Total loss should be scalar"
        assert 'bpr_loss' in loss_info
        assert 'distance' in loss_info
        assert 'ile_term' in loss_info
        assert 'group_losses' in loss_info

        # Verify ILE term = lambda * distance
        expected_ile = 0.25 * loss_info['distance']
        assert abs(loss_info['ile_term'] - expected_ile) < 1e-5, "ILE term mismatch"

        expected_total = loss_info['bpr_loss'] + loss_info['ile_term']
        assert abs(loss_info['total_loss'] - expected_total) < 1e-5, "Total loss mismatch"

        print(f"  {distance}: loss={loss.item():.4f}, bpr={loss_info['bpr_loss']:.4f}, "
              f"dist={loss_info['distance']:.4f}")

    print("  PASSED\n")


def test_gradient_flow():
    """Test that gradients flow correctly through ILE loss."""
    print("Testing gradient flow...")

    batch_size = 16
    embed_dim = 32
    n_users = 100
    n_items = 50

    from models import ILEBPRModel

    model = ILEBPRModel(n_users, n_items, embed_dim)
    criterion = ILELoss(distance='std', lambda_ile=0.5)

    users = torch.randint(0, n_users, (batch_size,))
    pos_items = torch.randint(0, n_items, (batch_size,))
    neg_items = torch.randint(0, n_items, (batch_size,))

    group_masks = {
        'H': torch.rand(batch_size) > 0.7,
        'M': torch.rand(batch_size) > 0.7,
        'T': torch.rand(batch_size) > 0.7
    }

    user_embeds, pos_embeds, neg_embeds = model(users, pos_items, neg_items)
    loss, _ = criterion(user_embeds, pos_embeds, neg_embeds, group_masks)
    loss.backward()

    assert model.user_embedding.weight.grad is not None, "User grad missing"
    assert model.item_embedding.weight.grad is not None, "Item grad missing"
    assert not torch.all(model.user_embedding.weight.grad == 0), "User grad is zero"
    assert not torch.all(model.item_embedding.weight.grad == 0), "Item grad is zero"

    print("  User embedding grad norm:", model.user_embedding.weight.grad.norm().item())
    print("  Item embedding grad norm:", model.item_embedding.weight.grad.norm().item())
    print("  PASSED\n")


if __name__ == '__main__':
    print("=" * 60)
    print("Running Loss Function Tests")
    print("=" * 60 + "\n")

    test_bpr_loss()
    test_distance_functions()
    test_ile_loss()
    test_gradient_flow()

    print("=" * 60)
    print("All loss tests passed!")
    print("=" * 60)
