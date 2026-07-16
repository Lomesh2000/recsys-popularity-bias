"""Unit tests for evaluation metrics.

Run with: python tests/test_metrics.py
"""

import numpy as np
from metrics import compute_ndcg, compute_ad, compute_ee, compute_upd


def test_ndcg():
    """Test nDCG computation."""
    print("Testing nDCG...")

    # Perfect ranking
    recommendations = {0: [1, 2, 3, 4, 5]}
    test_items = {0: [1, 2, 3]}
    ndcg = compute_ndcg(recommendations, test_items, k=5)
    assert ndcg > 0.9, f"Perfect ranking nDCG should be > 0.9, got {ndcg}"
    print(f"  Perfect ranking: {ndcg:.4f}")

    # Worst ranking
    recommendations = {0: [4, 5, 6, 7, 8]}
    ndcg = compute_ndcg(recommendations, test_items, k=5)
    assert ndcg == 0.0, f"No relevant items should give nDCG=0, got {ndcg}"
    print(f"  No relevant items: {ndcg:.4f}")

    # Partial match
    recommendations = {0: [1, 4, 5, 6, 7]}
    ndcg = compute_ndcg(recommendations, test_items, k=5)
    assert 0 < ndcg < 1, f"Partial match should give 0 < nDCG < 1, got {ndcg}"
    print(f"  Partial match: {ndcg:.4f}")
    print("  PASSED\n")


def test_ad():
    """Test Aggregate Diversity."""
    print("Testing AD...")

    n_items = 100

    # Low diversity
    recommendations = {u: [0, 1, 2, 3, 4] for u in range(10)}
    ad = compute_ad(recommendations, n_items, k=5)
    assert abs(ad - 0.05) < 1e-6, f"Expected 0.05, got {ad}"
    print(f"  Low diversity: {ad:.4f}")

    # High diversity
    recommendations = {u: list(range(u*5, u*5+5)) for u in range(10)}
    ad = compute_ad(recommendations, n_items, k=5)
    assert abs(ad - 0.5) < 1e-6, f"Expected 0.5, got {ad}"
    print(f"  High diversity: {ad:.4f}")

    # Full diversity
    recommendations = {u: [u] for u in range(100)}
    ad = compute_ad(recommendations, n_items, k=1)
    assert abs(ad - 1.0) < 1e-6, f"Expected 1.0, got {ad}"
    print(f"  Full diversity: {ad:.4f}")
    print("  PASSED\n")


def test_ee():
    """Test Equality of Exposure."""
    print("Testing EE...")

    n_items = 100

    # Equal exposure
    recommendations = {u: list(range(10)) for u in range(10)}
    ee = compute_ee(recommendations, n_items, k=10)
    assert ee > 0.8, f"Equal exposure should have high EE, got {ee}"
    print(f"  Equal exposure: {ee:.4f}")

    # Unequal exposure
    recommendations = {0: list(range(10))}
    for u in range(1, 10):
        recommendations[u] = [0] * 10
    ee = compute_ee(recommendations, n_items, k=10)
    assert ee < 0.5, f"Unequal exposure should have low EE, got {ee}"
    print(f"  Unequal exposure: {ee:.4f}")
    print("  PASSED\n")


def test_upd():
    """Test User Popularity Deviation."""
    print("Testing UPD...")

    # Perfect match
    train_items = {0: [1, 2, 3, 4, 5]}
    recommendations = {0: [1, 2, 3]}
    item_groups = {1: 'H', 2: 'H', 3: 'M', 4: 'M', 5: 'T'}

    upd = compute_upd(recommendations, train_items, item_groups, k=3)
    assert upd < 0.1, f"Perfect match should have low UPD, got {upd}"
    print(f"  Perfect match: {upd:.4f}")

    # Mismatch
    recommendations = {0: [5, 5, 5]}
    upd = compute_upd(recommendations, train_items, item_groups, k=3)
    assert upd > 0.1, f"Mismatch should have higher UPD, got {upd}"
    print(f"  Mismatch: {upd:.4f}")
    print("  PASSED\n")


if __name__ == '__main__':
    print("=" * 60)
    print("Running Metric Tests")
    print("=" * 60 + "\n")

    test_ndcg()
    test_ad()
    test_ee()
    test_upd()

    print("=" * 60)
    print("All metric tests passed!")
    print("=" * 60)
