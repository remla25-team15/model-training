import numpy as np
import pytest


def test_feature_swap_mutamorphic_two_samples(trained_model, test_data):
    """
    Mutamorphic test: For two random test samples, swap two random features and check that the prediction does not change.
    """
    X = test_data["X"]
    n_samples, n_features = X.shape
    rng = np.random.default_rng(42)

    for _ in range(2):
        idx = rng.integers(0, n_samples)
        x_orig = X[idx].copy()
        i, j = rng.choice(n_features, size=2, replace=False)
        x_swapped = x_orig.copy()
        x_swapped[i], x_swapped[j] = x_swapped[j], x_swapped[i]
        pred_orig = trained_model.predict([x_orig])[0]
        pred_swap = trained_model.predict([x_swapped])[0]
        assert pred_orig == pred_swap, f"Swapping features {i} and {j} in sample {idx} changed the predicted class" 