import numpy as np
MIN_SLICE_ACCURACY = 0.80


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
        assert (
            pred_orig == pred_swap
        ), f"Swapping features {i} and {j} in sample {idx} changed the predicted class"

def test_negative_keywords(trained_model, test_data):
    """
    Model 6: Model quality is sufficient on all important data slices.
    Test model on reviews containing selected negative keywords (slice)
    """

    X, y = test_data["X"], test_data["y"]

    # All reviews containing the word "bad"
    negative_reviews = [
        i
        for i, review in enumerate(X)
        if "bad" in str(review) or "terrible" in str(review)
    ]

    X_slice = X[negative_reviews]
    y_slice = y[negative_reviews]

    if len(X_slice) > 0:
        metrics = evaluate_model(trained_model, X_slice, y_slice)
        slice_accuracy = metrics["accuracy"]

        assert (
            slice_accuracy >= MIN_SLICE_ACCURACY
        ), f"Model accuracy on negative keyword slice is too low: {slice_accuracy:.2f} < {MIN_SLICE_ACCURACY}"
    else:
        # Skip test if no negative keywords found
        pytest.skip("No reviews with negative keywords found")
