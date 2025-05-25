'''
   TESTS FOR MODEL DEVELOPMENT
'''
import numpy as np
from scipy.stats import wilcoxon
from src.evaluate import evaluate_model
MIN_SLICE_ACCURACY = 0.80


def test_negative_keywords(trained_model, test_data):

    """
    Model 6: Model quality is sufficient on all important data slices.
    Test model on reviews containing selected negative keywords (slice)
    """

    X, y = test_data["X"], test_data["y"]

    # All reviews containing the word "bad"
    negative_reviews = [i for i, review in enumerate(X)
                        if "bad" in str(review) or "terrible" in str(review)]

    X_slice = X[negative_reviews]
    y_slice = y[negative_reviews]

    if len(X_slice) > 0:
        metrics = evaluate_model(trained_model, X_slice, y_slice)
        acc = metrics["accuracy"]
        print(f"Negative review accuracy: {acc:.2f}")
        assert acc >= MIN_SLICE_ACCURACY, (
            f"Negative review accuracy ({acc:.2f}) below threshold\n"
            f"Slice size: {len(X_slice)} samples"
        )


def test_robustness(trained_model, test_data, slice_size=100, repetitions=5, alpha=0.05):
    """
    Test model robustness by evaluating consistency of performance on random data slices.

    """
    X, y = test_data["X"], test_data["y"]
    n_samples = len(X)
    if slice_size > n_samples:
        raise ValueError(f"slice_size ({slice_size}) cannot be larger than test data size ({n_samples})")

    slice_accuracies = []

    for _ in range(repetitions):
        indices = np.random.choice(n_samples, size=slice_size, replace=False)
        X_slice = X[indices]
        y_slice = y[indices]

        metrics = evaluate_model(trained_model, X_slice, y_slice)
        acc = metrics["accuracy"]
        slice_accuracies.append(acc)

    differences = np.diff(slice_accuracies)
    if len(differences) == 0:
        is_robust = True
        p_value = None
    else:
        stat, p_value = wilcoxon(differences, alternative='two-sided')
        is_robust = p_value > alpha

    if is_robust:
        print(f"No statistically significant difference in performance across slices (p = {p_value:.4f})")
    else:
        print(f"Statistically significant difference detected in slice performance (p = {p_value:.4f})")

    assert is_robust, (
        f"Model is not robust: statistically significant difference in performance across slices (p = {p_value:.4f})"
    )

    return is_robust, slice_accuracies