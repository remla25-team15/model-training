from src.evaluate import evaluate_model

# Choose an appropriate threshold for the model
MIN_ACCURACY = 0.6


# Maybe change test_data to validation_data
def test_model_serving_validation(trained_model, test_data):
    """
    INFRA 4: Validate model quality before serving
    Compares against both absolute and relative thresholds
    """
    X, y = test_data["X"], test_data["y"]
    metrics = evaluate_model(trained_model, X, y)
    val_accuracy = metrics["accuracy"]

    # Absolute quality check
    assert val_accuracy >= MIN_ACCURACY, (
        f"Model accuracy {val_accuracy:.2f} below serving threshold {MIN_ACCURACY}\n"
        f"Failing samples:\n{X[trained_model.predict(X) != y][:3]}"
    )

    # Relative check vs previous version
    PREV_MODEL_ACCURACY = 0.70  # TODO: Replace with actual previous model accuracy
    assert (
        val_accuracy >= PREV_MODEL_ACCURACY * 0.95
    ), (  # Allow 5% degradation (choose an appropriate threshold)
        f"Model accuracy dropped >5% from baseline {PREV_MODEL_ACCURACY:.2f}"
    )
