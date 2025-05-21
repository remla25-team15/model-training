'''
   TESTS FOR MODEL DEVELOPMENT
'''


from sklearn.metrics import accuracy_score
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
        y_pred = trained_model.predict(X_slice)
        acc = accuracy_score(y_slice, y_pred)
        print(f"Negative review accuracy: {acc:.2f}")
        assert acc >= MIN_SLICE_ACCURACY, (
            f"Negative review accuracy ({acc:.2f}) below threshold\n"
            f"Slice size: {len(X_slice)} samples"
        )