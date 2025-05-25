'''
Tests for features and data.
'''
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import yaml

from src.evaluate import parse_args

MIN_ROWS = 100
MAX_TEXT_LENGTH = 1000

def test_dataset_exists(real_data):
    """Verify dataset loads and has basic validity"""
    assert len(real_data) >= MIN_ROWS, f"Dataset has too few rows ({len(real_data)})"
    assert len(real_data.columns) > 1, "Dataset has too many columns"
    assert "Review" in real_data.columns, "Missing 'Review' column"
    assert "Liked" in real_data.columns, "Missing 'Liked' column"

# Do we need to check for duplicates? The dataset contrains some duplicates.
# def test_no_duplicates(real_data):
#     """No duplicate reviews"""
#     duplicates = real_data.duplicated(subset=["Review"], keep=False)
#     assert not duplicates.any(), f"Duplicate reviews found: {real_data[duplicates]['Review'].head().tolist()}"


def test_feature_schema(real_data):
    """
    Data 1: Feature expectations are captured in a schema.
    Validate schema of real data
    """
    schema = {
        "Review": {
            "type": str,
            "null_allowed": False,
            "max_length": MAX_TEXT_LENGTH
        },
        "Liked": {
            "type": int,
            "allowed_values": [0, 1]
        }
    }

    for col, rules in schema.items():
        assert all(isinstance(x, rules["type"]) for x in real_data[col] if pd.notnull(x)), f"Invalid type in {col}"

        if not rules.get("null_allowed", True):
            assert not real_data[col].isnull().any(), f"Null values found in {col}"

        if "allowed_values" in rules:
            assert all(x in rules["allowed_values"] for x in real_data[col] if pd.notnull(x)), f"Invalid values found in {col}"

        if "max_length" in rules:
            lengths = real_data[col].str.len()
            assert (lengths <= rules["max_length"]).all(), f"Value too long in {col}"



def test_feature_cost():
    """
    Loads data and params, then tests the impact of removing each feature on accuracy.
    Saves the top 10 most impactful features (by absolute accuracy change) to a JSON file.

    Returns:
        dict: Feature index -> accuracy difference (top 10 only).
    """
    args = parse_args()

    # Load data
    X = np.load(args.X_test)
    y = np.load(args.y_test)

    # Infer base directory from X_test path
    base_dir = os.path.abspath(os.path.join(os.path.dirname(args.X_test), ".."))
    params_path = os.path.join(base_dir, "params.yaml")

    # Load params
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    train_params = params.get("train", {})
    test_size = train_params.get("test_size", 0.2)
    random_state = train_params.get("random_state", 20)
    var_smoothing = float(train_params.get("var_smoothing", 1e-9))
    priors = params.get("priors", None)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train baseline model
    model = GaussianNB(var_smoothing=var_smoothing, priors=priors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    baseline_acc = accuracy_score(y_test, y_pred)

    feature_costs = {}
    n_features = X.shape[1]

    for i in range(n_features):
        # Remove feature i
        X_removed = np.delete(X, i, axis=1)

        # Split modified data
        X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
            X_removed, y, test_size=test_size, random_state=random_state
        )

        model_i = GaussianNB(var_smoothing=var_smoothing, priors=priors)
        model_i.fit(X_train_i, y_train_i)
        y_pred_i = model_i.predict(X_test_i)
        acc_i = accuracy_score(y_test_i, y_pred_i)

        feature_costs[i] = baseline_acc - acc_i

    # Select top 10 most impactful features (by abs accuracy difference)
    top_feature_costs = dict(
        sorted(feature_costs.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.metrics_output), exist_ok=True)

    # Save results
    with open(args.metrics_output, "w", encoding="utf-8") as f:
        json.dump(top_feature_costs, f, indent=2)

    return top_feature_costs