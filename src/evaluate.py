"""
Evaluation script for a trained sentiment classifier.

- Loads test data and a trained model.
- Computes accuracy and confusion matrix.
- Saves metrics as a JSON file.
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def load_data(X_path, y_path):
    """Load test features and labels."""
    X_test = np.load(X_path)
    y_test = np.load(y_path)
    return X_test, y_test


def load_model(model_path):
    """Load trained model from disk."""
    return joblib.load(model_path)


def evaluate_model(model, X_test, y_test):
    """Predict and compute evaluation metrics."""
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics


def save_metrics(metrics, output_path):
    """Save metrics dictionary as a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return output_path


def run_evaluation(X_path, y_path, model_path, metrics_output_path):
    X_test, y_test = load_data(X_path, y_path)
    model = load_model(model_path)
    metrics = evaluate_model(model, X_test, y_test)
    save_metrics(metrics, metrics_output_path)
    return metrics


def parse_args():
    # Avoid parsing args when run inside pytest
    if "PYTEST_CURRENT_TEST" in os.environ:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        return argparse.Namespace(
            X_test=os.path.join(base_dir, "data", "split", "X_test.npy"),
            y_test=os.path.join(base_dir, "data", "split", "y_test.npy"),
            model=os.path.join(base_dir, "output", "c2_Classifier_Sentiment_Model.pkl"),
            metrics_output=os.path.join(base_dir, "metrics", "feature_costs.json")
        )
    parser = argparse.ArgumentParser()
    parser.add_argument("--X_test", type=str, required=True)
    parser.add_argument("--y_test", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--metrics_output", type=str, required=True)

    return parser.parse_args()

def main():
    args = parse_args()
    metrics = run_evaluation(args.X_test, args.y_test, args.model, args.metrics_output)
    print("Evaluation complete. Accuracy:", metrics["accuracy"])


if __name__ == "__main__":
    main()