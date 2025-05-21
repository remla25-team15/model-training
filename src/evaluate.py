"""
Evaluation script for a trained sentiment classifier.

- Loads test data and a trained model.
- Computes accuracy and confusion matrix.
- Saves metrics as a JSON file.
"""

import argparse
import json

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def main():
    """
    Main function for model evaluation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--X_test", type=str, required=True)
    parser.add_argument("--y_test", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--metrics_output", type=str, required=True)
    args = parser.parse_args()

    X_test = np.load(args.X_test)
    y_test = np.load(args.y_test)
    model = joblib.load(args.model)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {"accuracy": acc, "confusion_matrix": cm.tolist()}

    with open(args.metrics_output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Evaluation complete. Accuracy:", acc)


if __name__ == "__main__":
    main()
