"""
Training script for sentiment classifier using Gaussian Naive Bayes.

- Loads preprocessed data (X and y).
- Either trains on the full dataset or performs a train/test split.
- Saves the trained model and optionally the test set for evaluation.
"""

import argparse
import json
import os

import joblib
import numpy as np
import yaml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def parse_args():
    """
    Parse command-line arguments for the training script.

    Special handling is included to avoid parsing arguments when running inside pytest.

    Returns:
        argparse.Namespace: Parsed arguments with attributes:
            - data (str): Path to the input features (X) NumPy file.
            - labels (str): Path to the input labels (y) NumPy file.
            - output (str): Directory to save the trained model.
            - split_output_dir (str, optional): Directory to save test split data.
            - train_metrics_output (str, optional): File path to save training metrics JSON.
    """
    # Avoid parsing args when run inside pytest
    if "PYTEST_CURRENT_TEST" in os.environ:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        return argparse.Namespace(
            data=os.path.join(base_dir, "data", "X.npy"),
            labels=os.path.join(base_dir, "data", "y.npy"),
            output=os.path.join(base_dir, "models"),
            split_output_dir=os.path.join(base_dir, "data", "split"),
            train_metrics_output=os.path.join(
                base_dir, "metrics", "train_metrics.json"
            ),
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--split_output_dir", type=str)
    parser.add_argument("--train_metrics_output", type=str)
    return parser.parse_args()


def load_params(path="params.yaml"):
    """
    Load training configuration parameters from a YAML file.

    Extracts settings for training behavior, test split size, random seed, and model hyperparameters.

    Args:
        path (str, optional): Path to the YAML config file. Defaults to "params.yaml".

    Returns:
        dict: Configuration dictionary with keys:
            - train_all (bool): Whether to train on full dataset without splitting.
            - test_size (float): Proportion of data to reserve for testing.
            - random_state (int): Random seed for reproducibility.
            - priors (list or None): Prior probabilities for GaussianNB.
            - var_smoothing (float): Variance smoothing parameter for GaussianNB.
    """
    with open(path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    train = params.get("train", {})
    return {
        "train_all": train.get("train_all", False),
        "test_size": train.get("test_size", 0.2),
        "random_state": train.get("random_state", 20),
        "priors": params.get("priors", None),
        "var_smoothing": params.get("var_smoothing", 1e-9),
    }


def save_json(path, obj):
    """
    Save a Python object as a formatted JSON file.

    Creates the directory path if it does not exist.

    Args:
        path (str): File path where JSON data will be saved.
        obj (object): The Python object (e.g., dict) to serialize.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_split_data(output_dir, X_test, y_test):
    """
    Save test split feature and label arrays as NumPy files.

    Creates the output directory if it does not exist.

    Args:
        output_dir (str): Directory to save the test split data.
        X_test (np.ndarray): Test set features.
        y_test (np.ndarray): Test set labels.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)


def fit_naive_bayes(X_train, y_train, config):
    """
    Trains and returns a GaussianNB model without saving.
    Useful for programmatic use.
    """
    model = GaussianNB(var_smoothing=config["var_smoothing"], priors=config["priors"])
    model.fit(X_train, y_train)
    return model


def train_model(X, y, config, args):
    """
    Full pipeline with saving and splitting, used from CLI.
    """
    if config["train_all"]:
        model = fit_naive_bayes(X, y, config)
        if args.train_metrics_output:
            acc = accuracy_score(y, model.predict(X))
            save_json(args.train_metrics_output, {"train_accuracy": acc})
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config["test_size"], random_state=config["random_state"]
        )
        model = fit_naive_bayes(X_train, y_train, config)

        if args.train_metrics_output:
            acc = accuracy_score(y_train, model.predict(X_train))
            save_json(args.train_metrics_output, {"train_accuracy": acc})

        if args.split_output_dir:
            save_split_data(args.split_output_dir, X_test, y_test)

    os.makedirs(args.output, exist_ok=True)
    joblib.dump(model, os.path.join(args.output, "c2_Classifier_Sentiment_Model.pkl"))


def main():
    """
    Main entry point for the training script.

    Parses command-line arguments, loads configuration and data, then initiates the training process.
    """
    args = parse_args()
    config = load_params()
    X = np.load(args.data)
    y = np.load(args.labels)
    train_model(X, y, config, args)


if __name__ == "__main__":
    main()
