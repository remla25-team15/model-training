"""
Data preparation script for sentiment analysis pipeline.

- Loads the TSV dataset.
- Applies text preprocessing using `libml._preprocess`.
- Saves the resulting features (X), labels (y), and the vectorizer (cv).

Expected to be used as the first stage in a DVC pipeline.
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from libml import preprocessing as libml


def parse_args():
    """
    Parses command-line arguments or sets defaults during pytest runs.

    Returns:
        argparse.Namespace: Parsed arguments with attributes:
            - dataset (str): Path to the input dataset TSV file.
            - output_dir (str): Directory where processed numpy arrays will be saved.
            - bow_dir (str): Directory where the vectorizer pickle will be saved.
    """
    if "PYTEST_CURRENT_TEST" in os.environ:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        return argparse.Namespace(
            dataset=os.path.join(
                base_dir, "datasets", "a1_RestaurantReviews_HistoricDump.tsv"
            ),
            output_dir=os.path.join(base_dir, "data"),
            bow_dir=os.path.join(base_dir, "output"),
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--bow_dir", type=str, required=True)
    return parser.parse_args()


def preprocess_and_save(dataset_path, output_dir, bow_dir):
    """
    Loads dataset, applies preprocessing, and saves features, labels, and vectorizer.

    Args:
        dataset_path (str): Path to the input TSV dataset file.
        output_dir (str): Directory where numpy arrays (X.npy, y.npy) will be saved.
        bow_dir (str): Directory where the vectorizer pickle file will be saved.

    Returns:
        tuple:
            - X (np.ndarray): Preprocessed feature matrix.
            - y (np.ndarray): Label array.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(bow_dir, exist_ok=True)

    messages = pd.read_csv(dataset_path, delimiter="\t", quoting=3)
    X, cv = libml._preprocess(messages)  # pylint: disable=protected-access
    y = messages.iloc[:, -1].values

    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)

    with open(os.path.join(bow_dir, "c1_BoW_Sentiment_Model.pkl"), "wb") as f:
        pickle.dump(cv, f)
    return X, y


def main():
    """
    Main function to parse arguments and run the preprocessing pipeline.
    """
    args = parse_args()
    preprocess_and_save(args.dataset, args.output_dir, args.bow_dir)


if __name__ == "__main__":
    main()
