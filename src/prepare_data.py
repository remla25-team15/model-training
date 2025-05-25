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
    if "PYTEST_CURRENT_TEST" in os.environ:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        return argparse.Namespace(
            dataset=os.path.join(base_dir, "datasets", "a1_RestaurantReviews_HistoricDump.tsv"),
            output_dir=os.path.join(base_dir, "data"),
            bow_dir=os.path.join(base_dir, "output")
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--bow_dir", type=str, required=True)
    return parser.parse_args()

def preprocess_and_save(dataset_path, output_dir, bow_dir):
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
    args = parse_args()
    preprocess_and_save(args.dataset, args.output_dir, args.bow_dir)


if __name__ == "__main__":
    main()