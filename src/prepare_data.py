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


def main():
    """
    Main function for parsing arguments and executing the preprocessing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--bow_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.bow_dir, exist_ok=True)

    messages = pd.read_csv(args.dataset, delimiter="\t", quoting=3)
    X, cv = libml._preprocess(messages)
    y = messages.iloc[:, -1].values

    np.save(os.path.join(args.output_dir, "X.npy"), X)
    np.save(os.path.join(args.output_dir, "y.npy"), y)
    with open(os.path.join(args.bow_dir, "c1_BoW_Sentiment_Model.pkl"), "wb") as f:
        pickle.dump(cv, f)


if __name__ == "__main__":
    main()
