"""
Training script for sentiment classifier using Gaussian Naive Bayes.

- Loads preprocessed data (X and y).
- Either trains on the full dataset or performs a train/test split.
- Saves the trained model and optionally the test set for evaluation.
"""

import argparse
import os

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def main():
    """
    Main function for training a Naive Bayes classifier.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--train_all", action="store_true")
    parser.add_argument(
        "--split_output_dir", type=str, help="Optional dir to save split test sets"
    )
    args = parser.parse_args()

    X = np.load(args.data)
    y = np.load(args.labels)

    os.makedirs(args.output, exist_ok=True)

    model = GaussianNB()

    if args.train_all:
        model.fit(X, y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=20
        )
        model.fit(X_train, y_train)
        if args.split_output_dir:
            os.makedirs(args.split_output_dir, exist_ok=True)
            np.save(os.path.join(args.split_output_dir, "X_test.npy"), X_test)
            np.save(os.path.join(args.split_output_dir, "y_test.npy"), y_test)

    joblib.dump(model, args.output + "c2_Classifier_Sentiment_Model.pkl")


if __name__ == "__main__":
    main()
