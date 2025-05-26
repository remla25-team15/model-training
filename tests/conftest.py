import os

import numpy as np
import pytest
import pandas as pd
import joblib

# DATASET_PATH = "../datasets/a1_RestaurantReviews_HistoricDump.tsv"
# TEST_DATA_DIR = "data/processed/test"

DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "../datasets/a1_RestaurantReviews_HistoricDump.tsv"
)

FRESH_DUMP_PATH = os.path.join(
    os.path.dirname(__file__), "../datasets/a2_RestaurantReviews_FreshDump.tsv"
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "../output/c2_Classifier_Sentiment_Model.pkl"
)
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/split")


@pytest.fixture(scope="module")
def real_data():
    if not os.path.exists(DATASET_PATH):
        pytest.skip(f"Dataset not found at {DATASET_PATH}")
    return pd.read_csv(DATASET_PATH, delimiter="\t")


@pytest.fixture(scope="module")
def fresh_dump_data():
    if not os.path.exists(FRESH_DUMP_PATH):
        pytest.skip(f"Fresh dump dataset not found at {FRESH_DUMP_PATH}")
    return pd.read_csv(FRESH_DUMP_PATH, delimiter="\t")


@pytest.fixture(scope="module")
def trained_model():
    """Load the pre-trained model"""
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def test_data():
    """Load test data from your preprocess script's output"""
    return {
        "X": np.load(f"{TEST_DATA_DIR}/X_test.npy"),
        "y": np.load(f"{TEST_DATA_DIR}/y_test.npy"),
    }
