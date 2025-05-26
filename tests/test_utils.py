import os
import numpy as np
import json
import tempfile
import pandas as pd
import joblib
from sklearn.naive_bayes import GaussianNB
from src.train import save_json, save_split_data
from src.prepare_data import preprocess_and_save
from src import evaluate


def test_save_json_and_split_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test save_json
        data = {"a": 1, "b": [1, 2, 3]}
        json_path = os.path.join(tmpdir, "test.json")
        save_json(json_path, data)
        with open(json_path) as f:
            loaded = json.load(f)
        assert loaded == data

        # Test save_split_data
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        save_split_data(tmpdir, X, y)
        assert np.array_equal(np.load(os.path.join(tmpdir, "X_test.npy")), X)
        assert np.array_equal(np.load(os.path.join(tmpdir, "y_test.npy")), y)


def test_preprocess_and_save():
    # Create a fake dataset
    df = pd.DataFrame({"Review": ["good", "bad"], "Liked": [1, 0]})
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = os.path.join(tmpdir, "data.tsv")
        df.to_csv(dataset_path, sep="\t", index=False)
        output_dir = os.path.join(tmpdir, "out")
        bow_dir = os.path.join(tmpdir, "bow")
        X, y = preprocess_and_save(dataset_path, output_dir, bow_dir)
        assert X.shape[0] == 2
        assert set(y) == {0, 1}
        # Check files saved
        assert os.path.exists(os.path.join(output_dir, "X.npy"))
        assert os.path.exists(os.path.join(output_dir, "y.npy"))
        assert os.path.exists(os.path.join(bow_dir, "c1_BoW_Sentiment_Model.pkl"))


def test_evaluate_load_data_and_model_and_save_metrics():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy data
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        X_path = os.path.join(tmpdir, "X.npy")
        y_path = os.path.join(tmpdir, "y.npy")
        np.save(X_path, X)
        np.save(y_path, y)
        # Create and save dummy model
        model = GaussianNB()
        model.fit(X, y)
        model_path = os.path.join(tmpdir, "model.pkl")
        joblib.dump(model, model_path)
        # Test load_data
        X_loaded, y_loaded = evaluate.load_data(X_path, y_path)
        assert np.array_equal(X_loaded, X)
        assert np.array_equal(y_loaded, y)
        # Test load_model
        loaded_model = evaluate.load_model(model_path)
        assert hasattr(loaded_model, "predict")
        # Test evaluate_model
        metrics = evaluate.evaluate_model(loaded_model, X, y)
        assert "accuracy" in metrics and "f1_score" in metrics
        # Test save_metrics
        metrics_path = os.path.join(tmpdir, "metrics.json")
        out_path = evaluate.save_metrics(metrics, metrics_path)
        assert os.path.exists(out_path)
        with open(metrics_path) as f:
            loaded_metrics = json.load(f)
        assert loaded_metrics["accuracy"] == metrics["accuracy"]


def test_evaluate_run_evaluation():
    with tempfile.TemporaryDirectory() as tmpdir:
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        X_path = os.path.join(tmpdir, "X.npy")
        y_path = os.path.join(tmpdir, "y.npy")
        np.save(X_path, X)
        np.save(y_path, y)
        model = GaussianNB()
        model.fit(X, y)
        model_path = os.path.join(tmpdir, "model.pkl")
        joblib.dump(model, model_path)
        metrics_path = os.path.join(tmpdir, "metrics.json")
        metrics = evaluate.run_evaluation(X_path, y_path, model_path, metrics_path)
        assert os.path.exists(metrics_path)
        assert "accuracy" in metrics


def test_evaluate_error_cases():
    # Test missing files
    try:
        evaluate.load_data("missing_X.npy", "missing_y.npy")
    except FileNotFoundError:
        pass
    else:
        assert False, "Expected FileNotFoundError for missing files"
    try:
        evaluate.load_model("missing_model.pkl")
    except FileNotFoundError:
        pass
    else:
        assert False, "Expected FileNotFoundError for missing model"
    # Test empty data
    try:
        model = GaussianNB()
        evaluate.evaluate_model(model, np.array([]), np.array([]))
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for empty data"
