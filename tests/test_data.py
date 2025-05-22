'''
Tests for features and data.
'''
import pandas as pd

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