import gdown
import joblib
import numpy as np
import pandas as pd
from libml import preprocessing as libml

url = "https://drive.google.com/file/d/1CJ1RQs7LSf1izuE-qPfyAZlLUd7K946U/view?usp=sharing"

downloaded_file = gdown.download(url, "downloaded_model.pkl", quiet=False, fuzzy=True)
print(f"Downloaded file: {downloaded_file}")

with open("../output/downloaded_model.pkl", "rb") as f:
    head = f.read(100)
    print(head[:100])

model = joblib.load("../output/downloaded_model.pkl")
messages = pd.read_csv("../datasets/a1_RestaurantReviews_HistoricDump.tsv", delimiter='\t', quoting=3)
X = libml._preprocess(messages).toarray()
exampleInput = X[0]  # Use the first example from the dataset
print(f"Example input: {exampleInput}")

prediction = model.predict(exampleInput.reshape(1, -1))  # Reshape to 2D array for prediction
print(f"Prediction: {prediction}")
