# train_model.py
from libml.preprocessing import get_vectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

# Sample data
texts = ["The food was amazing!", "Terrible customer service."]
labels = [1, 0]  # 1 = positive, 0 = negative

# Create pipeline
pipeline = make_pipeline(get_vectorizer(), MultinomialNB())
pipeline.fit(texts, labels)

# Save model
joblib.dump(pipeline, "model.joblib")
