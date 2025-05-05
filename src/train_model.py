import argparse
import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score
import nltk
nltk.download('wordnet')
from libml import preprocessing as libml


def eval_performance(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("Accuracy:", accuracy_score(y_test, y_pred))

def main():
    parser = argparse.ArgumentParser(description="Train a sentiment analysis model for restaurant reviews.")
    parser.add_argument('--dataset', type=str, default="../datasets/a1_RestaurantReviews_HistoricDump.tsv",
                        help="Path to the dataset (TSV file).")
    parser.add_argument('--train_all', action='store_true', help="Train on the full dataset without evaluation.")
    parser.add_argument('--output', type=str, default="../output/",
                        help="Path to save the trained model.")

    args = parser.parse_args()

    messages = pd.read_csv(args.dataset, delimiter='\t', quoting=3)
    X, cv = libml._preprocess(messages)
    y = messages.iloc[:, -1].values

    os.makedirs("output", exist_ok=True)
    bow_path = args.output + 'c1_BoW_Sentiment_Model.pkl'
    pickle.dump(cv, open(bow_path, "wb"))

    if args.train_all:
        # Train on the full dataset
        classifier = GaussianNB()
        classifier.fit(X, y)
        joblib.dump(classifier, args.output + 'c2_Classifier_Sentiment_Model.pkl')
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=20)
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        joblib.dump(classifier, args.output + 'c2_Classifier_Sentiment_Model.pkl')

        # Evaluate the performance of the model
        eval_performance(classifier, X_test, y_test)


if __name__ == '__main__':
    main()
