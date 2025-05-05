import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score
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

    args = parser.parse_args()

    messages = pd.read_csv(args.dataset, delimiter='\t', quoting=3)
    X = libml._preprocess(messages).toarray()
    y = messages.iloc[:, -1].values

    if args.train_all:
        # Train on the full dataset
        classifier = GaussianNB()
        classifier.fit(X, y)
        joblib.dump(classifier, '../output/c2_Classifier_Sentiment_Model.pkl')
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=20)
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        joblib.dump(classifier, '../output/c2_Classifier_Sentiment_Model.pkl')

        # Evaluate the performance of the model
        eval_performance(classifier, X_test, y_test)


if __name__ == '__main__':
    main()
