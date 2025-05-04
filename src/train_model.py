import argparse
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score

# This has to be in lib-ml
def preprocess_data(dataset):


    nltk.download('stopwords')
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    corpus = []

    for i in range(0, 900):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)

    return corpus

# This needs to be in lib-ml
def data_transform(corpus, dataset):
    cv = CountVectorizer(max_features=1420)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values
    # Saving BoW dictionary to later use in prediction
    import pickle
    bow_path = 'c1_BoW_Sentiment_Model.pkl'
    # pickle.dump(cv, open(bow_path, "wb"))
    return X, y


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

    dataset = pd.read_csv(args.dataset, delimiter='\t', quoting=3)

    # Preprocess the data
    corpus = preprocess_data(dataset)

    # Transform the data into BoW
    X, y = data_transform(corpus, dataset)

    if args.train_all:
        # Train on the full dataset
        classifier = GaussianNB()
        classifier.fit(X, y)
        joblib.dump(classifier, 'c2_Classifier_Sentiment_Model')
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=20)
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        joblib.dump(classifier, 'c2_Classifier_Sentiment_Model')

        # Evaluate the performance of the model
        eval_performance(classifier, X_test, y_test)


if __name__ == '__main__':
    main()
