import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(filepath):
    data = pd.read_csv(filepath, encoding='ISO-8859-1')
    return data

def preprocess_descriptions(data, column_name='description'):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

    tfidf_matrix = vectorizer.fit_transform(data[column_name])
    return tfidf_matrix, vectorizer

if __name__ == '__main__':
    movies = load_data('netflix_titles.csv')
    tfidf_matrix, vectorizer = preprocess_descriptions(movies)
